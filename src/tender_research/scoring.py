"""技术分评测与校准的核心组件：标签准入、连通分组、划分、系统特征、方法、指标与区间。

目标量尺统一为满分百分制 y = 100·value/maximum。所有拟合只用训练折，测试折标签只用于计算误差。
只依赖 NumPy 与标准库。
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .contracts import (
    EvalProtocol,
    PairSample,
    PairSet,
    ReportedResult,
    SystemPrediction,
)
from .score_model import PF1, fit_linear_model, linear_output, predict
from .score_text import knn_targets

# 任何协议都不能放行的特征名：事后信息。
DENIED_FEATURES = ("*official*", "*label*", "*known_at*", "*stage_result*", "*field_score*", "*cutoff*")
ANCHOR_FEATURES = ("anchor_pct", "anchor_agree", "anchor_n")
DIRECT_METHODS = {"raw": "raw_pct", "anchor": "anchor_pct"}


@dataclass
class Sample:
    """准入后的一份评测样本：官方标签（百分制）与该样本的系统预测记录。"""
    sample_id: str
    project_key: str
    template_group: str
    body_sha256: str
    region: str
    engineering_type: str
    scoring_scheme: str
    own_company: bool
    submitted_at: date
    known_at: date | None
    label_source: str
    label_verified: bool
    y: float
    maximum: float
    body_path: Path | None
    system_status: str
    raw_pct: float | None
    native_low_pct: float | None
    native_high_pct: float | None
    features: dict[str, float | None]
    system_items: list[dict] = field(default_factory=list)
    label_items: list[dict] = field(default_factory=list)
    group: int = -1


# ---------- 准入 ----------

DEFAULT_LABEL_SOURCES = ("official_document", "company_ledger")


def label_admission(record: PairSample, sources=DEFAULT_LABEL_SOURCES) -> tuple[bool, str]:
    """只接受已公布数值的技术总分，且来源类型在协议允许的 label_sources 内（此时待核验也可进入）；综合分不进入。"""
    if not record.eligible:
        return False, record.exclusion
    label = record.label
    if label is None or label.value is None:
        return False, "missing_label"
    if label.target != "technical_total":
        return False, "composite_not_technical"
    if label.source_kind not in sources:
        return False, "label_source_not_admitted"
    if label.value > label.maximum:
        return False, "score_out_of_range"
    return True, "eligible"


def prediction_admission(record: PairSample, prediction: SystemPrediction | None) -> tuple[bool, str]:
    """系统预测须存在且输入指纹与配对集一致；运行失败或拒绝估分照常进入评测（属于系统表现）。"""
    if prediction is None:
        return False, "prediction_missing"
    if prediction.sample_id != record.sample_id:
        return False, "prediction_sample_mismatch"
    tender = [f.sha256 for f in record.tender_files]
    body = record.body_file.sha256 if record.body_file else None
    if prediction.inputs.body_sha256 != body or sorted(prediction.inputs.tender_sha256) != sorted(tender):
        return False, "prediction_input_mismatch"
    return True, "eligible"


def load_pairs(path: Path) -> tuple[PairSet, list[PairSample], list[dict]]:
    pairs = PairSet.model_validate_json(path.read_text(encoding="utf-8"))
    records, invalid = [], []
    for row in pairs.samples:
        try:
            records.append(PairSample.model_validate(row))
        except ValidationError as error:
            invalid.append({"sample_id": row.get("sample_id") if isinstance(row, dict) else None,
                            "reason": "invalid_pair_record", "detail": error.errors(include_url=False)[0]["msg"]})
    ids = [r.sample_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("配对集样本编号重复")
    return pairs, records, invalid


def load_predictions(directory: Path) -> tuple[dict[str, tuple[SystemPrediction, dict]], list[dict]]:
    found, invalid = {}, []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            parsed = SystemPrediction.model_validate(raw)
        except ValidationError as error:
            invalid.append({"file": path.name, "reason": "prediction_invalid", "detail": error.errors(include_url=False)[0]["msg"]})
            continue
        if parsed.sample_id != path.stem:
            invalid.append({"file": path.name, "reason": "prediction_sample_mismatch"})
            continue
        found[parsed.sample_id] = (parsed, raw)
    return found, invalid


def connected_groups(records: list[PairSample]) -> dict[str, int]:
    """项目、模板组、相同正文 SHA 任一相同即并入同一连通组（并查集，含传递）。"""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(records):
        keys = [("project", r.project_key), ("template", r.template_group)]
        if r.body_file:
            keys.append(("body", r.body_file.sha256))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    numbering, groups = {}, {}
    for i, r in enumerate(records):
        groups[r.sample_id] = numbering.setdefault(root(i), len(numbering))
    return groups


def load_sources(pairs_paths: list[Path], prediction_dirs: list[Path], datasets_root: Path | None = None,
                 label_sources=DEFAULT_LABEL_SOURCES) -> dict:
    """合并多份配对集与多个预测目录（如批量回放与产品导出）：相同正文 SHA 只保留一份，优先保留标签可用的一份。"""
    records, invalid, projects, body_paths, pair_sets = [], [], {}, {}, []
    by_body, by_id = {}, {}
    for path in pairs_paths:
        pairs, found, bad = load_pairs(path)
        pair_sets.append(pairs)
        invalid += bad
        root = datasets_root or Path(pairs.datasets_root)
        root = root if root.is_absolute() else path.parent / root
        for key, value in pairs.projects.items():
            projects.setdefault(key, value)
        for r in found:
            body = r.body_file.sha256 if r.body_file else None
            if r.sample_id in by_id and (body is None or records[by_id[r.sample_id]].body_file is None
                                         or records[by_id[r.sample_id]].body_file.sha256 != body):
                raise ValueError(f"样本编号 {r.sample_id} 在不同来源指向不同正文")
            previous = by_body.get(body) if body else by_id.get(r.sample_id)
            if previous is not None:
                kept = records[previous]
                if not label_admission(kept, label_sources)[0] and label_admission(r, label_sources)[0]:
                    records[previous], r, kept = r, kept, r
                    by_id.pop(r.sample_id, None)
                    by_id[kept.sample_id] = previous
                    body_paths[kept.sample_id] = root / kept.body_file.path if kept.body_file else None
                invalid.append({"sample_id": r.sample_id, "reason": "duplicate_source", "kept": kept.sample_id})
                continue
            if body:
                by_body[body] = len(records)
            by_id[r.sample_id] = len(records)
            body_paths[r.sample_id] = root / r.body_file.path if r.body_file else None
            records.append(r)
    predictions, invalid_predictions = {}, []
    for directory in prediction_dirs:
        found, bad = load_predictions(directory)
        invalid_predictions += [{**b, "directory": str(directory)} for b in bad]
        for sid, item in found.items():
            record = records[by_id[sid]] if sid in by_id else None
            current = predictions.get(sid)
            if current is None or (record and not prediction_admission(record, current[0])[0]
                                   and prediction_admission(record, item[0])[0]):
                predictions[sid] = item
    return {"pair_sets": pair_sets, "records": records, "invalid": invalid, "projects": projects,
            "body_paths": body_paths, "predictions": predictions, "invalid_predictions": invalid_predictions}


def build_samples(records: list[PairSample], predictions: dict[str, tuple[SystemPrediction, dict]],
                  body_paths: dict[str, Path | None] | None = None,
                  label_sources=DEFAULT_LABEL_SOURCES) -> tuple[list[Sample], list[dict]]:
    groups = connected_groups(records)
    samples, excluded = [], []
    for r in records:
        ok, reason = label_admission(r, label_sources)
        pred = predictions.get(r.sample_id)
        if ok:
            ok, reason = prediction_admission(r, pred[0] if pred else None)
        if not ok:
            excluded.append({"sample_id": r.sample_id, "reason": reason})
            continue
        parsed = pred[0]
        p = parsed.prediction
        candidate = parsed.status == "succeeded" and p is not None and p.status == "candidate"
        scale = 100.0 / p.maximum if candidate else None
        samples.append(Sample(
            sample_id=r.sample_id, project_key=r.project_key, template_group=r.template_group,
            body_sha256=r.body_file.sha256, region=r.region, engineering_type=r.engineering_type,
            scoring_scheme=r.scoring_scheme, own_company=r.own_company, submitted_at=r.submitted_at,
            known_at=r.label.known_at, label_source=r.label.source_kind,
            label_verified=r.label.verification_status == "verified", y=100.0 * r.label.value / r.label.maximum, maximum=r.label.maximum,
            body_path=(body_paths or {}).get(r.sample_id),
            system_status="failed" if parsed.status == "failed" else (p.status if p else "no_total_prediction"),
            raw_pct=_scaled(p.estimate, scale) if p else None, native_low_pct=_scaled(p.low, scale) if p else None,
            native_high_pct=_scaled(p.high, scale) if p else None, features=pf1_features(parsed),
            system_items=[i.model_dump() for i in parsed.items],
            label_items=[i.model_dump() for i in r.label.items], group=groups[r.sample_id]))
    return samples, excluded


def _scaled(value, scale):
    return None if value is None or scale is None else value * scale


# ---------- 系统特征（pf-1） ----------

def pf1_features(parsed: SystemPrediction) -> dict[str, float | None]:
    """评分特征以产品后端写入的 artifact_extra.features（pf-1）为准；早期格式 B 无此字段时由顶层字段派生同名特征。"""
    extra = parsed.artifact_extra
    if extra is not None and extra.features:
        return {k: (None if v is None else float(v)) for k, v in extra.features.items()}
    f: dict[str, float | None] = dict.fromkeys(PF1)
    if parsed.status != "succeeded":
        return f
    p = parsed.prediction
    if p is not None and p.status == "candidate":
        f["raw_pct"] = 100.0 * p.estimate / p.maximum
        if p.low is not None and p.high is not None:
            f["low_pct"], f["high_pct"] = 100.0 * p.low / p.maximum, 100.0 * p.high / p.maximum
            f["width_pct"] = f["high_pct"] - f["low_pct"]
    items = parsed.items
    f["items_count"] = float(len(items))
    if items:
        weights = np.array([i.max_points if i.max_points and i.max_points > 0 else 1.0 for i in items])
        f["failed_share"] = float(np.mean([i.failed for i in items]))
        for name, key in (("coverage_mean", "coverage"), ("specificity_mean", "specificity")):
            pairs = [(w, v) for w, i in zip(weights, items) if isinstance(v := getattr(i, key, None), (int, float))]
            if pairs:
                f[name] = float(np.average([v for _, v in pairs], weights=[w for w, _ in pairs]))
                if name == "coverage_mean":
                    f["coverage_min"] = float(min(v for _, v in pairs))
        grades = [(w, 1 - i.grade_index / (i.grade_count - 1)) for w, i in zip(weights, items)
                  if isinstance(getattr(i, "grade_index", None), int) and isinstance(getattr(i, "grade_count", None), int)
                  and i.grade_count > 1]
        if grades:
            f["grade_mean"] = float(np.average([g for _, g in grades], weights=[w for w, _ in grades]))
        total = sum(i.max_points or 0 for i in items)
        if parsed.technical_full_score and total:
            f["max_ratio"] = total / parsed.technical_full_score
    if parsed.body:
        f["body_pages"] = _number(parsed.body.get("pages"))
        chars = _number(parsed.body.get("chars"))
        f["body_chars_log"] = math.log1p(chars) if chars is not None and chars >= 0 else None  # 与产品特征定义一致：ln(1+字数)
    if parsed.compliance:
        f["qualification_violated"] = _number(parsed.compliance.get("violated"))
        f["qualification_unclear"] = _number(parsed.compliance.get("unclear"))
    if parsed.findings:
        f["findings_consistency"] = _number(parsed.findings.get("consistency"))
        f["findings_scope"] = _number(parsed.findings.get("scope"))
    return f


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def feature_allowed(name: str, include: list[str], exclude: list[str]) -> bool:
    """可进入多特征模型的列：属于 pf-1、不在硬性禁止名单、匹配协议 include 且不匹配 exclude。"""
    if name not in PF1 or any(fnmatchcase(name, pattern) for pattern in DENIED_FEATURES):
        return False
    return any(fnmatchcase(name, p) for p in include) and not any(fnmatchcase(name, p) for p in exclude)


# ---------- 划分 ----------

def grouped_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """连通组整体分配到 k 折：组顺序由种子打乱，按组大小降序贪心放入当前样本最少的折。"""
    unique = np.unique(groups)
    k = max(1, min(k, len(unique)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    sizes = {g: int((groups == g).sum()) for g in unique}
    order = sorted(order, key=lambda g: -sizes[g])
    load, assign = np.zeros(k), {}
    for g in order:
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += sizes[g]
    return np.array([assign[g] for g in groups])


def assert_disjoint(samples: list[Sample], train: np.ndarray, test: np.ndarray):
    """任一划分都复核：项目、模板组、相同正文不得同时出现在训练与测试。"""
    def keys(idx):
        return {(kind, getattr(samples[i], name)) for i in idx
                for kind, name in (("p", "project_key"), ("t", "template_group"), ("b", "body_sha256"))}
    if keys(train) & keys(test) or set(map(int, train)) & set(map(int, test)):
        raise ValueError("划分存在项目／模板／正文泄漏")


def time_holdout(samples: list[Sample], fraction: float) -> dict:
    """提交日期最晚的连通组作最终测试；训练侧剔除跨越测试起点的组与当时尚未取得的标签。"""
    by_group = defaultdict(list)
    for i, s in enumerate(samples):
        by_group[s.group].append(i)
    if not by_group:
        return {"train": [], "test": [], "excluded": [], "cutoff": None}
    order = sorted(by_group, key=lambda g: (max(samples[i].submitted_at for i in by_group[g]),
                                            min(samples[i].sample_id for i in by_group[g])))
    n_test = max(1, math.ceil(fraction * len(order)))
    test = sorted(i for g in order[-n_test:] for i in by_group[g])
    cutoff = min(samples[i].submitted_at for i in test)
    train, excluded = [], []
    for g in order[:-n_test]:
        members = by_group[g]
        if any(samples[i].submitted_at >= cutoff for i in members):
            excluded += [{"sample_id": samples[i].sample_id, "reason": "group_crosses_time_boundary"} for i in members]
            continue
        for i in members:
            known = samples[i].known_at
            if known is not None and known > cutoff:
                excluded.append({"sample_id": samples[i].sample_id, "reason": "label_not_known_at_cutoff"})
            else:
                train.append(i)
    return {"train": sorted(train), "test": test, "excluded": excluded, "cutoff": cutoff.isoformat()}


# ---------- 方法 ----------

@dataclass
class Context:
    """一次评测的固定输入：样本、目标向量、pf-1 特征矩阵、文本相似度与协议。方法只读取训练折的目标。"""
    samples: list[Sample]
    y: np.ndarray
    protocol: EvalProtocol
    sim: np.ndarray
    bounds: tuple[float, float] = (0.0, 100.0)

    def __post_init__(self):
        self.names = list(PF1)
        self.x = np.array([[np.nan if s.features.get(n) is None else s.features[n] for n in self.names]
                           for s in self.samples], dtype=float).reshape(len(self.samples), len(self.names))
        f = self.protocol.features
        allowed = [n for n in self.names if feature_allowed(n, list(f.include), list(f.exclude))]
        self.columns = {"ridge": [n for n in allowed if n not in ANCHOR_FEATURES], "ridge_anchor": allowed}
        self.groups = np.array([s.group for s in self.samples])
        self.raw = np.array([np.nan if s.raw_pct is None else s.raw_pct for s in self.samples])

    def matrix(self, columns: list[str], tr, te, use_knn: bool):
        idx = [self.names.index(c) for c in columns]
        x_tr, x_te = self.x[np.ix_(tr, idx)], self.x[np.ix_(te, idx)]
        if not use_knn:
            return list(columns), x_tr, x_te
        k = self.protocol.knn.k
        k_tr = knn_targets(self.sim, self.y, tr, tr, self.groups, k)
        k_te = knn_targets(self.sim, self.y, te, tr, self.groups, k)
        return [*columns, "knn"], np.column_stack([x_tr, k_tr]), np.column_stack([x_te, k_te])

    def clip(self, values):
        return np.clip(values, *self.bounds)


def pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """加权保序回归（池相邻违反者），相同 x 先合并。返回各块的 x 均值与拟合值。"""
    ux, inverse = np.unique(x, return_inverse=True)
    w = np.bincount(inverse).astype(float)
    v = np.bincount(inverse, weights=y) / w
    blocks = []
    for xi, vi, wi in zip(ux, v, w):
        blocks.append([vi * wi, wi, xi * wi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, sx = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
            blocks[-1][2] += sx
    return np.array([b[2] / b[1] for b in blocks]), np.array([b[0] / b[1] for b in blocks])


def _direct_model(ctx: Context, tr, feature: str) -> dict:
    """单一 pf-1 特征直接作预测（系数 1）；缺失取训练折中位数，训练折全缺失时取训练目标均值。"""
    column = ctx.x[tr, ctx.names.index(feature)]
    present = column[~np.isnan(column)]
    fill = float(np.median(present)) if len(present) else float(ctx.y[tr].mean())
    return {"features": [feature], "standardize": {"mean": {feature: 0.0}, "scale": {feature: 1.0}},
            "impute": {feature: fill}, "coefficients": {feature: 1.0}, "intercept": 0.0, "isotonic": None,
            "clip": list(ctx.bounds)}


def _ridge_model(ctx: Context, tr, columns, seed, use_knn) -> tuple[dict, list[str]]:
    """岭回归；alpha 在训练折内用嵌套分组交叉验证从协议候选中选 MAE 最低者（并列取较大 alpha）。"""
    alphas = list(ctx.protocol.ridge_alphas)
    best, inner_mae = alphas[0], {}
    if len(alphas) > 1 and len(np.unique(ctx.groups[tr])) >= 3:
        folds = grouped_folds(ctx.groups[tr], ctx.protocol.cv.folds, seed)
        errors = {a: [] for a in alphas}
        for f in np.unique(folds):
            itr, ite = tr[folds != f], tr[folds == f]
            names, x_tr, x_te = ctx.matrix(columns, itr, ite, use_knn)
            for a, model in fit_linear_model(names, x_tr, ctx.y[itr], alphas, ctx.bounds).items():
                errors[a].extend(np.abs(predict(model, names, x_te) - ctx.y[ite]))
        inner_mae = {a: float(np.mean(e)) for a, e in errors.items()}
        best = min(alphas, key=lambda a: (round(inner_mae[a], 9), -a))
    names, x_tr, _ = ctx.matrix(columns, tr, tr[:0], use_knn)
    model = fit_linear_model(names, x_tr, ctx.y[tr], [best], ctx.bounds)[best]
    model["alpha"] = best
    model["inner_cv_mae_by_alpha"] = {str(a): m for a, m in inner_mae.items()}
    return model, names


def fit_model(method: str, ctx: Context, tr: np.ndarray, seed: int) -> tuple[dict, list[str]]:
    """拟合一个以 pf-1 特征为输入的模型（可导出为 scoring-model/1），返回模型与其输入列名。"""
    if method in DIRECT_METHODS:
        return _direct_model(ctx, tr, DIRECT_METHODS[method]), list(ctx.names)
    if method == "linear":
        names, x_tr, _ = ctx.matrix(["raw_pct"], tr, tr[:0], False)
        return fit_linear_model(names, x_tr, ctx.y[tr], [0.0], ctx.bounds)[0.0], names
    if method == "isotonic":
        model = _direct_model(ctx, tr, "raw_pct")
        known = ~np.isnan(ctx.raw[tr])
        if known.sum() >= 2:
            bx, by = pav(ctx.raw[tr][known], ctx.y[tr][known])
            model["isotonic"] = {"x": bx.tolist(), "y": by.tolist()}
        return model, list(ctx.names)
    if method in ("ridge", "ridge_anchor", "ridge_knn", "ridge_isotonic"):
        use_knn = method == "ridge_knn"
        columns = ctx.columns["ridge_anchor" if method == "ridge_anchor" else "ridge"]
        model, names = _ridge_model(ctx, tr, columns, seed, use_knn)
        if method == "ridge_isotonic" and len(np.unique(ctx.groups[tr])) >= 2:
            oof = np.full(len(tr), np.nan)
            folds = grouped_folds(ctx.groups[tr], ctx.protocol.cv.folds, seed + 3)
            for f in np.unique(folds):
                mask = folds == f
                inner_names, x_itr, x_ite = ctx.matrix(columns, tr[~mask], tr[mask], False)
                inner = fit_linear_model(inner_names, x_itr, ctx.y[tr[~mask]], [model["alpha"]], ctx.bounds)
                oof[mask] = linear_output(inner[model["alpha"]], inner_names, x_ite)
            bx, by = pav(oof, ctx.y[tr])
            model["isotonic"] = {"x": bx.tolist(), "y": by.tolist()}
        return model, names
    raise ValueError(f"未知方法 {method}")


def fit_predict(method: str, ctx: Context, tr: np.ndarray, te: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """在训练折拟合方法并预测测试折；可导出方法的参数即 scoring-model/1 模型本体，预测走同一公式。"""
    y = ctx.y[tr]
    mean = float(y.mean())
    if method == "mean":
        return np.full(len(te), mean), {"value": mean}
    if method == "median":
        median = float(np.median(y))
        return np.full(len(te), median), {"value": median}
    if method == "scheme_mean":
        by = defaultdict(list)
        for i in tr:
            by[ctx.samples[i].scoring_scheme].append(ctx.y[i])
        table = {k: float(np.mean(v)) for k, v in by.items()}
        return np.array([table.get(ctx.samples[i].scoring_scheme, mean) for i in te]), {"by_scheme": table, "fallback": mean}
    if method == "knn":
        pred = knn_targets(ctx.sim, ctx.y, te, tr, ctx.groups, ctx.protocol.knn.k)
        return ctx.clip(np.where(np.isnan(pred), mean, pred)), {**_knn_params(ctx, tr), "fallback": mean}
    model, names = fit_model(method, ctx, tr, seed)
    if method == "ridge_knn":
        names, _, x_te = ctx.matrix(ctx.columns["ridge"], tr, te, True)
        model["knn"] = _knn_params(ctx, tr)
        return predict(model, names, x_te), model
    return predict(model, names, ctx.x[te]), model


def _knn_params(ctx: Context, tr) -> dict:
    return {"k": ctx.protocol.knn.k, "ngrams": ctx.protocol.knn.ngrams, "hash_bits": ctx.protocol.knn.hash_bits,
            "neighbours": [{"sample_id": ctx.samples[i].sample_id, "body_sha256": ctx.samples[i].body_sha256,
                            "target": float(ctx.y[i])} for i in tr if not np.isnan(ctx.sim[i, i])]}


def inner_oof(method: str, ctx: Context, tr: np.ndarray, seed: int) -> np.ndarray:
    """训练折内的分组折外预测，用于分组 split-conformal 残差；只有一个组时无残差。"""
    out = np.full(len(tr), np.nan)
    if len(np.unique(ctx.groups[tr])) < 2:
        return out
    folds = grouped_folds(ctx.groups[tr], ctx.protocol.cv.folds, seed)
    for f in np.unique(folds):
        mask = folds == f
        out[mask] = fit_predict(method, ctx, tr[~mask], tr[mask], seed + 7)[0]
    return out


def learning_curve(ctx: Context, idx: np.ndarray, methods, fractions) -> list[dict]:
    """学习曲线：固定第一次重复的分组折，每折训练组按种子排列后取前 25%/50%/… 组（嵌套子集），测试折不变。"""
    p = ctx.protocol
    folds = grouped_folds(ctx.groups[idx], p.cv.folds, p.seed)
    rows = []
    for fraction in sorted(fractions):
        errors, train_sizes, group_sizes = {m: [] for m in methods}, [], []
        for f in np.unique(folds):
            tr_all, te = idx[folds != f], idx[folds == f]
            groups = np.random.default_rng(p.seed + int(f)).permutation(np.unique(ctx.groups[tr_all]))
            keep = groups[:max(2, math.ceil(fraction * len(groups)))]
            tr = tr_all[np.isin(ctx.groups[tr_all], keep)]
            train_sizes.append(len(tr))
            group_sizes.append(len(keep))
            for m in methods:
                errors[m].extend(np.abs(fit_predict(m, ctx, tr, te, p.seed)[0] - ctx.y[te]))
        rows.append({"fraction": fraction, "train_samples_mean": float(np.mean(train_sizes)),
                     "train_groups_mean": float(np.mean(group_sizes)),
                     "mae": {m: float(np.mean(e)) for m, e in errors.items()}})
    return rows


def conformal_quantile(abs_residuals: np.ndarray, level: float) -> float:
    """有限样本修正的分位数：第 ⌈(n+1)·level⌉ 小的绝对残差；不足时区间为全量尺。"""
    r = np.sort(abs_residuals[~np.isnan(abs_residuals)])
    rank = math.ceil((len(r) + 1) * level)
    return float(r[rank - 1]) if 0 < rank <= len(r) else math.inf


def exceed_probability(residuals: np.ndarray, prediction: float, threshold: float) -> float:
    """保形预测分布下 P(y ≥ 阈值)：训练折外有符号残差中满足 pred+r ≥ 阈值 的比例（加 0.5 平滑）。"""
    r = residuals[~np.isnan(residuals)]
    return float((np.sum(prediction + r >= threshold) + 0.5) / (len(r) + 1))


def evaluate_split(ctx: Context, tr: np.ndarray, te: np.ndarray, methods, seed: int) -> dict:
    """单次训练／测试：各方法的测试预测、训练折内折外有符号残差与拟合参数。"""
    assert_disjoint(ctx.samples, tr, te)
    out = {}
    for m in methods:
        pred, params = fit_predict(m, ctx, tr, te, seed)
        residuals = ctx.y[tr] - inner_oof(m, ctx, tr, seed + 1)
        out[m] = {"pred": pred, "residuals": residuals, "params": params}
    return out


# ---------- 指标 ----------

def error_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    if not len(y) or len(y) != len(pred) or not np.isfinite(pred).all():
        raise ValueError("评测要求非空、等长、有限的预测")
    e = pred - y
    a = np.abs(e)
    return {"n": len(y), "mae": float(a.mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "abs_p50": float(np.percentile(a, 50)), "abs_p90": float(np.percentile(a, 90))}


def cluster_bootstrap_gain(abs_base: np.ndarray, abs_method: np.ndarray, groups: np.ndarray, resamples: int,
                           confidence: float, seed: int) -> dict:
    """MAE 增益（基线 MAE − 方法 MAE）及按连通组重采样的百分位区间。"""
    d = abs_base - abs_method
    _, inverse = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=d), np.bincount(inverse).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(sums), size=(resamples, len(sums)))
    gains = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - confidence) / 2 * 100
    return {"gain": float(d.mean()), "low": float(np.percentile(gains, tail)),
            "high": float(np.percentile(gains, 100 - tail)), "resamples": resamples, "confidence": confidence,
            "groups": len(sums)}


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values), dtype=float)
    for v in np.unique(values):
        tie = values == v
        ranks[tie] = ranks[tie].mean()
    return ranks


def ranking_metrics(samples: list[Sample], idx: np.ndarray, y: np.ndarray, pred: np.ndarray, min_projects: int) -> dict:
    """同场排名：项目内 ≥2 份有标签样本；Spearman 均值、两两排序正确率（预测并列计 0.5）、最高分命中率。"""
    by = defaultdict(list)
    for pos, i in enumerate(idx):
        by[samples[i].project_key].append(pos)
    projects = [p for p in by.values() if len(p) >= 2]
    if len(projects) < min_projects:
        return {"status": "not_evaluable", "projects": len(projects), "reason": f"可比项目数 < {min_projects}"}
    rhos, correct, pairs, hits = [], 0.0, 0, []
    for members in projects:
        yt, pt = y[members], pred[members]
        if np.ptp(yt) > 0 and np.ptp(pt) > 0:
            rhos.append(float(np.corrcoef(rankdata(yt), rankdata(pt))[0, 1]))
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                if yt[a] == yt[b]:
                    continue
                pairs += 1
                diff = (pt[a] - pt[b]) * (yt[a] - yt[b])
                correct += 1.0 if diff > 0 else 0.5 if pt[a] == pt[b] else 0.0
        top_pred = np.flatnonzero(pt == pt.max())
        hits.append(float(np.isin(top_pred, np.flatnonzero(yt == yt.max())).mean()))
    return {"status": "evaluated", "projects": len(projects), "samples": sum(map(len, projects)),
            "spearman_mean": float(np.mean(rhos)) if rhos else None, "spearman_projects": len(rhos),
            "pairwise_accuracy": correct / pairs if pairs else None, "pairs": pairs, "top1_hit_rate": float(np.mean(hits))}


def probability_metrics(prob: np.ndarray, outcome: np.ndarray, min_n: int, bins: int = 5) -> dict:
    """入围概率：Brier、对数损失、AUC 与常数基率对照、等宽校准分箱。"""
    n = len(prob)
    if n < min_n:
        return {"status": "not_evaluable", "n": n, "reason": f"有入围线的样本 < {min_n}"}
    o = outcome.astype(float)
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    base = float(o.mean())
    b = np.clip(base, 1e-6, 1 - 1e-6)
    pos, neg = prob[o == 1], prob[o == 0]
    auc = None
    if len(pos) and len(neg):
        auc = float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    table = [{"range": [float(edges[k]), float(edges[k + 1])], "n": int((which == k).sum()),
              "mean_probability": float(prob[which == k].mean()) if (which == k).any() else None,
              "observed_rate": float(o[which == k].mean()) if (which == k).any() else None} for k in range(bins)]
    return {"status": "evaluated", "n": n, "positives": int(o.sum()), "brier": float(((prob - o) ** 2).mean()),
            "log_loss": float(-(o * np.log(p) + (1 - o) * np.log(1 - p)).mean()), "auc": auc,
            "base_rate": base, "base_rate_brier": float(((base - o) ** 2).mean()),
            "base_rate_log_loss": float(-(o * np.log(b) + (1 - o) * np.log(1 - b)).mean()), "calibration_bins": table}


def field_percentile(value: float, field_values: np.ndarray) -> float:
    return float(((field_values < value).sum() + 0.5 * (field_values == value).sum()) / len(field_values))


def _normalize_item(text: str) -> str:
    return re.sub(r"[\W_]+", "", text or "").lower()


def _bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)} or ({text} if text else set())


def match_items(system_items: list[dict], label_items: list[dict], threshold: float) -> tuple[list[tuple[dict, dict, float]], int]:
    """官方分项（多评委按分项合并取均值）与系统分项按字符二元组 Jaccard 一对一贪心匹配。"""
    merged = {}
    for item in label_items:
        if item.get("value") is None:
            continue
        key = _normalize_item(item["item"])
        entry = merged.setdefault(key, {"item": item["item"], "maximum": item["maximum"], "values": []})
        entry["values"].append(100.0 * item["value"] / item["maximum"])
    labels = list(merged.values())
    candidates = []
    for a, s in enumerate(system_items):
        sa = _bigrams(_normalize_item(s.get("statement", "")))
        for b, lab in enumerate(labels):
            sb = _bigrams(_normalize_item(lab["item"]))
            sim = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
            if sim >= threshold:
                candidates.append((sim, a, b))
    used_a, used_b, matches = set(), set(), []
    for sim, a, b in sorted(candidates, key=lambda t: (-t[0], t[1], t[2])):
        if a not in used_a and b not in used_b:
            used_a.add(a); used_b.add(b)
            matches.append((system_items[a], labels[b], sim))
    return matches, len(labels)


# ---------- 档案就绪核验 ----------

def audit_archive(datasets: Path, results: Path) -> dict:
    """已归档案例的评分就绪情况：用户报送单独统计；official_results 中来源在默认 label_sources 内、已公布的技术总分计入可评测配对，另计已核验官方原件数。"""
    cases = []
    for path in sorted(datasets.glob("CASE-*/资料清单.json")):
        data = json.loads(path.read_text())
        assessment = data.get("assessment", {})
        files = {r["file_id"]: r for r in data["file_records"]}
        reports = [ReportedResult.model_validate(r) for r in assessment.get("reported_results", [])]
        official = assessment.get("official_results", [])
        for r in [*reports, *official]:
            fid, sha = (r.submission_file_id, r.submission_sha256) if isinstance(r, ReportedResult) else (
                r.get("submission_file_id"), r.get("submission_sha256"))
            if fid not in files or files[fid]["sha256"] != sha or files[fid]["kind"] != "技术标":
                raise ValueError("成绩未关联登记的技术标原件版本")
        admitted = [r for r in official if r.get("target") == "technical_total" and r.get("value") is not None
                    and r.get("source_kind") in DEFAULT_LABEL_SOURCES and 0 <= r["value"] <= r.get("maximum", 0)]
        verified = [r for r in admitted if r.get("source_kind") == "official_document" and r.get("verification_status") == "verified"]
        cases.append({"case_id": data["case_id"], "project_id": data["tender_id"],
                      "submitted_body_files": [r["file_id"] for r in data["file_records"] if r["kind"] == "技术标"],
                      "recorded_technical_max": assessment.get("max_scores", {}).get("technical"),
                      "user_reported_results": len(reports),
                      "user_reported_technical_results": sum(r.target == "technical_total" for r in reports),
                      "official_results": len(official), "admissible_technical_results": len(admitted),
                      "verified_official_technical_results": len(verified),
                      "eligible_for_training": bool(admitted),
                      "exclusion": None if admitted else "no_admissible_technical_label"})
    responses = [json.loads(p.read_text()) for p in results.glob("llm-*/*.response.json")]
    received = sum(r.get("status") == "received" for r in responses)
    pairs = sum(c["admissible_technical_results"] for c in cases)
    verified_pairs = sum(c["verified_official_technical_results"] for c in cases)
    projects = len({c["project_id"] for c in cases if c["eligible_for_training"]})
    minimums = EvalProtocol.model_fields["minimums"].default
    ready = pairs >= minimums.labeled_samples and projects >= minimums.independent_projects
    return {"experiment_id": "score-readiness",
            "status": "ready_for_score_eval" if ready else "independent_official_scoring_not_evaluable",
            "archived_cases": cases, "eligible_real_score_pairs": pairs, "verified_official_pairs": verified_pairs,
            "independent_projects": projects, "label_sources": list(DEFAULT_LABEL_SOURCES),
            "user_reported_technical_pairs": sum(c["user_reported_technical_results"] for c in cases),
            "trained_on_real_cases": False, "model_inference_calls": received,
            "reason": (f"可评测技术总分配对 {pairs} 份（其中已核验官方原件 {verified_pairs} 份）、独立项目 {projects} 个"
                       f"（评测下限 {minimums.labeled_samples} 份、{minimums.independent_projects} 个）；用户报送结果不计入"),
            "llm_route": "actual_responses_archived" if received else "not_run",
            "ranking_route": "score_eval_within_project_ranking"}

