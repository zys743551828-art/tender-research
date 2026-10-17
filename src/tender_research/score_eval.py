"""技术分评测与校准：用评分配对集（格式 A）与系统批量预测（格式 B）在冻结协议下比较方法并给出结论。

python -m tender_research.score_eval --pairs 评分配对集.json --predictions <格式B目录> --protocol 评测协议.json \
    --name <实验名> [--mode cv|holdout]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import paths
from .contracts import BASELINES, EXPORTABLE, EvalProtocol
from .score_model import METHOD_LABELS, PF1, canonical, model_id
from .score_text import body_text, ngram_counts, similarity_matrix
from .scoring import (
    Context,
    build_samples,
    cluster_bootstrap_gain,
    conformal_quantile,
    error_metrics,
    evaluate_split,
    exceed_probability,
    field_percentile,
    fit_model,
    grouped_folds,
    inner_oof,
    learning_curve,
    load_predictions,
    load_sources,
    match_items,
    prediction_admission,
    probability_metrics,
    ranking_metrics,
    time_holdout,
)

TEXT_METHODS = ("knn", "ridge_knn")
REGISTRY = "score-eval-holdout-registry.json"
METHOD_NAMES = {"mean": "训练均值", "median": "训练中位数", "scheme_mean": "评分办法分组均值", "raw": "系统原始估计",
                "linear": "线性校准", "isotonic": "保序回归校准", "anchor": "锚定对比估计",
                "ridge": "岭回归（系统特征）", "ridge_anchor": "岭回归＋锚定特征", "ridge_isotonic": "岭回归＋保序校准",
                "knn": "正文近邻", "ridge_knn": "岭回归＋正文近邻"}
SLICE_NAMES = {"scoring_scheme": "评分办法", "engineering_type": "工程类型", "own_company": "本公司",
               "region": "地区", "system_status": "系统状态"}
STATUS_NAMES = {"passed": "通过", "not_passed": "未通过", "not_evaluable": "不可评价"}
SOURCE_NAMES = {"official_document": "官方原件", "company_ledger": "公司台账", "user_statement": "口头报送"}
TEXT_NAMES = {"cached": "已缓存", "extracted": "本次抽取", "body_pdf_missing": "原件缺失", "body_sha256_mismatch": "校验值不符"}
VALUE_NAMES = {"True": "是", "False": "否", "candidate": "给出估计", "abstained": "拒绝估分", "failed": "运行失败",
               "no_total_prediction": "无总分预测"}
REASON_NAMES = {"invalid_pair_record": "配对记录不合格式", "missing_label": "无官方分或未公布",
                "composite_not_technical": "综合总分（非技术分）",
                "score_out_of_range": "分数超满分", "no_text_layer": "正文无文字层", "duplicate_body": "重复正文",
                "missing_files": "缺文件", "prediction_missing": "缺系统预测", "prediction_sample_mismatch": "预测样本编号不符",
                "prediction_input_mismatch": "预测输入指纹与配对集不符",
                "duplicate_source": "多来源重复（相同正文）",
                "label_source_not_admitted": "标签来源不在协议准入范围"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_digest(directory: Path) -> str:
    rows = [f"{p.name}:{sha256_file(p)}" for p in sorted(directory.glob("*.json"))]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def code_digest() -> dict:
    here = Path(__file__).parent
    return {name: sha256_file(here / name) for name in ("score_eval.py", "scoring.py", "score_model.py", "score_text.py", "contracts.py")}


# ---------- 评测循环 ----------

def empty_store(methods, levels, repeats, n):
    return {"pred": {m: np.full((repeats, n), np.nan) for m in methods},
            "lo": {m: {q: np.full((repeats, n), np.nan) for q in levels} for m in methods},
            "hi": {m: {q: np.full((repeats, n), np.nan) for q in levels} for m in methods},
            "prob": {m: np.full((repeats, n), np.nan) for m in methods},
            "fold": np.full((repeats, n), -1)}


def record_split(store, ctx, out, positions, te, r, levels, cutoffs):
    for m, result in out.items():
        pred = result["pred"]
        store["pred"][m][r, positions] = pred
        abs_res = np.abs(result["residuals"])
        for q in levels:
            width = conformal_quantile(abs_res, q)
            store["lo"][m][q][r, positions] = np.clip(pred - width, *ctx.bounds)
            store["hi"][m][q][r, positions] = np.clip(pred + width, *ctx.bounds)
        for pos, i, p in zip(positions, te, pred):
            if not np.isnan(cutoffs[i]):
                store["prob"][m][r, pos] = exceed_probability(result["residuals"], p, cutoffs[i])


def run_cv(ctx: Context, idx: np.ndarray, methods, cutoffs) -> dict:
    """重复分组 K 折：每次重复用不同种子重新分组成折，组不跨折。"""
    p = ctx.protocol
    store = empty_store(methods, p.interval_levels, p.cv.repeats, len(idx))
    for r in range(p.cv.repeats):
        folds = grouped_folds(ctx.groups[idx], p.cv.folds, p.seed + 1000 * r)
        store["fold"][r] = folds
        for f in np.unique(folds):
            positions = np.flatnonzero(folds == f)
            tr, te = idx[folds != f], idx[positions]
            out = evaluate_split(ctx, tr, te, methods, p.seed + 1000 * r + 17 * int(f))
            record_split(store, ctx, out, positions, te, r, p.interval_levels, cutoffs)
    store["idx"] = idx
    return store


def run_holdout(ctx: Context, train: np.ndarray, test: np.ndarray, methods, cutoffs) -> tuple[dict, dict]:
    p = ctx.protocol
    store = empty_store(methods, p.interval_levels, 1, len(test))
    out = evaluate_split(ctx, train, test, methods, p.seed)
    record_split(store, ctx, out, np.arange(len(test)), test, 0, p.interval_levels, cutoffs)
    store["fold"][0] = 0
    store["idx"] = test
    return store, out


def summarize_methods(ctx: Context, store: dict, methods) -> dict:
    idx, repeats = store["idx"], store["fold"].shape[0]
    y = ctx.y[idx]
    out = {}
    for m in methods:
        pred = store["pred"][m]
        pooled = error_metrics(np.tile(y, repeats), pred.ravel())
        pooled["mae_by_repeat"] = [float(np.abs(pred[r] - y).mean()) for r in range(repeats)]
        out[m] = pooled
    return out


def select_method(protocol: EvalProtocol, selection_metrics: dict, methods) -> str:
    if protocol.selection != "cv_mae":
        return protocol.selection
    return min((m for m in methods if m in EXPORTABLE), key=lambda m: (selection_metrics[m]["mae"], methods.index(m)))


def interval_summary(ctx: Context, store: dict, methods, levels) -> dict:
    y = ctx.y[store["idx"]]
    out = {}
    for m in methods:
        out[m] = {}
        for q in levels:
            lo, hi = store["lo"][m][q], store["hi"][m][q]
            out[m][str(q)] = {"nominal": q, "coverage": float(((y >= lo) & (y <= hi)).mean()),
                              "mean_width": float((hi - lo).mean())}
    return out


def native_interval(samples, idx) -> dict:
    rows = [(samples[i].y, samples[i].native_low_pct, samples[i].native_high_pct) for i in idx
            if samples[i].native_low_pct is not None and samples[i].native_high_pct is not None]
    if not rows:
        return {"n": 0, "coverage": None, "mean_width": None}
    y, lo, hi = map(np.array, zip(*rows))
    return {"n": len(rows), "coverage": float(((y >= lo) & (y <= hi)).mean()), "mean_width": float((hi - lo).mean())}


def slice_summary(ctx: Context, store, methods, fields, minimum) -> dict:
    idx = store["idx"]
    out = {}
    for name in fields:
        values = defaultdict(list)
        for pos, i in enumerate(idx):
            values[str(getattr(ctx.samples[i], name))].append(pos)
        out[name] = {}
        for value, positions in sorted(values.items()):
            if len(positions) < minimum:
                out[name][value] = {"n": len(positions), "status": "not_evaluable"}
                continue
            y = ctx.y[idx[positions]]
            out[name][value] = {"n": len(positions), "status": "evaluated",
                                "mae": {m: float(np.abs(store["pred"][m][:, positions] - y).mean()) for m in methods}}
    return out


def gain_against_baseline(ctx, store, method, baseline) -> dict:
    p = ctx.protocol
    idx = store["idx"]
    y = ctx.y[idx]
    abs_base = np.abs(store["pred"][baseline] - y).mean(axis=0)
    abs_method = np.abs(store["pred"][method] - y).mean(axis=0)
    return {"method": method, "baseline": baseline,
            **cluster_bootstrap_gain(abs_base, abs_method, ctx.groups[idx], p.bootstrap_resamples, p.confidence, p.seed)}


def evaluable(samples, idx, n_min, projects_min) -> str | None:
    n, projects = len(idx), len({samples[i].project_key for i in idx})
    if n < n_min or projects < projects_min:
        return f"有标签样本 {n} 份（下限 {n_min}）、独立项目 {projects} 个（下限 {projects_min}）"
    return None


# ---------- 附加评测 ----------

def field_arrays(samples, projects) -> tuple[np.ndarray, list[np.ndarray | None]]:
    cutoffs, fields = np.full(len(samples), np.nan), []
    for i, s in enumerate(samples):
        project = projects.get(s.project_key)
        if project and project.shortlist_cutoff is not None:
            cutoffs[i] = 100.0 * project.shortlist_cutoff / s.maximum
        known = [v for v in (project.field_scores or []) if v is not None] if project else []
        fields.append(np.array(known) * 100.0 / s.maximum if len(known) >= 2 else None)
    return cutoffs, fields


def shortlist_summary(ctx, store, methods, cutoffs) -> dict:
    idx = store["idx"]
    positions = np.flatnonzero(~np.isnan(cutoffs[idx]))
    outcome = ctx.y[idx[positions]] >= cutoffs[idx[positions]]
    return {m: probability_metrics(np.nanmean(store["prob"][m][:, positions], axis=0) if len(positions) else np.zeros(0),
                                   outcome, ctx.protocol.minimums.shortlist_samples) for m in methods}


def percentile_summary(ctx, store, methods, fields) -> dict:
    idx = store["idx"]
    positions = [pos for pos, i in enumerate(idx) if fields[i] is not None]
    if len(positions) < ctx.protocol.minimums.slice_samples:
        return {"status": "not_evaluable", "n": len(positions), "reason": f"有同场成绩的样本 < {ctx.protocol.minimums.slice_samples}"}
    actual = np.array([field_percentile(ctx.y[idx[p]], fields[idx[p]]) for p in positions])
    out = {"status": "evaluated", "n": len(positions), "mae_percentile_points": {}}
    for m in methods:
        mean_pred = store["pred"][m][:, positions].mean(axis=0)
        predicted = np.array([field_percentile(v, fields[idx[p]]) for v, p in zip(mean_pred, positions)])
        out["mae_percentile_points"][m] = float(100 * np.abs(predicted - actual).mean())
    return out


def relative_target(ctx, fields, methods, mode, split) -> dict:
    """项目相对目标：本样本百分制减同场中位数，剔除评委宽严等项目级效应后重跑方法比较。"""
    p = ctx.protocol
    y_rel = np.array([s.y - np.median(f) if f is not None else np.nan for s, f in zip(ctx.samples, fields)])
    rel = Context(ctx.samples, y_rel, p, ctx.sim, bounds=(-100.0, 100.0))
    methods = [m for m in methods if m not in ("raw", "anchor")]
    subset = np.flatnonzero(~np.isnan(y_rel))
    nan_cut = np.full(len(ctx.samples), np.nan)
    if mode == "cv":
        reason = evaluable(ctx.samples, subset, p.minimums.labeled_samples, p.minimums.independent_projects)
        if reason:
            return {"status": "not_evaluable", "reason": reason}
        store = run_cv(rel, subset, methods, nan_cut)
    else:
        train = np.array([i for i in split["train"] if i in set(subset)], dtype=int)
        test = np.array([i for i in split["test"] if i in set(subset)], dtype=int)
        if len(test) < p.holdout.min_test_samples or len(np.unique(rel.groups[train])) < 2:
            return {"status": "not_evaluable", "reason": f"留出测试中有同场成绩的样本 {len(test)} 份"}
        store = run_holdout(rel, train, test, methods, nan_cut)[0]
    metrics = summarize_methods(rel, store, methods)
    baseline = min((m for m in methods if m in BASELINES), key=lambda m: metrics[m]["mae"])
    best = min((m for m in methods if m not in BASELINES), key=lambda m: metrics[m]["mae"])
    return {"status": "evaluated", "n": len(store["idx"]), "metrics": metrics, "best_baseline": baseline,
            "best_method": best, "gain": gain_against_baseline(rel, store, best, baseline)}


def item_summary(samples, threshold, minimum) -> dict:
    """分项层：系统分项与官方分项匹配后的百分制误差；评委间离散度作为可达精度参照。"""
    label_items = matched = 0
    errors, spreads, max_agree = [], [], []
    for s in samples:
        if not s.label_items:
            continue
        matches, n_labels = match_items(s.system_items, s.label_items, threshold)
        label_items += n_labels
        matched += len(matches)
        for system, label, _ in matches:
            if len(label["values"]) >= 2:
                spreads.append(float(np.std(label["values"])))
            if system.get("max_points"):
                max_agree.append(abs(system["max_points"] - label["maximum"]) < 1e-9)
                if system.get("estimate") is not None:
                    errors.append(100.0 * system["estimate"] / system["max_points"] - float(np.mean(label["values"])))
    out = {"label_items": label_items, "matched": matched,
           "match_rate": matched / label_items if label_items else None,
           "max_points_agreement": float(np.mean(max_agree)) if max_agree else None,
           "evaluator_spread_mean_sd": float(np.mean(spreads)) if spreads else None}
    if len(errors) < minimum:
        return {**out, "status": "not_evaluable", "n": len(errors), "reason": f"匹配且有估计的分项 < {minimum}"}
    e = np.array(errors)
    return {**out, "status": "evaluated", "n": len(e), "mae_percent_of_item_max": float(np.abs(e).mean()),
            "bias": float(e.mean()), "abs_p90": float(np.percentile(np.abs(e), 90))}


def robustness_summary(samples, records, repeat_dirs) -> dict:
    """同一子集多次运行：系统原始估计（百分制）的跨运行波动。"""
    by_id = {r.sample_id: r for r in records}
    runs = [{s.sample_id: s.raw_pct for s in samples}]
    for directory in repeat_dirs:
        found, _ = load_predictions(directory)
        run = {}
        for sid, (parsed, _) in found.items():
            p = parsed.prediction
            if sid in by_id and prediction_admission(by_id[sid], parsed)[0] and parsed.status == "succeeded" \
                    and p and p.status == "candidate":
                run[sid] = 100.0 * p.estimate / p.maximum
        runs.append(run)
    spans, sds, complete_spans = [], [], []
    for s in samples:
        values = [r[s.sample_id] for r in runs if r.get(s.sample_id) is not None]
        if len(values) == len(runs):
            complete_spans.append(max(values) - min(values))
        if len(values) >= 2:
            spans.append(max(values) - min(values)); sds.append(float(np.std(values, ddof=1)))
    if not spans:
        return {"status": "not_evaluable", "runs": len(runs), "reason": "没有在两次以上运行中都给出估计的样本"}
    return {"status": "evaluated", "runs": len(runs), "n": len(spans), "mean_sd": float(np.mean(sds)),
            "range_p50": float(np.percentile(spans, 50)), "range_p90": float(np.percentile(spans, 90)),
            "complete_n": len(complete_spans), "complete_range_p90": float(np.percentile(complete_spans, 90)) if complete_spans else None}


# ---------- 主流程 ----------

def text_similarity(samples, protocol, cache: Path) -> tuple[np.ndarray, dict]:
    docs, status = [], defaultdict(int)
    for s in samples:
        text, state = body_text(s.body_path or Path("/nonexistent"), s.body_sha256, cache)
        status[state] += 1
        docs.append(ngram_counts(text, tuple(protocol.knn.ngrams), protocol.knn.hash_bits) if text and text.strip() else None)
    return similarity_matrix(docs, protocol.knn.hash_bits), dict(status)


def scope(samples, idx) -> dict:
    out = {}
    for name in ("region", "engineering_type", "scoring_scheme"):
        counts = defaultdict(int)
        for i in idx:
            counts[getattr(samples[i], name)] += 1
        out[name] = dict(sorted(counts.items()))
    return out


def check_registry(results: Path, protocol_sha: str, force: bool) -> dict:
    path = results / REGISTRY
    registry = json.loads(path.read_text()) if path.exists() else {}
    if registry.get(protocol_sha) and not force:
        raise RuntimeError(f"该协议（SHA {protocol_sha[:12]}）的时间留出评测已运行：{registry[protocol_sha][-1]['name']}；"
                           "最终测试只运行一次，确需重跑须加 --force 并在报告中说明")
    return registry


def run(pairs_paths: list[Path], prediction_dirs: list[Path], protocol_path: Path, name: str, mode: str = "cv",
        results: Path = paths.RESULTS, datasets_root: Path | None = None, force: bool = False,
        repeat_dirs: tuple[Path, ...] = ()) -> dict:
    """pairs_paths／prediction_dirs 可为多个来源（批量回放、产品导出），按正文 SHA 去重后合并评测。"""
    pairs_paths = [pairs_paths] if isinstance(pairs_paths, Path) else list(pairs_paths)
    prediction_dirs = [prediction_dirs] if isinstance(prediction_dirs, Path) else list(prediction_dirs)
    started = time.perf_counter()
    protocol = EvalProtocol.model_validate_json(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(protocol_path)
    if repeat_dirs and not protocol.allow_repeat_runs:
        raise ValueError("协议未允许读取重复运行目录（allow_repeat_runs）")
    output = results / f"score-eval-{name}"
    if output.exists() and not force:
        raise FileExistsError(f"{output} 已存在；不覆盖既有结果，请换实验名或加 --force")
    registry = check_registry(results, protocol_sha, force) if mode == "holdout" else None
    prior_runs = len(registry.get(protocol_sha, [])) if registry is not None else 0

    label_sources = tuple(protocol.label_sources)
    sources = load_sources(pairs_paths, prediction_dirs, datasets_root, label_sources)
    records, invalid_pairs = sources["records"], sources["invalid"]
    samples, excluded = build_samples(records, sources["predictions"], sources["body_paths"], label_sources)
    excluded = invalid_pairs + excluded
    methods = list(protocol.methods)
    notes = []
    sim, text_status = np.full((len(samples), len(samples)), np.nan), {}
    if any(m in TEXT_METHODS for m in methods) and samples:
        sim, text_status = text_similarity(samples, protocol, results / "score-eval-text-cache")
        if np.isnan(np.diag(sim)).all():
            methods = [m for m in methods if m not in TEXT_METHODS]
            notes.append("全部正文均无法取得文字，正文近邻相关方法未运行")
    ctx = Context(samples, np.array([s.y for s in samples]), protocol, sim)
    cutoffs, fields = field_arrays(samples, sources["projects"])
    extra_keys = sorted({k for s in samples for k in s.features} - set(PF1))
    if extra_keys:
        notes.append("格式 B 中非 pf-1 的特征键不进入模型：" + "、".join(extra_keys))
    every = np.arange(len(samples))

    report = {"experiment_id": f"score-eval-{name}", "mode": mode, "generated_at": datetime.now(UTC).isoformat(),
              "protocol": {"path": protocol_path.name, "sha256": protocol_sha, "content": protocol.model_dump(mode="json", by_alias=True)},
              "data": {"pairs_sha256": [sha256_file(p) for p in pairs_paths],
                       "pairs_code_sha256": [p.code_sha256 for p in sources["pair_sets"]],
                       "predictions_digest": [directory_digest(d) for d in prediction_dirs],
                       "predictions_files": len(sources["predictions"])},
              "code_sha256": code_digest(), "methods_run": methods, "notes": notes,
              "holdout_prior_runs_same_protocol": prior_runs,
              "admission": {"records": len(records) + len(invalid_pairs), "admitted": len(samples),
                            "independent_projects": len({s.project_key for s in samples}),
                            "connected_groups": len({s.group for s in samples}),
                            "excluded_by_reason": dict(sorted(_count(e["reason"] for e in excluded).items())),
                            "excluded": excluded, "invalid_prediction_files": sources["invalid_predictions"],
                            "system_status": dict(sorted(_count(s.system_status for s in samples).items())),
                            "body_text": text_status, "label_sources": list(label_sources),
                            "label_mix": dict(sorted(_count(f"{s.label_source}/{'verified' if s.label_verified else 'pending'}"
                                                            for s in samples).items()))},
              "scope": scope(samples, every)}

    reason = evaluable(samples, every, protocol.minimums.labeled_samples, protocol.minimums.independent_projects)
    split = None
    if not reason and mode == "holdout":
        split = time_holdout(samples, protocol.holdout.test_fraction)
        report["holdout"] = {"cutoff": split["cutoff"], "train": len(split["train"]), "test": len(split["test"]),
                             "excluded": split["excluded"]}
        if len(split["test"]) < protocol.holdout.min_test_samples:
            reason = f"留出测试样本 {len(split['test'])} 份（下限 {protocol.holdout.min_test_samples}）"
        elif len({samples[i].group for i in split["train"]}) < protocol.cv.folds:
            reason = f"留出训练连通组少于 {protocol.cv.folds} 个"
    if reason:
        report["verdict"] = {"status": "not_evaluable", "reasons": [reason], "scope": report["scope"]}
    else:
        report.update(evaluate(ctx, mode, split, methods, cutoffs, fields, records, repeat_dirs, every))
        report["verified_official_subset"] = verified_subset(ctx, mode, split, methods, report["verdict"]["selected_method"])
    bundle = scoring_model(ctx, report)
    report["calibration_bundle"] = {"model_id": bundle["model_id"], "method": bundle["method"],
                                    "verdict": bundle["evaluation"]["verdict"]}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)

    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "calibration_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    if registry is not None:
        registry.setdefault(protocol_sha, []).append({"name": name, "run_at": report["generated_at"], "forced": force,
                                                      "pairs_sha256": report["data"]["pairs_sha256"]})
        (results / REGISTRY).write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def verified_subset(ctx: Context, mode: str, split, methods, selected: str) -> dict:
    """只用已核验官方原件标签的子集复算：基线、系统原始估计与选定方法（同一划分规则），样本不足时不可评价。"""
    p = ctx.protocol
    keep = [i for i, s in enumerate(ctx.samples) if s.label_source == "official_document" and s.label_verified]
    position = {old: new for new, old in enumerate(keep)}
    samples = [ctx.samples[i] for i in keep]
    out = {"n": len(samples), "projects": len({s.project_key for s in samples})}
    reason = evaluable(samples, range(len(samples)), p.minimums.labeled_samples, p.minimums.independent_projects)
    sub = [m for m in methods if m in BASELINES or m in ("raw", selected)]
    if not reason and mode == "holdout":
        train = np.array([position[i] for i in split["train"] if i in position], dtype=int)
        test = np.array([position[i] for i in split["test"] if i in position], dtype=int)
        if len(test) < p.holdout.min_test_samples or len(np.unique([samples[i].group for i in train])) < p.cv.folds:
            reason = f"留出测试中已核验官方标签 {len(test)} 份，或训练连通组不足"
    if reason:
        return {**out, "status": "not_evaluable", "reason": reason}
    sim = ctx.sim[np.ix_(keep, keep)]
    sub_ctx = Context(samples, ctx.y[keep], p, sim)
    no_cut = np.full(len(samples), np.nan)
    store = (run_cv(sub_ctx, np.arange(len(samples)), sub, no_cut) if mode == "cv"
             else run_holdout(sub_ctx, train, test, sub, no_cut)[0])
    metrics = summarize_methods(sub_ctx, store, sub)
    baseline = min((m for m in sub if m in BASELINES), key=lambda m: metrics[m]["mae"])
    gain = gain_against_baseline(sub_ctx, store, selected, baseline)
    return {**out, "status": "evaluated", "selected_method": selected, "metrics": metrics, "best_baseline": baseline,
            "gain": gain, "mae": metrics[selected]["mae"], "meets_target": metrics[selected]["mae"] <= p.target_mae,
            "gain_excludes_zero": gain["low"] > 0}


def _count(values) -> dict:
    out = defaultdict(int)
    for v in values:
        out[str(v)] += 1
    return out


def acceptance_summary(protocol, errors, gain, robustness, sample_count) -> dict:
    limits = protocol.acceptance
    if limits is None:
        return {"status": "not_evaluable", "reasons": ["协议未冻结融合验收门槛"], "checks": {}}
    checks = {}
    def check(name, actual, threshold, direction="max"):
        ok = actual is not None and (actual <= threshold if direction == "max" else actual >= threshold)
        checks[name] = {"actual": actual, "threshold": threshold, "direction": direction,
                        "status": "not_evaluable" if actual is None else "passed" if ok else "not_passed"}
    check("mae", float(np.mean(np.abs(errors))), limits.max_mae)
    check("abs_p90", float(np.percentile(np.abs(errors), 90)), limits.max_abs_p90)
    check("within_ratio", float(np.mean(np.abs(errors) <= limits.within_points)), limits.min_within_ratio, "min")
    check("abs_bias", abs(float(np.mean(errors))), limits.max_abs_bias)
    checks["gain_ci95_low"] = {"actual": gain["low"], "threshold": 0, "direction": "strict_min",
                              "status": "passed" if gain["low"] > 0 else "not_passed"}
    complete = (robustness.get("runs", 0) >= limits.minimum_repeat_runs and robustness.get("complete_n", 0) == sample_count)
    check("repeat_range_p90", robustness.get("complete_range_p90") if complete else None, limits.max_repeat_range_p90)
    failed = [name for name, row in checks.items() if row["status"] == "not_passed"]
    missing = [name for name, row in checks.items() if row["status"] == "not_evaluable"]
    return {"status": "not_passed" if failed else "not_evaluable" if missing else "passed",
            "reasons": [*(f"{name} 未达到冻结门槛" for name in failed), *(f"{name} 缺少完整复跑数据" for name in missing)],
            "checks": checks, "within_points": limits.within_points}


def evaluate(ctx, mode, split, methods, cutoffs, fields, records, repeat_dirs, every) -> dict:
    p = ctx.protocol
    if mode == "cv":
        store = run_cv(ctx, every, methods, cutoffs)
        selection_metrics = summarize_methods(ctx, store, methods)
        population = every
    else:
        train, test = np.array(split["train"]), np.array(split["test"])
        selection_metrics = summarize_methods(ctx, run_cv(ctx, train, methods, cutoffs), methods)
        store = run_holdout(ctx, train, test, methods, cutoffs)[0]
        population = train
    metrics = summarize_methods(ctx, store, methods)
    selected = select_method(p, selection_metrics, methods)
    baseline = min((m for m in methods if m in BASELINES), key=lambda m: metrics[m]["mae"])
    curve = learning_curve(ctx, population, [selected, baseline], p.learning_curve_fractions)
    gain = gain_against_baseline(ctx, store, selected, baseline)
    idx = store["idx"]
    reasons, mae = [], metrics[selected]["mae"]
    robustness = robustness_summary(ctx.samples, records, repeat_dirs) if repeat_dirs else {"status": "not_evaluable", "runs": 1}
    errors = store["pred"][selected].mean(axis=0) - ctx.y[idx]
    acceptance = acceptance_summary(p, errors, gain, robustness, len(ctx.samples))
    if mae > p.target_mae:
        reasons.append(f"MAE {mae:.2f} 高于目标 {p.target_mae:.2f}")
    if gain["low"] <= 0:
        reasons.append(f"相对最佳基线（{METHOD_NAMES[baseline]}）的 MAE 增益 {gain['gain']:.2f}，"
                       f"{p.confidence:.0%} 区间 [{gain['low']:.2f}, {gain['high']:.2f}] 未全部大于 0")
    verdict = {"status": "not_passed" if reasons else "passed", "reasons": reasons, "selected_method": selected,
               "selection_basis": "重复分组交叉验证 MAE 最低的可导出方法" if p.selection == "cv_mae" else "协议指定",
               "evaluation_set": "重复分组交叉验证折外预测" if mode == "cv" else "时间留出最终测试集",
               "mae": mae, "target_mae": p.target_mae, "best_baseline": baseline, "gain": gain,
               "scope": scope(ctx.samples, every)}
    if len({sample.input_method for sample in ctx.samples}) > 1:
        verdict["reasons"].append("样本混合了不同章节读取方法，应分别评测")
        verdict["status"] = "not_passed"
    verdict["acceptance"] = acceptance
    if p.acceptance is not None and acceptance["status"] != "passed":
        known_failure = bool(reasons) or acceptance["status"] == "not_passed"
        verdict["reasons"].extend(acceptance["reasons"])
        verdict["status"] = "not_passed" if known_failure else "not_evaluable"
    if mode == "cv" and p.selection == "cv_mae":
        verdict["caveat"] = "方法选择与误差估计使用同一组交叉验证折，误差估计略偏乐观；最终结论以时间留出为准"
    per_sample = []
    levels = p.interval_levels
    for pos, i in enumerate(idx):
        s = ctx.samples[i]
        sel = store["pred"][selected][:, pos]
        per_sample.append({
            "sample_id": s.sample_id, "project_key": s.project_key, "group": s.group, "system_status": s.system_status,
            "folds": store["fold"][:, pos].tolist(), "official_percent": s.y, "maximum": s.maximum,
            "raw_percent": s.raw_pct, "predictions_percent": {m: float(store["pred"][m][:, pos].mean()) for m in methods},
            "selected_by_repeat": sel.tolist(), "selected_points": float(sel.mean() * s.maximum / 100),
            "official_points": s.y * s.maximum / 100,
            "intervals_percent": {str(q): [float(store["lo"][selected][q][:, pos].mean()),
                                           float(store["hi"][selected][q][:, pos].mean())] for q in levels}})
    ranking = {m: ranking_metrics(ctx.samples, idx, ctx.y[idx], store["pred"][m].mean(axis=0), p.minimums.ranking_projects)
               for m in methods if m not in ("mean", "median")}
    out = {"verdict": verdict, "selection_metrics": selection_metrics, "metrics": metrics,
           "exportable_methods": [m for m in methods if m in EXPORTABLE], "learning_curve": curve,
           "intervals": interval_summary(ctx, store, methods, levels), "native_interval": native_interval(ctx.samples, idx),
           "slices": slice_summary(ctx, store, methods, p.slices, p.minimums.slice_samples),
           "ranking": ranking, "shortlist": shortlist_summary(ctx, store, methods, cutoffs),
           "field_percentile": percentile_summary(ctx, store, methods, fields),
           "relative_target": relative_target(ctx, fields, methods, mode, split),
           "items": item_summary(ctx.samples, p.item_match_threshold, p.minimums.item_matches),
           "per_sample": per_sample}
    if repeat_dirs:
        out["robustness"] = robustness
    return out


def scoring_model(ctx: Context, report: dict) -> dict:
    """导出 scoring-model/1：评测结束后用全部准入样本重拟合选定方法；区间半宽取该方法在全部样本上分组折外残差的保形分位数。"""
    verdict = report["verdict"]
    data_sha = hashlib.sha256(canonical(report["data"]).encode()).hexdigest()
    code_sha = hashlib.sha256(canonical(report["code_sha256"]).encode()).hexdigest()
    every = np.arange(len(ctx.samples))
    scope_ = {"engineering_types": sorted({s.engineering_type for s in ctx.samples}),
              "regions": sorted({s.region for s in ctx.samples})}  # 训练数据里出现过的完整地区，产品端按地名包含关系判断适用
    evaluation = {"n": len(ctx.samples), "projects": len({s.project_key for s in ctx.samples}),
                  "verdict": verdict["status"], "protocol_sha256": report["protocol"]["sha256"], "mode": report["mode"],
                  "label_sources": report["admission"]["label_sources"], "label_mix": report["admission"]["label_mix"],
                  "acceptance": verdict.get("acceptance", {"status": "not_evaluable"})}
    methods = {s.input_method for s in ctx.samples}
    bundle = {"input_method": next(iter(methods)) if len(methods) == 1 else "mixed", "schema": "scoring-model/1", "target": "technical_total", "feature_spec": "pf-1",
              "output": "technical_percent_of_max", "clip": [0.0, 100.0], "scope": scope_,
              "data_sha256": data_sha, "code_sha256": code_sha, "created_at": report["generated_at"]}
    if verdict["status"] == "not_evaluable":
        bundle.update({"method": None, "features": [], "standardize": {"mean": {}, "scale": {}}, "impute": {},
                       "coefficients": {}, "intercept": None, "isotonic": None, "interval": {"0.8": None, "0.9": None},
                       "evaluation": {**evaluation, "reason": "；".join(verdict["reasons"])}})
    else:
        method = verdict["selected_method"]
        model, _ = fit_model(method, ctx, every, ctx.protocol.seed)
        residuals = ctx.y - inner_oof(method, ctx, every, ctx.protocol.seed + 1)
        half = {q: conformal_quantile(np.abs(residuals), float(q)) for q in ("0.8", "0.9")}
        metrics, interval = report["metrics"], report["intervals"][method]
        bundle.update({"method": METHOD_LABELS[method], "features": model["features"], "standardize": model["standardize"],
                       "impute": model["impute"], "coefficients": model["coefficients"], "intercept": model["intercept"],
                       "isotonic": model["isotonic"],
                       "interval": {q: (h if np.isfinite(h) else None) for q, h in half.items()},
                       "evaluation": {**evaluation, "mae_pct": verdict["mae"], "best_baseline": verdict["best_baseline"],
                                      "baseline_mae_pct": metrics[verdict["best_baseline"]]["mae"],
                                      "gain_ci95": [verdict["gain"]["low"], verdict["gain"]["high"]],
                                      "coverage_0.8": interval.get("0.8", {}).get("coverage"),
                                      "ridge_alpha": model.get("alpha")}})
    bundle["model_id"] = model_id(bundle)
    return bundle


# ---------- 报告 ----------

def f2(value, digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def render_markdown(r: dict) -> str:
    v = r["verdict"]
    lines = [f"# 技术分评测：{r['experiment_id']}", "",
             f"- 结论：**{STATUS_NAMES[v['status']]}**", f"- 模式：{'分组交叉验证' if r['mode'] == 'cv' else '时间留出（最终测试，只运行一次）'}",
             f"- 协议：`{r['protocol']['path']}`，SHA-256 `{r['protocol']['sha256']}`，冻结于 {r['protocol']['content']['frozen_at']}",
             "- 配对集 SHA-256 " + "、".join(f"`{h}`" for h in r["data"]["pairs_sha256"]) + "；系统预测目录摘要 "
             + "、".join(f"`{h}`" for h in r["data"]["predictions_digest"]),
             "- 代码 SHA-256：" + "，".join(f"{k} `{h[:12]}`" for k, h in r["code_sha256"].items())]
    if r["holdout_prior_runs_same_protocol"]:
        lines.append(f"- **同一协议的时间留出已运行 {r['holdout_prior_runs_same_protocol']} 次，本次为强制重跑（登记在案）**")
    lines.append("")
    if v["status"] == "not_evaluable":
        lines += ["## 结论", "", "不可评价：" + "；".join(v["reasons"]) + "。本次不报告精度。", ""]
    else:
        g = v["gain"]
        lines += ["## 结论", "",
                  f"选定方法：{METHOD_NAMES[v['selected_method']]}（{v['selection_basis']}）。评测集：{v['evaluation_set']}。",
                  (f"MAE {f2(v['mae'])} 分（百分制），目标 ≤ {f2(v['target_mae'])}；最佳基线 {METHOD_NAMES[v['best_baseline']]}，"
                   f"MAE 增益 {f2(g['gain'])}，组级 bootstrap {g['confidence']:.0%} 区间 [{f2(g['low'])}, {f2(g['high'])}]"
                   f"（{g['resamples']} 次，{g['groups']} 个连通组）。"), ""]
        if v["status"] == "passed":
            lines.append("通过，适用范围限定为下表样本覆盖的地区、工程类型与评分办法。")
        else:
            lines.append("未通过：" + "；".join(v["reasons"]) + "。")
        if v.get("caveat"):
            lines.append(f"说明：{v['caveat']}。")
        lines.append("")
    acceptance = v.get("acceptance", {})
    lines += ["## 融合验收门槛", "", f"状态：{STATUS_NAMES.get(acceptance.get('status'), '不可评价')}", "",
              "| 指标 | 实测 | 门槛 | 状态 |", "| --- | --- | --- | --- |"]
    for name, row in acceptance.get("checks", {}).items():
        lines.append(f"| {name} | {f2(row['actual'])} | {row['direction']} {f2(row['threshold'])} | {STATUS_NAMES[row['status']]} |")
    lines += ["", *acceptance.get("reasons", []), ""]
    a = r["admission"]
    lines += ["## 样本与准入", "",
              (f"配对记录 {a['records']} 份，准入 {a['admitted']} 份，独立项目 {a['independent_projects']} 个，"
               f"连通组（项目／模板／相同正文）{a['connected_groups']} 个。"), "",
              "| 排除原因 | 份数 |", "| --- | --- |"]
    lines += [f"| {REASON_NAMES.get(k, k)} | {n} |" for k, n in a["excluded_by_reason"].items()] or ["| （无） | 0 |"]
    lines += ["", "标签来源（准入范围 " + "、".join(SOURCE_NAMES.get(k, k) for k in a["label_sources"]) + "）：" + "，".join(
        f"{SOURCE_NAMES.get(k.split('/')[0], k)}{'（已核验）' if k.endswith('/verified') else '（待核验）'} {n}"
        for k, n in a["label_mix"].items()) + "。"]
    lines += ["", "系统状态：" + "，".join(f"{VALUE_NAMES.get(k, k)} {n}" for k, n in a["system_status"].items()) + "。"]
    if a["body_text"]:
        lines.append("正文文字：" + "，".join(f"{TEXT_NAMES.get(k, k)} {n}" for k, n in a["body_text"].items()) + "。")
    if r.get("holdout"):
        h = r["holdout"]
        lines.append(f"时间留出：测试起点 {h['cutoff']}，训练 {h['train']} 份、测试 {h['test']} 份，训练侧隔离 {len(h['excluded'])} 份。")
    lines += [""] + [f"- {n}" for n in r["notes"]]
    lines += ["## 适用范围", ""]
    for key, label in (("region", "地区"), ("engineering_type", "工程类型"), ("scoring_scheme", "评分办法")):
        lines.append(f"- {label}：" + "，".join(f"{k}（{n}）" for k, n in r["scope"][key].items()))
    lines.append("")
    if v["status"] == "not_evaluable":
        return "\n".join(lines) + "\n"
    lines += ["## 方法对比（百分制）", "",
              ("每次重复每份样本各有一个折外预测，“预测数”= 样本数 × 重复次数。" if r["mode"] == "cv"
               else "“预测数”为留出测试样本数。"), "",
              "| 方法 | 预测数 | MAE | RMSE | 偏差 | 绝对误差 P50 | 绝对误差 P90 | 各次重复 MAE |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, x in r["metrics"].items():
        mark = "（选定）" if m == v["selected_method"] else "（最佳基线）" if m == v["best_baseline"] else ""
        mark += "（仅评测，不可导出）" if m not in r["exportable_methods"] and m not in BASELINES else ""
        lines.append(f"| {METHOD_NAMES[m]}{mark} | {x['n']} | {f2(x['mae'])} | {f2(x['rmse'])} | {f2(x['bias'])} | "
                     f"{f2(x['abs_p50'])} | {f2(x['abs_p90'])} | {' / '.join(f2(e) for e in x['mae_by_repeat'])} |")
    lines += ["", "可导出方法只以 pf-1 特征为输入（scoring-model/1）；正文近邻类方法依赖历史正文库，只作对照，不参与选定。"]
    if r["mode"] == "holdout":
        lines += ["", "方法选择依据（训练集内重复分组交叉验证 MAE）：" +
                  "，".join(f"{METHOD_NAMES[m]} {f2(x['mae'])}" for m, x in r["selection_metrics"].items()) + "。"]
    lines += ["", "## 预测区间", "", "分组 split-conformal：区间半宽取训练折内折外绝对残差的有限样本分位数。", "",
              "| 方法 | 名义 | 实际覆盖 | 平均宽度 |", "| --- | --- | --- | --- |"]
    for m, levels in r["intervals"].items():
        for x in levels.values():
            lines.append(f"| {METHOD_NAMES[m]} | {pct(x['nominal'])} | {pct(x['coverage'])} | {f2(x['mean_width'])} |")
    ni = r["native_interval"]
    lines += ["", f"系统原生 low–high 区间：{ni['n']} 份有区间，覆盖率 {pct(ni['coverage'])}，平均宽度 {f2(ni['mean_width'])}。", ""]
    lines += ["## 分层（MAE，百分制）", "", f"层内样本少于 {r['protocol']['content']['minimums']['slice_samples']} 份标“不可评价”。", ""]
    for field, values in r["slices"].items():
        lines += [f"**{SLICE_NAMES[field]}**", "", f"| 层 | n | {METHOD_NAMES[v['selected_method']]} | {METHOD_NAMES[v['best_baseline']]} |",
                  "| --- | --- | --- | --- |"]
        for value, x in values.items():
            if x["status"] == "not_evaluable":
                lines.append(f"| {VALUE_NAMES.get(value, value)} | {x['n']} | 不可评价 | 不可评价 |")
            else:
                lines.append(f"| {VALUE_NAMES.get(value, value)} | {x['n']} | {f2(x['mae'][v['selected_method']])} | {f2(x['mae'][v['best_baseline']])} |")
        lines.append("")
    vs = r["verified_official_subset"]
    lines += ["## 已核验官方标签子集", ""]
    if vs["status"] == "not_evaluable":
        lines += [f"已核验官方原件标签 {vs['n']} 份、项目 {vs['projects']} 个，不可评价：{vs['reason']}。", ""]
    else:
        g = vs["gain"]
        lines += [(f"已核验官方原件标签 {vs['n']} 份、项目 {vs['projects']} 个：{METHOD_NAMES[vs['selected_method']]} MAE {f2(vs['mae'])}"
                   f"（目标 {'达到' if vs['meets_target'] else '未达到'}），最佳基线 {METHOD_NAMES[vs['best_baseline']]} "
                   f"MAE {f2(vs['metrics'][vs['best_baseline']]['mae'])}，增益 {f2(g['gain'])} [{f2(g['low'])}, {f2(g['high'])}]"
                   f"（{'区间不含 0' if vs['gain_excludes_zero'] else '区间含 0'}）。"),
                  "系统原始估计 MAE " + f2(vs["metrics"]["raw"]["mae"]) + "。" if "raw" in vs["metrics"] else "", ""]
    curve = r["learning_curve"]
    lines += ["## 学习曲线", "", "同一组分组折，训练折只取部分连通组（嵌套子集），测试折不变；末段仍明显下降说明增加样本仍有收益。", "",
              "| 训练组比例 | 平均训练样本 | " + " | ".join(METHOD_NAMES[m] for m in curve[0]["mae"]) + " |",
              "| --- | --- | " + " | ".join("---" for _ in curve[0]["mae"]) + " |"]
    lines += [f"| {pct(c['fraction'])} | {c['train_samples_mean']:.0f} | " + " | ".join(f2(e) for e in c["mae"].values()) + " |"
              for c in curve]
    lines.append("")
    lines += ["## 同场排名", ""]
    first = next(iter(r["ranking"].values()), None)
    if not first or first["status"] == "not_evaluable":
        lines += [f"不可评价：{first['reason'] if first else '无方法'}（可比项目 {first['projects'] if first else 0} 个）。", ""]
    else:
        lines += [f"项目内 ≥2 份有标签样本的项目 {first['projects']} 个、样本 {first['samples']} 份、有效配对 {first['pairs']} 对。", "",
                  "| 方法 | Spearman 均值 | 两两排序正确率 | 最高分命中率 |", "| --- | --- | --- | --- |"]
        lines += [f"| {METHOD_NAMES[m]} | {f2(x['spearman_mean'])} | {pct(x['pairwise_accuracy'])} | {pct(x['top1_hit_rate'])} |"
                  for m, x in r["ranking"].items()]
        lines.append("")
    lines += ["## 入围概率", ""]
    s = r["shortlist"].get(v["selected_method"])
    if not s or s["status"] == "not_evaluable":
        lines += [f"不可评价：{s['reason'] if s else '无数据'}（{s['n'] if s else 0} 份）。", ""]
    else:
        lines += [f"有入围线的样本 {s['n']} 份，入围 {s['positives']} 份（基率 {pct(s['base_rate'])}）；概率取保形预测分布 P(得分 ≥ 入围线)。", "",
                  "| 方法 | Brier | 基率 Brier | 对数损失 | 基率对数损失 | AUC |", "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {METHOD_NAMES[m]} | {f2(x['brier'], 3)} | {f2(x['base_rate_brier'], 3)} | {f2(x['log_loss'], 3)} | "
                  f"{f2(x['base_rate_log_loss'], 3)} | {f2(x['auc'], 3)} |" for m, x in r["shortlist"].items() if x["status"] == "evaluated"]
        lines += ["", f"{METHOD_NAMES[v['selected_method']]}校准分箱：" + "；".join(
            f"[{b['range'][0]:.1f},{b['range'][1]:.1f}] n={b['n']} 预测 {pct(b['mean_probability'])} 实际 {pct(b['observed_rate'])}"
            for b in s["calibration_bins"] if b["n"]) + "。", ""]
    lines += ["## 项目相对表现", ""]
    rel = r["relative_target"]
    if rel["status"] == "not_evaluable":
        lines += [f"相对目标（本样本减同场中位数）不可评价：{rel['reason']}。"]
    else:
        lines += [f"相对目标（本样本百分制减同场中位数）{rel['n']} 份：" +
                  "，".join(f"{METHOD_NAMES[m]} MAE {f2(x['mae'])}" for m, x in rel["metrics"].items()) +
                  f"。最佳方法 {METHOD_NAMES[rel['best_method']]} 相对 {METHOD_NAMES[rel['best_baseline']]} 增益 "
                  f"{f2(rel['gain']['gain'])} [{f2(rel['gain']['low'])}, {f2(rel['gain']['high'])}]。"]
    fp = r["field_percentile"]
    if fp["status"] == "not_evaluable":
        lines += ["", f"同场百分位不可评价：{fp['reason']}。", ""]
    else:
        lines += ["", f"同场百分位误差（百分位点，{fp['n']} 份）：" +
                  "，".join(f"{METHOD_NAMES[m]} {f2(x, 1)}" for m, x in fp["mae_percentile_points"].items()) + "。", ""]
    it = r["items"]
    lines += ["## 分项", ""]
    head = (f"官方分项 {it['label_items']} 项，匹配 {it['matched']} 项（匹配率 {pct(it['match_rate'])}），"
            f"分项满分一致率 {pct(it['max_points_agreement'])}，评委间标准差均值 {f2(it['evaluator_spread_mean_sd'])}。")
    if it["status"] == "not_evaluable":
        lines += [head + f"分项误差不可评价：{it['reason']}。", ""]
    else:
        lines += [head + f"系统分项估计 MAE {f2(it['mae_percent_of_item_max'])}（分项满分百分制），偏差 {f2(it['bias'])}，P90 {f2(it['abs_p90'])}。", ""]
    if r.get("robustness"):
        rb = r["robustness"]
        lines += ["## 重复运行波动", ""]
        lines += [f"不可评价：{rb['reason']}。" if rb["status"] == "not_evaluable" else
                  f"{rb['runs']} 次运行、{rb['n']} 份样本：系统原始估计标准差均值 {f2(rb['mean_sd'])}，极差 P50 {f2(rb['range_p50'])}、P90 {f2(rb['range_p90'])}（百分制）。", ""]
    b = r["calibration_bundle"]
    lines += ["## 导出模型", "",
              (f"calibration_bundle.json（scoring-model/1）：方法 {b['method']}，model_id `{b['model_id'][:16]}…`，"
               f"结论 {STATUS_NAMES[b['verdict']]}。评测结束后用全部准入样本重拟合；区间半宽取该模型在全部样本上"
               "分组折外残差的保形分位数。产品只应加载结论为通过的模型。"), ""]
    lines += ["## 限制", "",
              "- 误差为满分百分制；原量尺误差见 report.json 每样本记录。",
              "- 结论只适用于上述样本覆盖范围；分层“不可评价”的层不作结论。",
              "- 入围概率与项目相对表现使用同场成绩，只作评测，不作特征。", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, nargs="+", required=True, help="交接格式 A：评分配对集.json（可多份，按正文 SHA 去重合并）")
    p.add_argument("--predictions", type=Path, nargs="+", required=True, help="交接格式 B 目录（每样本一个 <sample_id>.json；可多个）")
    p.add_argument("--protocol", type=Path, required=True, help="冻结的评测协议 JSON")
    p.add_argument("--name", required=True, help="实验名；输出 results/score-eval-<名称>/")
    p.add_argument("--mode", choices=("cv", "holdout"), default="cv")
    p.add_argument("--results", type=Path, default=paths.RESULTS)
    p.add_argument("--datasets-root", type=Path, help="覆盖配对集中的 datasets_root（用于定位正文 PDF）")
    p.add_argument("--repeat-predictions", type=Path, nargs="*", default=[], help="同一子集的其他运行目录（协议允许时）")
    p.add_argument("--force", action="store_true", help="覆盖同名输出；时间留出模式下允许同一协议重跑（登记在案）")
    a = p.parse_args(argv)
    report = run(a.pairs, a.predictions, a.protocol, a.name, a.mode, a.results, a.datasets_root, a.force,
                 tuple(a.repeat_predictions))
    print(json.dumps({"experiment_id": report["experiment_id"], "verdict": report["verdict"]["status"],
                      "admitted": report["admission"]["admitted"], "model_id": report["calibration_bundle"]["model_id"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
