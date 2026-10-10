"""可导出评分模型契约 scoring-model/1：特征规格 pf-1、拟合与参考应用公式（产品后端按同一公式复现）。

预测 = clip(isotonic(intercept + Σ coef_f · z_f))，z_f = (x_f − mean_f) / scale_f（scale 为 0 时取 1）。
x_f 为 null 时取 impute_f；列名以 "__missing" 结尾的是缺失指示列，原特征为 null 时取 1，否则取 0。
isotonic 为分段线性插值，两端外推取端点值；区间 = 预测 ± 半宽后再截到 clip，半宽为 null 表示整个量尺。
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

PF1 = ("raw_pct", "low_pct", "high_pct", "width_pct", "grade_mean", "coverage_mean", "coverage_min",
       "specificity_mean", "repeat_spread_pct", "failed_share", "mapped_share", "items_count", "max_ratio",
       "body_pages", "body_chars_log", "qualification_violated", "qualification_unclear", "findings_consistency",
       "findings_scope", "anchor_pct", "anchor_agree", "anchor_n")
MISSING = "__missing"
METHOD_LABELS = {"raw": "raw", "linear": "linear", "isotonic": "raw+isotonic", "anchor": "anchor", "ridge": "ridge",
                 "ridge_anchor": "ridge+anchor", "ridge_isotonic": "ridge+isotonic"}


def base_name(column: str) -> str:
    return column.removesuffix(MISSING)


def design(model: dict, names: list[str], x: np.ndarray) -> np.ndarray:
    """按模型列构造标准化设计矩阵；x 为样本×names 的矩阵，缺失为 NaN。"""
    position = {n: j for j, n in enumerate(names)}
    z = np.zeros((len(x), len(model["features"])))
    for c, column in enumerate(model["features"]):
        j = position.get(base_name(column))
        values = np.full(len(x), np.nan) if j is None else x[:, j]
        missing = column.endswith(MISSING)
        raw = np.isnan(values).astype(float) if missing else np.where(np.isnan(values), model["impute"][column], values)
        scale = model["standardize"]["scale"][column] or 1.0
        z[:, c] = (raw - model["standardize"]["mean"][column]) / scale
    return z


def linear_output(model: dict, names: list[str], x: np.ndarray) -> np.ndarray:
    coef = np.array([model["coefficients"][c] for c in model["features"]])
    return model["intercept"] + design(model, names, x) @ coef if len(coef) else np.full(len(x), float(model["intercept"]))


def predict(model: dict, names: list[str], x: np.ndarray) -> np.ndarray:
    out = linear_output(model, names, x)
    if model.get("isotonic"):
        out = np.interp(out, model["isotonic"]["x"], model["isotonic"]["y"])
    return np.clip(out, *model["clip"])


def fit_linear_model(names: list[str], x: np.ndarray, y: np.ndarray, alphas: list[float], clip) -> dict[float, dict]:
    """训练折内：中位数插补；仅当训练折内缺失与否有变化时加缺失指示列；训练折全缺失或恒定的列丢弃；
    标准化后闭式 L2（alpha=0 即最小二乘）。每个 alpha 返回一份模型。"""
    columns, impute = [], {}
    for j, name in enumerate(names):
        values = x[:, j]
        present = ~np.isnan(values)
        if not present.any():
            continue
        missing = not present.all()
        if missing or np.ptp(values[present]) > 0:
            columns.append(name)
            impute[name] = float(np.median(values[present]))
        if missing:
            columns.append(name + MISSING)
    shell = {"features": columns, "impute": impute,
             "standardize": {"mean": dict.fromkeys(columns, 0.0), "scale": dict.fromkeys(columns, 1.0)}}
    raw = design(shell, names, x)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale == 0] = 1.0
    z = (raw - mean) / scale
    intercept = float(y.mean())
    gram, rhs = z.T @ z, z.T @ (y - intercept)
    models = {}
    for a in alphas:
        w = np.linalg.lstsq(gram + a * np.eye(len(gram)), rhs, rcond=None)[0] if len(gram) else np.zeros(0)
        models[a] = {"features": columns, "standardize": {"mean": dict(zip(columns, mean.tolist())),
                                                          "scale": dict(zip(columns, scale.tolist()))},
                     "impute": impute, "coefficients": dict(zip(columns, w.tolist())), "intercept": intercept,
                     "isotonic": None, "clip": list(clip)}
    return models


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def model_id(bundle: dict) -> str:
    """规范 JSON（键排序、紧凑分隔、UTF-8 原文）去掉 model_id 后的 SHA-256。"""
    return hashlib.sha256(canonical({k: v for k, v in bundle.items() if k != "model_id"}).encode()).hexdigest()


def apply_bundle(bundle: dict, features: dict) -> tuple[float, float, float, float, float]:
    """参考实现：pf-1 特征字典 → (百分制预测, 80% 下限, 80% 上限, 90% 下限, 90% 上限)。"""
    names = sorted({base_name(c) for c in bundle["features"]})
    x = np.array([[np.nan if features.get(n) is None else float(features[n]) for n in names]])
    pct = float(predict(bundle, names, x)[0])
    lo, hi = bundle["clip"]
    bounds = []
    for level in ("0.8", "0.9"):
        half = bundle["interval"].get(level)
        bounds += [lo, hi] if half is None or not math.isfinite(half) else [max(lo, pct - half), min(hi, pct + half)]
    return (pct, *bounds)
