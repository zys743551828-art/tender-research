"""合成评分配对集（格式 A）与系统预测（格式 B）生成器；只用于测试，已知真值，不是业务数据。"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

TYPES = ("市政道路", "排水管网", "园林")
SCHEMES = ("综合评估法", "技术入围后价格竞争")
REGIONS = ("安徽省合肥市", "安徽省宣城市")
ITEMS = ("施工方案与技术措施", "质量管理体系与措施", "安全文明施工措施", "进度计划与保证措施")
VOCAB = [chr(0x4E00 + k) for k in range(400)]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def dec(value: float | None) -> str | None:
    return None if value is None else f"{value:.2f}"


def generate(folder: Path, n_projects: int = 125, seed: int = 7, signal: bool = True, bias: float = -12.0,
             with_text: bool = True, cache: Path | None = None, pf1: bool = True, prefix: str = "CASE") -> dict:
    """y = 70 + 8q + 项目宽严 + 噪声；系统原始估计 raw = 70 + bias + 0.9·8q + 噪声（有信号）或与 q 无关（无信号）。

    pf1=True 时格式 B 带 artifact_extra.features（pf-1，含锚定特征，约 20% 缺锚定包），否则只有顶层字段。
    返回 {"pairs": 路径, "predictions": 目录, "truth": {sample_id: y}}。正文文字直接写入文本缓存（按正文 SHA）。
    """
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    predictions = folder / "predictions"
    predictions.mkdir(exist_ok=True)
    samples, projects, truth = [], {}, {}
    start = date(2023, 1, 1)
    counter = 0
    for p in range(n_projects):
        bidders = int(rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
        maximum = float(rng.choice([20, 30, 50, 60, 100]))
        leniency = rng.normal(0, 3)
        submitted = start + timedelta(days=int(p * 1200 / n_projects + rng.integers(0, 5)))
        etype, scheme, region = TYPES[p % 3], SCHEMES[(p // 3) % 2], REGIONS[p % 2]
        project_key = f"{prefix}-ZB{p:04d}|1"
        field = []
        for b in range(bidders):
            counter += 1
            sid = f"{prefix}-{counter:03d}"
            q = rng.normal()
            qs = q if signal else rng.normal()
            y = float(np.clip(70 + 8 * q + leniency + rng.normal(0, 2.5), 5, 100))
            value = round(y * maximum / 100, 2)
            field.append(value)
            truth[sid] = 100 * value / maximum
            raw = 70 + bias + 0.9 * 8 * qs + rng.normal(0, 2.5)
            raw = float(np.clip(raw, 1, 99))
            body = sha(sid + "body")
            tender = sha(project_key + "tender")
            template = f"TG-{p:03d}" if p % 17 else f"TG-SHARED-{p // 34}"
            label_items = []
            if b == 0 and p % 2 == 0:
                for name in ITEMS:
                    for judge in ("评委1", "评委2"):
                        label_items.append({"item": name + "（评审）", "maximum": maximum / 4, "evaluator": judge,
                                            "value": round(float(np.clip((y + rng.normal(0, 4)) / 100, 0, 1)) * maximum / 4, 2)})
            status, verified, target = "verified", "official_document", "technical_total"
            draw = rng.random()
            if draw < 0.5:
                status, verified = "pending", "company_ledger"
            elif draw < 0.53:
                status, verified = "pending", "user_statement"
            if rng.random() < 0.02:
                target = "composite_total"
            samples.append({
                "sample_id": sid, "project_key": project_key, "template_group": template, "region": region,
                "engineering_type": etype, "scoring_scheme": scheme, "bidder": f"企业{b}", "own_company": b == 0,
                "submitted_at": submitted.isoformat(),
                "tender_files": [{"file_id": "FILE-001", "path": f"{sid}/01-招标资料/招标文件.pdf", "sha256": tender,
                                  "pages": 200, "role": "tender"}],
                "body_file": {"file_id": "FILE-003", "path": f"{sid}/02-投标文件/技术标.pdf", "sha256": body,
                              "pages": 80, "text_layer": True},
                "label": {"target": target, "value": value, "maximum": maximum, "verification_status": status,
                          "source_kind": verified, "known_at": (submitted + timedelta(days=20)).isoformat(),
                          "stage_result": "unknown", "items": label_items},
                "eligible": True, "exclusion": None})
            failed = rng.random() < 0.03
            abstained = not failed and rng.random() < 0.03
            pred_max = maximum if rng.random() > 0.1 else maximum - 2
            items = []
            for name in ITEMS:
                cov = float(np.clip(0.6 + 0.1 * qs + rng.normal(0, 0.1), 0, 1))
                items.append({"statement": name, "max_points": dec(pred_max / 4),
                              "estimate": dec(np.clip(raw / 100 + rng.normal(0, 0.03), 0, 1) * pred_max / 4),
                              "low": None, "high": None, "evidence_count": int(rng.integers(0, 6)),
                              "rule_basis": "unconfirmed", "failed": False, "coverage": cov,
                              "grade_index": int(np.clip(round(2 - qs), 0, 4)), "grade_count": 5})
            record = {
                "schema": "system-prediction/1", "sample_id": sid, "status": "failed" if failed else "succeeded",
                "error": "timeout" if failed else None, "fingerprint": sha(sid), "model": "fixture-model",
                "prompt_version": "fixture", "code_sha256": "0" * 64, "started_at": "2026-10-01T00:00:00Z",
                "finished_at": "2026-10-01T00:10:00Z", "inputs": {"tender_sha256": [tender], "body_sha256": body},
                "technical_full_score": dec(maximum),
                "prediction": None if failed else (
                    {"status": "abstained", "estimate": None, "low": None, "high": None, "maximum": dec(pred_max)} if abstained else
                    {"status": "candidate", "estimate": dec(raw * pred_max / 100), "low": dec(max(raw - 8, 0) * pred_max / 100),
                     "high": dec(min(raw + 8, 100) * pred_max / 100), "maximum": dec(pred_max)}),
                "items": [] if failed else items,
                "rules": {"disqualification": 5, "mandatory": 12, "scoring": 4, "with_points": 4},
                "compliance": {"qualification_total": 17, "satisfied": int(rng.integers(10, 17)), "violated": int(rng.integers(0, 2)),
                               "unclear": int(rng.integers(0, 4)), "failed": 0, "coverage": "complete"},
                "findings": {k: int(rng.integers(0, 5)) for k in ("disqualification", "response", "consistency", "scope", "improvement")},
                "body": {"pages": 80, "chars": int(rng.integers(50000, 200000))},
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "calls": 3, "cache_hits": int(rng.integers(0, 3))}}
            if pf1 and not failed:
                anchor = None if rng.random() < 0.2 else float(np.clip(70 + 8 * qs + rng.normal(0, 4), 0, 100))
                record["artifact_extra"] = {"feature_spec": "pf-1", "features": {
                    "raw_pct": None if abstained else raw, "low_pct": None if abstained else max(raw - 8, 0),
                    "high_pct": None if abstained else min(raw + 8, 100), "width_pct": None if abstained else 16.0,
                    "grade_mean": float(np.clip(0.5 + 0.15 * qs + rng.normal(0, 0.1), 0, 1)),
                    "coverage_mean": float(np.mean([i["coverage"] for i in items])),
                    "coverage_min": float(np.min([i["coverage"] for i in items])), "specificity_mean": None,
                    "repeat_spread_pct": float(abs(rng.normal(0, 3))), "failed_share": 0.0, "mapped_share": 1.0,
                    "items_count": 4.0, "max_ratio": pred_max / maximum, "body_pages": 80.0,
                    "body_chars_log": float(np.log1p(record["body"]["chars"])), "qualification_violated": 0.0,
                    "qualification_unclear": float(rng.integers(0, 4)), "findings_consistency": float(rng.integers(0, 5)),
                    "findings_scope": 0.0, "anchor_pct": anchor, "anchor_agree": None if anchor is None else 0.8,
                    "anchor_n": None if anchor is None else 6.0}}
            (predictions / f"{sid}.json").write_text(json.dumps(record, ensure_ascii=False))
            if with_text and cache is not None:
                cache.mkdir(parents=True, exist_ok=True)
                shift = int(np.clip((qs + rng.normal(0, 1.2) + 3) * 40, 0, 300))
                tokens = rng.choice(VOCAB[shift:shift + 100], size=400)
                (cache / f"{body}.txt").write_text("".join(tokens), encoding="utf-8")
        if bidders >= 2:
            projects[project_key] = {"field_scores": field + [round(float(np.median(field)), 2)],
                                     "shortlist_cutoff": round(float(np.median(field)), 2)}
    pairs = {"schema": "score-pairs/1", "generated_at": "2026-10-10T00:00:00Z", "datasets_root": ".",
             "code_sha256": "0" * 64, "samples": samples, "projects": projects}
    path = folder / "评分配对集.json"
    path.write_text(json.dumps(pairs, ensure_ascii=False))
    return {"pairs": path, "predictions": predictions, "truth": truth}


def protocol(folder: Path, **overrides) -> Path:
    base = json.loads((Path(__file__).resolve().parents[1] / "protocols/评测协议.example.json").read_text(encoding="utf-8"))
    base["acceptance"] = None  # 既有校准机制夹具使用旧协议；融合门槛单独测试
    base.update(overrides)
    path = folder / "评测协议.json"
    path.write_text(json.dumps(base, ensure_ascii=False))
    return path
