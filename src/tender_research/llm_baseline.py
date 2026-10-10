"""CASE-001 全文基线：同一份完整招标与技术标，Flash 重复 3 次、Pro 1 次，非思考模式、自拟引文。"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from .llm_common import CASE_DIR, PROMPT_VERSION, case_messages, finite_score, freeze_input, invoke, save, sha, validate_case
from .model_service import ProviderError, create_provider
from .paths import Workspace
from .rule_evidence import load_documents

EXPERIMENT = "llm-fulltext-baseline"
RATES = {"deepseek-flash": (0.006, 0.3, 1.2), "deepseek-v4-pro": (0.044, 1.32, 3.96)}
JOBS = [("CASE-FLASH-1", "deepseek-flash"), ("CASE-FLASH-2", "deepseek-flash"),
        ("CASE-FLASH-3", "deepseek-flash"), ("CASE-PRO-1", "deepseek-v4-pro")]


def run(ws: Workspace) -> None:
    folder = ws.results / EXPERIMENT
    folder.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    code = {name: sha(here / name) for name in ("llm_baseline.py", "llm_common.py", "deepseek_client.py", "model_service.py", "rule_evidence.py")}
    protocol = {
        "experiment_id": EXPERIMENT, "prompt_version": PROMPT_VERSION, "code_sha256": code,
        "declared_before_calls": "Flash full-case 3 identical repeats plus Pro 1 on the same full input; no calibration or tuning",
        "max_completion_calls": len(JOBS), "maximum_output_tokens_per_call": 8000,
        "known_to_analyst": True, "blind_evaluation": False,
        "case_model_input": "all 219 tender pages and all 50 body pages; archived originals only",
        "excluded_from_model_input": ["user score", "observed shortlist cutoff", "case metadata", "case analysis", "reference answers"],
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/", "pricing_verified_date": "2026-10-07",
        "estimated_cost_currency": "USD", "pricing_per_million_peak": {m: list(r) for m, r in RATES.items()},
        "cost_method": "cache hit/miss and completion tokens × official rates; off-peak is half; range is estimate, not payment receipt",
        "external_transfer_authorization": {
            "source": "direct_user_message", "received_date": "2026-10-07",
            "scope": "CASE-001 full archived tender and submitted technical body",
            "destination": "https://api.deepseek.com/chat/completions",
            "purpose": "diagnostic and exploratory technical-total scoring experiments"},
    }
    freeze_input(folder / "protocol.json", protocol)
    client = create_provider(ws.secrets, "deepseek")
    health_path = folder / "health.json"
    if health_path.exists():
        health = json.loads(health_path.read_text())
    else:
        health = client.health()
        save(health_path, health)
    if health["status"] != "available" or not {"deepseek-flash", "deepseek-v4-pro"} <= set(health["models"]):
        raise ProviderError("required_models_unavailable")

    manifest = ws.datasets / CASE_DIR / "资料清单.json"
    docs = load_documents(manifest)
    case_pages = {(d.file_id, i + 1): p for d in docs for i, p in enumerate(d.pages)}
    messages = case_messages(docs)
    calls, cases = [], []
    for call_id, model in JOBS:
        record = invoke(client, folder, call_id, messages, model)
        calls.append(record)
        row = {"call_id": call_id, "model": model, "status": record["status"]}
        if record["status"] == "received":
            output = record["response"]["output"]
            row.update(candidate_score=output.get("estimated_value"), validation=validate_case(output, case_pages))
        cases.append(row)
        if record.get("http_status") in (401, 402, 403, 429):
            break

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}
    cost, cost_complete = 0.0, True
    for c in calls:
        if c["status"] != "received":
            cost_complete = False
            continue
        u = c["response"]["usage"]
        for key in usage:
            usage[key] += u.get(key, 0)
        if not all(k in u for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens")):
            cost_complete = False
        rates = RATES[c["response"]["requested_model"]]
        cost += sum(u.get(k, 0) * r for k, r in zip(("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens"), rates)) / 1e6

    # 报送结果只在全部候选生成后读取，用于对照。
    target = json.loads(manifest.read_text())["assessment"]["reported_results"][0]["value"]
    for c in cases:
        value = c.get("candidate_score")
        if finite_score(value, 5):
            c["comparison_to_user_report"] = {"target": target, "signed_difference": round(value - target, 4),
                                              "absolute_difference": round(abs(value - target), 4),
                                              "usable_candidate": c["validation"]["automatic_gate_passed"]}
    values = [c["candidate_score"] for c in cases if c["model"] == "deepseek-flash" and finite_score(c.get("candidate_score"), 5)]
    summary = {
        "experiment_id": EXPERIMENT, "finished_at": datetime.now(UTC).isoformat(),
        "actual_completion_calls": len(calls), "successful_completions": sum(c["status"] == "received" for c in calls),
        "health": health, "originals": [d.summary() for d in docs], "case_calls": cases,
        "flash_repeat": {"n": len(values), "values": values,
                         "mean": statistics.mean(values) if values else None,
                         "range": round(max(values) - min(values), 4) if values else None,
                         "population_std": statistics.pstdev(values) if values else None},
        "reference_kind": "user_report_pending_official_source_single_known_case_not_blind",
        "official_verified_pairs": 0, "trained_on_real_cases": False,
        "usage": usage, "estimated_cost_usd_offpeak_to_peak": [round(cost / 2, 6), round(cost, 6)] if cost_complete else None,
        "payment_receipt_available": False,
        "artifact_sha256": {p.name: sha(p) for p in sorted(folder.glob("*.json"))},
        "limits": ["单例差值和重复波动不等于独立预测误差", "引用匹配不证明语义或工程判断", "没有官方分项和纯AI成绩",
                   "输入为抽取文字，图形及复杂版式未全面核验", "供应商模型别名无法固定服务权重版本"],
    }
    save(ws.results / (EXPERIMENT + ".json"), summary)
    print(json.dumps({"event": "finished", "calls": len(calls), "successful": summary["successful_completions"],
                      "flash_repeat": summary["flash_repeat"], "usage": usage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run(Workspace())
