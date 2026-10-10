"""执行真实资料上的有限句式、单案例疑点与评分就绪核验。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from . import paths
from .diagnostics import extract_limited_rules, pipe_heading_conflicts, resolve_interval, revision_gate
from .rule_evidence import check_evidence, load_documents, runtime_info
from .scoring import audit_archive


def run(datasets: Path, output: Path):
    started = time.perf_counter()
    public = load_documents(datasets / "公开规则资料/资料清单.json")
    rules = extract_limited_rules(public)
    case_docs = load_documents(datasets / "CASE-001_广德_污水管网改造及新建/资料清单.json")
    bodies = [d for d in case_docs if d.file_id == "FILE-002"]
    findings = pipe_heading_conflicts(bodies)
    expected = [
        ("road_ai_enabled", any(r["project_id"] == "2026BFFGZ51067" and r["key"] == "ai_assistance" and r["value"] is True for r in rules)),
        ("landscape_ai_disabled", any(r["project_id"] == "2026BFJGZ50917" and r["key"] == "ai_assistance" and r["value"] is False for r in rules)),
        ("road_technical_max_100", any(r["project_id"] == "2026BFFGZ51067" and r["key"] == "technical_max" and r["value"] == 100 for r in rules)),
        ("recommended_not_blocking", any(r["key"] == "recommended_page_limit" and r["category"] == "recommendation" and r["value"] == 50 for r in rules)),
        ("interval_patch_preserves_both_versions", resolve_interval(rules, "2026BFFGZ51067").get("value", {}).get("upper") == 90),
        ("year_conflict_no_auto_correction", any(r["key"] == "date_year_conflict" and r["value"]["source_year"] == 2023 for r in rules)),
        ("testing_material_exception", any(r["key"] == "testing_material_timing" and r["value"] == "bid_or_post_award" for r in rules)),
        ("all_rule_quotes_are_source_spans", all(check_evidence(e, public)[0] for r in rules for e in r["evidence"])),
        ("case_known_issue_located_page_3", any(e["pdf_page"] == 3 for f in findings for e in f["evidence"])),
        ("no_official_deduction_created", all(f["official_score_effect"] is None for f in findings)),
        ("no_unreviewed_material_replacement", bool(findings) and all(revision_gate(f, None, False)["status"] == "awaiting_engineering_basis" for f in findings)),
    ]
    result = {"experiment_id": "rule-checks", "runtime": runtime_info(),
              "evaluation_kind": "known_scope_regression_not_engineering_acceptance",
              "rules": rules, "interval_resolution": resolve_interval(rules, "2026BFFGZ51067"),
              "findings": findings, "revision_gates": [revision_gate(f, None, False) for f in findings],
              "checks": [{"check_id": k, "passed": bool(v)} for k, v in expected],
              "summary": {"checks": len(expected), "passed": sum(bool(v) for _, v in expected), "rules": len(rules), "findings": len(findings)},
              "source_files": [{"file_id": d.file_id, "sha256": d.sha256} for d in public + case_docs],
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
              "elapsed_seconds": round(time.perf_counter() - started, 3),
              "limits": ["仅少量显式句式，不是通用语义解析", "CASE-001 已知疑点回归，非盲测召回或精确率", "无图纸清单与工程复核，未生成可提交修订稿"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "rule-checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    audit = audit_archive(datasets, output); audit["experiment_id"] = "score-readiness"
    (output / "score-readiness.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=Path, default=paths.DATASETS)
    p.add_argument("--output", type=Path, default=paths.RESULTS)
    a = p.parse_args()
    run(a.datasets, a.output)
