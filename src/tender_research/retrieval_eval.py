"""冻结规则参考集的检索回归；参考答案仅在评分阶段读取。"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import paths
from .rule_evidence import EvidenceIndex, check_evidence, load_documents, normalized, runtime_info


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(index: EvidenceIndex, reference: dict) -> dict:
    rows = []
    for task in reference["tasks"]:
        started = time.perf_counter()
        hits = index.search(task["query"], project_id=task.get("project_id"),
                            file_ids=task.get("file_ids"), top_k=5)
        checks = []
        for required in task["required_evidence"]:
            correct = [h for h in hits if h["file_id"] == required["file_id"]
                       and h["pdf_page"] in required["pdf_pages"]]
            def supported(k: int, hits=hits, correct=correct, required=required) -> bool:
                text = normalized("\n".join(h["quote"] for h in hits[:k] if h in correct))
                return bool(correct) and all(normalized(a) in text for a in required["anchors"])
            doc = next(d for d in index.documents if d.file_id == required["file_id"])
            source = normalized("\n".join(doc.pages[p - 1] for p in required["pdf_pages"]))
            checks.append({"file_id": required["file_id"], "reference_pages": required["pdf_pages"],
                           "reference_anchors_present": all(normalized(a) in source for a in required["anchors"]),
                           "page_at_1": any(h in correct for h in hits[:1]),
                           "page_at_3": any(h in correct for h in hits[:3]),
                           "page_at_5": bool(correct),
                           **{f"span_at_{k}": supported(k) for k in (1, 3, 5)}})
        valid = all(check_evidence(h, index.documents)[0] for h in hits)
        scope = all((not task.get("file_ids") or h["file_id"] in task["file_ids"])
                    and (not task.get("project_id") or next(d for d in index.documents
                         if d.file_id == h["file_id"]).project_ids == (task["project_id"],)) for h in hits)
        rows.append({"task_id": task["task_id"], "group": task["group"], "query": task["query"],
                     "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                     "evidence_valid": valid, "scope_valid": scope, "required_checks": checks,
                     **{f"all_{kind}_at_{k}": all(c[f"{kind}_at_{k}"] for c in checks)
                        for kind in ("page", "span") for k in (1, 3, 5)}, "hits": hits})
    def summarize(items: list[dict]) -> dict:
        return {"tasks": len(items), **{f"all_{kind}_at_{k}": sum(r[f"all_{kind}_at_{k}"] for r in items)
                                      for kind in ("page", "span") for k in (1, 3, 5)},
                "invalid_evidence": sum(not r["evidence_valid"] for r in items),
                "scope_errors": sum(not r["scope_valid"] for r in items),
                "missing_reference_anchors": sum(not c["reference_anchors_present"] for r in items for c in r["required_checks"]),
                "query_mean_ms": round(sum(r["elapsed_ms"] for r in items) / max(len(items), 1), 3)}
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    return {"summary": summarize(rows), "groups": {g: summarize(r) for g, r in by_group.items()}, "tasks": rows}


def run(manifest: Path, reference_path: Path, output: Path, experiment_id="retrieval-rules") -> dict:
    reference = json.loads(reference_path.read_text())
    archive = json.loads(manifest.read_text())
    actual = {r["file_id"]: (r["sha256"], r["pages"]) for r in archive["file_records"]}
    if any(actual.get(r["file_id"]) != (r["sha256"], r["pages"]) for r in reference["source_files"]):
        raise ValueError("参考集冻结原件与当前清单不符")
    result = {"experiment_id": experiment_id, "reference_id": reference["evaluation_id"],
              "evaluation_kind": "known_reference_regression_not_blind_test", "runtime": runtime_info(),
              "manifest_sha256": sha(manifest), "reference_sha256": sha(reference_path),
              "code_sha256": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("rule_evidence.py"))},
              "candidates": [], "limits": ["检索匹配不等于语义规则正确", "字符引用不等于全部页面的像素核验", "参考集由开发者建立且已知，不证明盲测泛化"]}
    for extractor in ("pypdf", "pdfplumber-crop", "pdfplumber-render-guard"):
        started = time.perf_counter()
        docs = load_documents(manifest, extractor)
        extraction_seconds = round(time.perf_counter() - started, 3)
        page_checks = []
        for c in reference["visible_page_checks"]:
            doc = next(d for d in docs if d.file_id == c["file_id"])
            page_checks.append({**c, "passed": normalized(c["must_not_contain"]) not in normalized(doc.pages[c["pdf_page"] - 1])})
        for size, overlap in ((900, 180), (1500, 300), (100000, 0)):
            started = time.perf_counter()
            index = EvidenceIndex(docs, size, overlap)
            candidate = {"candidate_id": f"{extractor}-{size}-{overlap}", "extractor": extractor,
                         "window_chars": size, "overlap_chars": overlap, "extraction_seconds": extraction_seconds,
                         "index_seconds": round(time.perf_counter() - started, 3),
                         "chunks": len(index.chunks), "documents": [d.summary() for d in docs],
                         "visible_page_checks": page_checks, **evaluate(index, reference)}
            result["candidates"].append(candidate)
            print(json.dumps({"candidate": candidate["candidate_id"], "summary": candidate["summary"],
                              "extraction_seconds": extraction_seconds, "page_checks_pass": all(p["passed"] for p in page_checks)}, ensure_ascii=False), flush=True)
    # 本值为整个顺序实验进程峰值，不能解释为单候选峰值；ru_maxrss 在 macOS 为字节，在 Linux 为 KiB。
    result["process_peak_rss"] = {"value": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "unit": "bytes" if sys.platform == "darwin" else "KiB"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=paths.RULES_MANIFEST)
    p.add_argument("--reference", type=Path, default=paths.RULES_REFERENCE)
    p.add_argument("--output", type=Path, default=paths.RESULTS / "retrieval-rules.json")
    p.add_argument("--experiment-id", default="retrieval-rules")
    a = p.parse_args()
    run(a.manifest, a.reference, a.output, a.experiment_id)
