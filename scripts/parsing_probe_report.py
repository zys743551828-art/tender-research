"""用指定参考协议复核保存的解析产物，不重新调用模型；不把锚点命中当作准确率。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from parsing_probe import normalize, without_embedded_images, write_json


def summarize(protocol_path: Path, run_path: Path) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    run_bytes = (run_path / "summary.json").read_bytes()
    run = json.loads(run_bytes)
    reference = {sample["id"]: sample for sample in protocol["samples"]}
    rows = []
    for row in run["samples"]:
        sample = reference[row["id"]]
        text_path = run_path / f"{row['id']}.md"
        checks = []
        digest = None
        if row["status"] == "parsed":
            text_bytes = text_path.read_bytes()
            text = without_embedded_images(text_bytes.decode("utf-8"))
            digest = hashlib.sha256(text_bytes).hexdigest()
            checks = [{"expected": a, "found": normalize(a) in normalize(text)} for a in sample["anchors"]]
        rows.append({"id": row["id"], "status": row["status"], "seconds": row["seconds"],
                     "anchors": checks, "artifact_sha256": digest, "error": row.get("error"),
                     "structure_review": "not_evaluated", "figure_semantics_review": sample["figure_semantics_review"]})
    return {
        "engine": run["engine"], "run_summary_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "inference_protocol_sha256": run["protocol_sha256"],
        "reference_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "reference_changed": run["protocol_sha256"] != hashlib.sha256(protocol_bytes).hexdigest(),
        "scope": protocol["scope"], "parsed_pages": sum(row["status"] == "parsed" for row in rows),
        "expected_pages": len(reference), "recorded_pages": len(rows),
        "anchor_hits": sum(a["found"] for row in rows for a in row["anchors"]),
        "evaluated_anchors": sum(len(row["anchors"]) for row in rows),
        "expected_anchors": sum(len(s["anchors"]) for s in reference.values()),
        "initialization_seconds": run["initialization_seconds"], "total_seconds": run.get("total_seconds"),
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, {"schema": "parsing-probe-comparison-1", "runs": [summarize(args.protocol, p) for p in args.run]})


if __name__ == "__main__":
    main()
