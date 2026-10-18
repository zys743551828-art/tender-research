"""用产品解析和固定 RULE-001 查询比较 BM25、精确匹配、BGE 与 RRF；参考答案只用于事后计分。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tender-app/backend/src"))

from bidwb.embeddings import MODEL_FINGERPRINT, LocalEmbedder  # noqa: E402
from bidwb.parsing import PARSER_VERSION, parse_pdf  # noqa: E402
from bidwb.retrieval import Scored, rank, rank_exact, reciprocal_rank_fusion  # noqa: E402


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = ROOT / "tender-datasets/公开规则资料/资料清单.json"
    reference_path = ROOT / "tender-datasets/评测集/RULE-001.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    chunks, documents = [], []
    started = time.perf_counter()
    for row in manifest["file_records"]:
        path = manifest_path.parent / row.get("relative_path", row.get("path"))
        if not path.is_file():
            path = Path.home() / ".cache/tender-datasets/公开规则资料" / row.get("relative_path", row.get("path"))
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"source_hash_mismatch:{row['file_id']}")
        document = parse_pdf(data)
        if document.page_count != row["pages"]:
            raise ValueError("page_count_mismatch")
        header = "\n".join(chunk.quote for chunk in document.chunks if chunk.physical_page <= 3)
        project_ids = sorted(set(re.findall(r"(?<![A-Za-z0-9])20\d{2}[A-Z]{3,10}\d{4,}(?:-\d+)?(?![A-Za-z0-9])", header)))
        documents.append({"file_id": row["file_id"], "sha256": row["sha256"], "pages": document.page_count,
                          "project_ids": project_ids, "gaps": document.gaps})
        for chunk in document.chunks:
            chunks.append({"file_id": row["file_id"], "source_sha256": row["sha256"], "pdf_page": chunk.physical_page,
                           "quote": chunk.quote, "project_ids": project_ids})
    extraction_seconds = time.perf_counter() - started
    embedder = LocalEmbedder(args.model)
    started = time.perf_counter()
    encoded = embedder.encode([chunk["quote"] for chunk in chunks])
    matrix = np.asarray(encoded.vectors, dtype=np.float32)
    embedding_seconds = time.perf_counter() - started
    print(json.dumps({"chunks": len(chunks), "embedding_seconds": embedding_seconds}), flush=True)
    results = {name: [] for name in ("keyword", "exact_keyword", "dense", "hybrid")}
    for task in reference["tasks"]:
        # 只用查询及输入过滤条件检索；不把 required_evidence 和 anchors 传入任何检索器。
        allowed = [i for i, chunk in enumerate(chunks)
                   if (not task.get("file_ids") or chunk["file_id"] in task["file_ids"])
                   and (not task.get("project_id") or chunk["project_ids"] == [task["project_id"]])]
        texts = [chunks[i]["quote"] for i in allowed]
        started = time.perf_counter()
        query_vector = embedder.encode([task["query"]], query=True)
        similarities = matrix[allowed] @ np.asarray(query_vector.vectors[0], dtype=np.float32)
        dense = [Scored(int(i), float(similarities[i])) for i in np.argsort(-similarities, kind="stable")[:50]]
        keyword, exact = rank(task["query"], texts, 50), rank_exact(task["query"], texts, 50)
        channels = {"keyword": keyword, "exact_keyword": reciprocal_rank_fusion([keyword, exact], 50), "dense": dense,
                    "hybrid": reciprocal_rank_fusion([keyword, exact, dense], 50)}
        elapsed_ms = (time.perf_counter() - started) * 1000
        for name, ranking in channels.items():
            hits = [{**{k: v for k, v in chunks[allowed[item.index]].items() if k != "project_ids"}, "score": item.score}
                    for item in ranking[:5]]
            checks = []
            for required in task["required_evidence"]:
                check = {}
                for k in (1, 3, 5):
                    correct = [hit for hit in hits[:k] if hit["file_id"] == required["file_id"] and hit["pdf_page"] in required["pdf_pages"]]
                    check[f"page_at_{k}"] = bool(correct)
                    joined = normalized("\n".join(hit["quote"] for hit in correct))
                    check[f"span_at_{k}"] = bool(correct) and all(normalized(anchor) in joined for anchor in required["anchors"])
                checks.append(check)
            results[name].append({"task_id": task["task_id"], "query": task["query"], "shared_retrieval_ms": round(elapsed_ms, 3),
                                  "query_truncated": query_vector.truncated[0], "hits": hits,
                                  **{f"all_{kind}_at_{k}": all(check[f"{kind}_at_{k}"] for check in checks)
                                     for kind in ("page", "span") for k in (1, 3, 5)}})
    summaries = {name: {"tasks": len(rows), **{f"all_{kind}_at_{k}": sum(row[f"all_{kind}_at_{k}"] for row in rows)
                                            for kind in ("page", "span") for k in (1, 3, 5)}} for name, rows in results.items()}
    output = {"evaluation_kind": "known_reference_development_regression_not_blind_test", "reference_status": reference["reference_status"],
              "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(), "parser": PARSER_VERSION,
              "model_fingerprint": MODEL_FINGERPRINT, "rrf_offset": 60, "channel_top_k": 50,
              "extraction_seconds": extraction_seconds, "embedding_seconds": embedding_seconds, "chunks": len(chunks),
              "truncated_chunks": sum(encoded.truncated), "documents": documents, "summary": summaries, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
