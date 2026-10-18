"""在独立 PostgreSQL 库测产品检索组件的合成规模耗时；不评价语义质量。"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tender-app/backend/src"))
from bidwb import retrieval
from bidwb.embeddings import MODEL_FINGERPRINT
from bidwb.services.vector_index import dense_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("使用新的结果路径")
    name = "bidwb_scale_" + uuid.uuid4().hex
    assert re.fullmatch(r"bidwb_scale_[0-9a-f]{32}", name)
    admin = create_engine("postgresql+psycopg://postgres@127.0.0.1:15432/postgres?connect_timeout=2", isolation_level="AUTOCOMMIT")
    engine = None
    created = False
    results = []
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        created = True
        engine = create_engine(f"postgresql+psycopg://postgres@127.0.0.1:15432/{name}?connect_timeout=2")
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION vector"))
            conn.execute(text("CREATE TABLE evidence_embeddings(evidence_id text NOT NULL,project_id text NOT NULL,model_fingerprint text NOT NULL,quote_sha256 text NOT NULL,embedding real[] NOT NULL,truncated boolean NOT NULL, PRIMARY KEY(evidence_id,model_fingerprint))"))
            conn.execute(text("CREATE INDEX idx_scope ON evidence_embeddings(project_id,model_fingerprint)"))
        queries = ["总工期180日历天", "编号JD-537", "基坑12m支护"]
        for n in (1000, 10000, 50000):
            evidence = [(f"ev_{i:08d}", f"合成施工记录编号JD-{i % 1000}，施工总工期180日历天；基坑开挖深度12m，采用支护、监测和分层开挖。安全责任明确，按本工程条件组织施工。") for i in range(n)]
            documents = [quote for _, quote in evidence]
            started = time.perf_counter()
            with engine.begin() as conn:
                conn.execute(text("TRUNCATE evidence_embeddings"))
                conn.execute(text("SET LOCAL statement_timeout='60s'"))
                conn.execute(text("SELECT setseed(0.125)"))
                for start in range(0, n, 1000):
                    rows = [{"id": identity, "i": start + offset, "h": hashlib.sha256(quote.encode()).hexdigest(), "m": MODEL_FINGERPRINT} for offset, (identity, quote) in enumerate(evidence[start:start+1000])]
                    conn.execute(text("""INSERT INTO evidence_embeddings VALUES (:id,'synthetic-scale',:m,:h,
                        ARRAY(SELECT (random()+:i*0)::real FROM generate_series(1,512)),false)"""), rows)
                conn.execute(text("ANALYZE evidence_embeddings"))
                size = conn.execute(text("SELECT pg_total_relation_size('evidence_embeddings')")).scalar_one()
            seed_seconds = time.perf_counter()-started
            cases = []
            for query in queries:
                timings = {}
                started = time.perf_counter()
                keyword = retrieval.rank(query, documents, 20)
                timings["keyword_seconds"] = time.perf_counter()-started
                started = time.perf_counter()
                exact = retrieval.rank_exact(query, documents, 20)
                timings["exact_seconds"] = time.perf_counter()-started
                started = time.perf_counter()
                with engine.connect() as conn:
                    conn.execute(text("SET statement_timeout='60s'"))
                    dense, truncated = dense_rank(conn, "synthetic-scale", evidence, [1.0]+[0.0]*511, 20)
                timings["dense_seconds"] = time.perf_counter()-started
                assert dense and not truncated
                started = time.perf_counter()
                fused = retrieval.reciprocal_rank_fusion([keyword, exact, dense], 5)
                timings["rrf_seconds"] = time.perf_counter()-started
                assert len(fused) == 5
                cases.append({"query": query, **timings, "sum_components_seconds": sum(timings.values())})
            result = {"chunks": n, "quote_characters": sum(map(len, documents)), "dense_dimensions": 512, "database_bytes": size, "seed_seconds": seed_seconds, "queries": cases, "keyword_median_seconds": statistics.median(c["keyword_seconds"] for c in cases), "dense_median_seconds": statistics.median(c["dense_seconds"] for c in cases), "component_sum_median_seconds": statistics.median(c["sum_components_seconds"] for c in cases)}
            results.append(result)
            print(json.dumps({k: result[k] for k in ("chunks", "database_bytes", "keyword_median_seconds", "dense_median_seconds", "component_sum_median_seconds")}), flush=True)
    finally:
        if engine is not None:
            engine.dispose()
        if created:
            with admin.connect() as conn:
                conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
    receipt = {"synthetic_component_benchmark": True, "semantic_recall_evaluated": False, "model_encoding_included": False, "api_authorization_and_document_loading_included": False, "production_database_modified": False, "formal_retrieval_mode_changed": False, "temporary_database_removed": True, "results": results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
