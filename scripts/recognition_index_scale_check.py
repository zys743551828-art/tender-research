"""独立恢复库中的真实 BGE 索引回填、历史过滤和合成核验复制规模实验。"""

import argparse
import json
import re
import time
import uuid
from pathlib import Path

from bidwb import db
from bidwb.access import Actor
from bidwb.config import Settings
from bidwb.embeddings import MODEL_FINGERPRINT, get_embedder
from bidwb.errors import Conflict
from bidwb.schemas import RecognitionReviewRequest, SearchRequest
from bidwb.services import evidence, parse_identity, recognition, vector_index
from bidwb.storage import MemoryObjectStore, artifact_key
from bidwb.util import new_id, sha256_hex
from mixed_identity_restore_check import ADMIN, CONTAINER, command, fingerprint
from sqlalchemy import create_engine


def worker(url, model):
    admin = create_engine(url)
    api = create_engine(url, connect_args={"options": "-c role=app_api"})
    indexer = create_engine(url, connect_args={"options": "-c role=app_worker"})
    store = MemoryObjectStore()
    settings = Settings(environment="test", retrieval_backend="pgvector", embedding_model_dir=model)
    encoder = get_embedder(model)
    with admin.connect() as conn:
        snapshot = db.one(conn, "SELECT * FROM input_snapshots WHERE parse_inputs_json IS NOT NULL ORDER BY as_of DESC LIMIT 1")
        assert snapshot
        pid, body = snapshot["project_id"], snapshot["body_version_id"]
        actor = Actor(db.scalar(conn, "SELECT owner_id FROM projects WHERE id=:p", {"p": pid}), "合成验证")
        before = fingerprint(conn)
    query = SearchRequest(query="15.12", top_k=50, mode="hybrid", snapshot_id=None)

    def search(request):
        with api.connect() as conn:
            return evidence.search(conn, actor, pid, request, settings)

    missing = search(query)
    assert missing.mode_used == "keyword" and any("hybrid_index_incomplete" in gap for gap in missing.gaps)
    started = time.perf_counter()
    count = vector_index.rebuild(indexer, pid, encoder)
    elapsed = time.perf_counter() - started
    ready = search(query)
    assert ready.mode_used == "hybrid" and ready.index_state == "ready" and ready.hits
    with api.connect() as conn:
        inputs, _ = parse_identity.capture(conn, pid, evidence.current_version_ids(conn, pid))
        current = {identity for value in inputs.values() for identity in value["evidence_ids"]}
    assert all(hit.evidence.evidence_id in current for hit in ready.hits)
    assert any(hit.evidence.quote == "Cost 15.12" and hit.evidence.semantic_review == "supported" for hit in ready.hits)
    frozen_query = query.model_copy(update={"snapshot_id": snapshot["id"]})
    frozen_hits = search(frozen_query).model_dump(mode="json")
    assert frozen_hits["mode_used"] == "hybrid"
    assert vector_index.rebuild(indexer, pid, encoder) == count
    with admin.connect() as conn:
        after = fingerprint(conn)
    assert {key: value for key, value in before.items() if key != "evidence_embeddings"} == {
        key: value for key, value in after.items() if key != "evidence_embeddings"
    }
    target = ready.hits[0].evidence.evidence_id
    with indexer.begin() as conn:
        db.run(conn, "UPDATE evidence_embeddings SET quote_sha256=:h WHERE evidence_id=:i", {"i": target, "h": "0" * 64})
    broken = search(query)
    assert broken.mode_used == "keyword" and any("hybrid_index_incomplete" in gap for gap in broken.gaps)
    vector_index.rebuild(indexer, pid, encoder)
    assert search(query).mode_used == "hybrid"
    scales = []
    for size in (1000, 10000, 20000, 20001):
        parent, art = new_id("prs"), new_id("art")
        raw = b'{"synthetic_scale":true}'
        store.put(artifact_key(art), raw, "application/json")
        with admin.begin() as conn:
            db.run(
                conn,
                """INSERT INTO artifacts(id,sha256,object_key,byte_size,media_type,contract_version,created_at,status)
                   VALUES (:a,:h,:k,:n,'application/json','fixture',now(),'committed')""",
                {"a": art, "h": sha256_hex(raw), "k": artifact_key(art), "n": len(raw)},
            )
            db.run(
                conn,
                """INSERT INTO parses(id,project_id,file_version_id,parser_version,page_count,coverage,gaps_json,artifact_id,created_at)
                   VALUES (:r,:p,:v,'synthetic-scale',1,'partial','["recognition_candidates_pending"]',:a,now())""",
                {"r": parent, "p": pid, "v": body, "a": art},
            )
            db.run(
                conn,
                """INSERT INTO evidence(id,project_id,file_version_id,parse_id,physical_page,quote,kind,seq,semantic_review)
                   SELECT :r || '_' || n,:p,:v,:r,1,'合成施工记录编号' || n || '，施工总工期180日历天，基坑分层开挖并监测。',
                     CASE WHEN n=1 THEN 'table' ELSE 'text' END,n,'pending' FROM generate_series(1,:n) n""",
                {"r": parent, "p": pid, "v": body, "n": size},
            )
        request = RecognitionReviewRequest(
            expected_parse_id=parent,
            evidence_id=parent + "_1",
            decision="accept",
            corrected_quote=None,
            reason="合成规模实验：对照确认180日历天",
            reviewed_original=True,
            reviewed_numbers=True,
        )
        started = time.perf_counter()
        if size <= recognition.MAX_EVIDENCE:
            with api.begin() as conn:
                result = recognition.review(conn, store, actor, body, request, "synthetic-scale-review")
            seconds = time.perf_counter() - started
            with admin.connect() as conn:
                assert db.scalar(conn, "SELECT count(*) FROM evidence WHERE parse_id=:r", {"r": parent}) == size
                assert db.scalar(conn, "SELECT count(*) FROM evidence WHERE parse_id=:r", {"r": result.result_parse_id}) == size
                assert db.scalar(conn, "SELECT semantic_review FROM evidence WHERE id=:i", {"i": parent + "_1"}) == "pending"
                assert db.scalar(conn, "SELECT semantic_review FROM evidence WHERE id=:i", {"i": result.result_evidence_id}) == "supported"
                mapping = db.scalar(conn, "SELECT evidence_map_json FROM recognition_reviews WHERE id=:i", {"i": result.review_id})
                assert len(mapping) == size and len(set(mapping.values())) == size
            scales.append({"evidence": size, "review_seconds": seconds, "original_preserved": True, "new_evidence": size})
        else:
            with admin.connect() as conn:
                before_limit = fingerprint(conn)
            try:
                with api.begin() as conn:
                    recognition.review(conn, store, actor, body, request, "synthetic-scale-limit")
            except Conflict as exc:
                assert exc.code == "recognition_review_too_large"
            else:
                raise AssertionError("超限核验没有拒绝")
            with admin.connect() as conn:
                assert fingerprint(conn) == before_limit
            scales.append({"evidence": size, "limit_rejected_without_writes": True})
        assert search(frozen_query).model_dump(mode="json") == frozen_hits
    with api.connect() as conn:
        assert not parse_identity.is_current(conn, pid, snapshot["id"])
    for engine in (admin, api, indexer):
        engine.dispose()
    return {
        "real_local_bge": True,
        "model_fingerprint": MODEL_FINGERPRINT,
        "initial_indexed_evidence": count,
        "initial_rebuild_seconds": elapsed,
        "missing_and_corrupt_index_fall_back": True,
        "worker_rebuild_recovers_hybrid": True,
        "api_read_and_worker_write_roles": True,
        "current_identity_filter": True,
        "rebuild_preserves_all_business_tables": True,
        "frozen_search_unchanged_after_new_parses_and_reviews": True,
        "scale_results": scales,
        "semantic_quality_evaluated": False,
        "http_oss_latency_included": False,
        "scale_store": "MemoryObjectStore",
        "scale_runs_per_size": 1,
        "production_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url")
    parser.add_argument("--model", default="/models/embedding")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker_url:
        print(json.dumps(worker(args.worker_url, args.model), ensure_ascii=False))
        return
    assert args.backup and args.output and not args.output.exists()
    name = "bidwb_recognition_" + uuid.uuid4().hex
    assert re.fullmatch(r"bidwb_recognition_[0-9a-f]{32}", name)
    admin = create_engine(ADMIN, isolation_level="AUTOCOMMIT")
    engine = None
    created = False
    root = Path(__file__).resolve().parents[2]
    try:
        with admin.connect() as conn:
            for role in ("app_api", "app_worker"):
                if not db.scalar(conn, "SELECT 1 FROM pg_roles WHERE rolname=:r", {"r": role}):
                    db.run(conn, f"CREATE ROLE {role} NOLOGIN")
            db.run(conn, f'CREATE DATABASE "{name}"')
        created = True
        command(["docker", "exec", "-i", CONTAINER, "pg_restore", "--exit-on-error", "--no-owner", "-U", "postgres", "-d", name], args.backup.read_bytes())
        engine = create_engine(ADMIN.replace("/postgres?", f"/{name}?"))
        with engine.begin() as conn:
            db.run(conn, "CREATE EXTENSION vector")
        data = command(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,size=64m",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "--mount",
                f"type=bind,source={root / 'tender-research/scripts'},target=/checks,readonly",
                "--mount",
                f"type=bind,source={root / 'tender-research/.runtime/embeddings/bge-small-zh-v1.5'},target=/models/embedding,readonly",
                "bidwb-backend:0.1.0",
                "python",
                "/checks/recognition_index_scale_check.py",
                "--worker-url",
                f"postgresql+psycopg://postgres@host.docker.internal:15432/{name}?connect_timeout=2",
            ]
        )
        result = json.loads(data)
    finally:
        if engine is not None:
            engine.dispose()
        if created:
            with admin.connect() as conn:
                db.run(conn, f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()
    result["temporary_database_removed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
