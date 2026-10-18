"""在独立库补建合成 input-v2，再备份恢复并逐表核对；不修改业务库或 OSS。"""

import argparse
import hashlib
import json
import re
import subprocess
import uuid
from pathlib import Path

from bidwb import db
from bidwb.schemas import SnapshotRequest
from bidwb.services import parse_identity, snapshots
from bidwb.storage import MemoryObjectStore
from bidwb.util import new_id
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

CONTAINER = "bidwb-fusion-check-pg"
ADMIN = "postgresql+psycopg://postgres@127.0.0.1:15432/postgres?connect_timeout=2"


def command(arguments, data=None):
    reply = subprocess.run(arguments, input=data, capture_output=True, timeout=120, check=False)
    if reply.returncode:
        raise RuntimeError(reply.stderr.decode("utf-8", errors="replace")[-2000:])
    return reply.stdout


def fingerprint(conn):
    names = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")).scalars().all()
    result = {}
    for name in names:
        assert re.fullmatch(r"[a-z_0-9]+", name)
        digest = hashlib.sha256()
        count = 0
        for row in conn.execute(text(f'SELECT to_jsonb(t)::text FROM "{name}" t ORDER BY to_jsonb(t)::text')):
            digest.update(row[0].encode() + b"\n")
            count += 1
        result[name] = {"count": count, "sha256": digest.hexdigest()}
    return result


def freeze_fixture(conn, store, receipt):
    """只在恢复副本复制合成 PDF 为招标资料，再调用正式冻结服务。"""
    pid, body = receipt["project_id"], receipt["version_id"]
    assert receipt["synthetic_only"] and receipt["real_table_extraction"]
    assert db.scalar(conn, "SELECT count(*) FROM recognition_reviews WHERE project_id=:p", {"p": pid}) >= 2
    file_id, version, parse, package = (new_id(prefix) for prefix in ("fil", "ver", "prs", "rpk"))
    params = {"p": pid, "b": body, "f": file_id, "v": version, "ps": parse, "r": package}
    db.run(conn, "INSERT INTO files(id,project_id,role,created_at) VALUES (:f,:p,'tender',now())", params)
    db.run(
        conn,
        """INSERT INTO file_versions(id,project_id,file_id,artifact_id,original_name,obtained_at,association_status,created_by)
           SELECT :v,:p,:f,artifact_id,'synthetic-restore-tender.pdf',now(),'confirmed',created_by
           FROM file_versions WHERE id=:b AND project_id=:p""",
        params,
    )
    db.run(
        conn,
        """INSERT INTO parses(id,project_id,file_version_id,parser_version,page_count,coverage,gaps_json,artifact_id,created_at)
           SELECT :ps,:p,:v,'restore-fixture-copy',page_count,coverage,gaps_json,artifact_id,now() FROM parses
           WHERE file_version_id=:b AND project_id=:p ORDER BY created_at DESC,id DESC LIMIT 1""",
        params,
    )
    for item in db.all_(
        conn,
        """SELECT e.* FROM evidence e WHERE parse_id=(SELECT id FROM parses WHERE file_version_id=:b
           AND project_id=:p ORDER BY created_at DESC,id DESC LIMIT 1) ORDER BY seq,id""",
        params,
    ):
        db.run(
            conn,
            """INSERT INTO evidence(id,project_id,file_version_id,parse_id,physical_page,printed_page,quote,kind,seq,bbox_json,semantic_review)
               VALUES (:id,:p,:v,:ps,:page,:printed,:quote,:kind,:seq,CAST(:bbox AS jsonb),:review)""",
            {
                **params,
                "id": new_id("evd"),
                "page": item["physical_page"],
                "printed": item["printed_page"],
                "quote": item["quote"],
                "kind": item["kind"],
                "seq": item["seq"],
                "bbox": json.dumps(item["bbox_json"]),
                "review": item["semantic_review"],
            },
        )
    identity, _ = parse_identity.capture(conn, pid, [version])
    db.run(
        conn,
        """INSERT INTO rule_packages(id,project_id,revision,status,artifact_id,source_versions_json,parse_inputs_json,created_at)
           SELECT :r,:p,1,'candidate',artifact_id,CAST(:sources AS jsonb),CAST(:inputs AS jsonb),now()
           FROM file_versions WHERE id=:v AND project_id=:p""",
        {**params, "sources": json.dumps([version]), "inputs": json.dumps(identity)},
    )
    request = SnapshotRequest(body_version_id=body, rule_set_id=package, fact_set_id=None, file_version_ids=[body, version], reference_dataset_id=None)
    snapshot = snapshots.freeze_inputs(conn, store, None, pid, request, "mixed-restore-fixture")
    manifest = snapshots.manifest_json(store, conn, snapshot)
    assert manifest["contract_version"] == "input-v2"
    assert manifest["parse_inputs"][body]["parse_id"] == receipt["corrected"]["result_parse_id"]
    return {"project_id": pid, "snapshot_id": snapshot, "manifest": manifest}


def check_reviews(conn):
    maps = 0
    reviews = db.all_(conn, "SELECT * FROM recognition_reviews ORDER BY id")
    for row in reviews:
        parent = {item["id"]: item for item in db.all_(conn, "SELECT * FROM evidence WHERE parse_id=:p", {"p": row["parent_parse_id"]})}
        result = {item["id"]: item for item in db.all_(conn, "SELECT * FROM evidence WHERE parse_id=:p", {"p": row["result_parse_id"]})}
        assert set(row["evidence_map_json"]) == set(parent)
        assert set(row["evidence_map_json"].values()) == set(result)
        assert row["evidence_map_json"][row["source_evidence_id"]] == row["result_evidence_id"]
        for old, new in row["evidence_map_json"].items():
            assert parent[old]["project_id"] == result[new]["project_id"] == row["project_id"]
            assert parent[old]["file_version_id"] == result[new]["file_version_id"] == row["file_version_id"]
            maps += 1
    assert reviews
    return {"reviews": len(reviews), "mapped_evidence_pairs": maps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-backup", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--mixed-backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.mixed_backup.exists() and not args.output.exists()
    source = args.source_backup.read_bytes()
    assert hashlib.sha256(source).hexdigest() == args.source_sha256
    receipt = json.loads((Path(__file__).resolve().parents[1] / "results/recognition-1/live/receipt.json").read_text(encoding="utf-8"))
    admin = create_engine(ADMIN, isolation_level="AUTOCOMMIT")
    engines, names = [], []
    store = MemoryObjectStore()
    result = {}
    try:
        with admin.connect() as conn:
            for role in ("app_api", "app_worker"):
                if not conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname=:r"), {"r": role}).scalar():
                    conn.execute(text(f"CREATE ROLE {role} NOLOGIN"))
        for stage in ("fixture", "restored"):
            name = "bidwb_mixed_" + uuid.uuid4().hex
            assert re.fullmatch(r"bidwb_mixed_[0-9a-f]{32}", name)
            with admin.connect() as conn:
                conn.execute(text(f'CREATE DATABASE "{name}"'))
            names.append(name)
            command(["docker", "exec", "-i", CONTAINER, "pg_restore", "--exit-on-error", "--no-owner", "-U", "postgres", "-d", name], source)
            engine = create_engine(ADMIN.replace("/postgres?", f"/{name}?"))
            engines.append(engine)
            if stage == "fixture":
                with engine.begin() as conn:
                    fixture = freeze_fixture(conn, store, receipt)
                with engine.connect() as conn:
                    before = fingerprint(conn)
                    chain = check_reviews(conn)
                    assert db.scalar(conn, "SELECT count(*) FROM input_snapshots WHERE parse_inputs_json IS NULL") > 0
                source = command(["docker", "exec", CONTAINER, "pg_dump", "-U", "postgres", "-d", name, "-Fc"])
                args.mixed_backup.parent.mkdir(parents=True, exist_ok=True)
                args.mixed_backup.write_bytes(source)
            else:
                with engine.connect() as conn:
                    after = fingerprint(conn)
                    assert before == after
                    assert check_reviews(conn) == chain
                    assert snapshots.manifest_json(store, conn, fixture["snapshot_id"]) == fixture["manifest"]
                    assert parse_identity.is_current(conn, fixture["project_id"], fixture["snapshot_id"])
                    frozen = db.scalar(conn, "SELECT count(*) FROM input_snapshots WHERE parse_inputs_json IS NOT NULL")
                    legacy = db.scalar(conn, "SELECT count(*) FROM input_snapshots WHERE parse_inputs_json IS NULL")
                    assert frozen >= 1 and legacy >= 1
                    assert db.scalar(conn, "SELECT count(*) FROM pg_constraint WHERE connamespace='public'::regnamespace AND NOT convalidated") == 0
                    revision = db.scalar(conn, "SELECT version_num FROM alembic_version")
                blocked = []
                for table in ("recognition_reviews", "input_snapshots", "evidence", "parses"):
                    with engine.begin() as conn:
                        nested = conn.begin_nested()
                        try:
                            db.run(conn, f'UPDATE "{table}" SET id=id WHERE id=(SELECT id FROM "{table}" LIMIT 1)')
                        except DBAPIError as exc:
                            assert getattr(exc.orig, "sqlstate", None) == "23000" and f"append_only:{table}" in str(exc.orig)
                            nested.rollback()
                            blocked.append(table)
                        else:
                            raise AssertionError(f"只追加保护失效：{table}")
                for role in ("app_api", "app_worker"):
                    with engine.begin() as conn:
                        db.run(conn, f"SET LOCAL ROLE {role}")
                        assert db.scalar(conn, "SELECT count(*) FROM recognition_reviews") == chain["reviews"]
                result = {
                    "source_backup_sha256": args.source_sha256,
                    "mixed_backup_sha256": hashlib.sha256(source).hexdigest(),
                    "mixed_backup_bytes": len(source),
                    "revision": revision,
                    "table_fingerprints_before": before,
                    "table_fingerprints_after": after,
                    "all_rows_preserved": True,
                    "legacy_snapshots": legacy,
                    "frozen_snapshots": frozen,
                    "recognition_chain": chain,
                    "frozen_manifest_and_evidence_preserved": True,
                    "append_only_updates_blocked": blocked,
                    "roles_can_read_reviews": ["app_api", "app_worker"],
                    "fixture_scope": "isolated restored copy; synthetic tender/rule package and real freeze service; no model calls",
                    "manifest_store_scope": "in-memory fixture object retained across database restore, not OSS restoration",
                    "production_database_modified": False,
                    "oss_objects_restored": False,
                    "ownership_restored": False,
                }
    finally:
        for engine in engines:
            engine.dispose()
        with admin.connect() as conn:
            for name in reversed(names):
                assert re.fullmatch(r"bidwb_mixed_[0-9a-f]{32}", name)
                conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
    result["temporary_databases_removed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("revision", "all_rows_preserved", "legacy_snapshots", "frozen_snapshots", "recognition_chain")}))


if __name__ == "__main__":
    main()
