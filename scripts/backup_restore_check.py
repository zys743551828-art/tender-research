"""把既有备份恢复到独立临时库，核对迁移、记录、只追加约束与角色权限。"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("使用新的结果路径")
    backup = args.backup.resolve(strict=True)
    digest = hashlib.sha256(backup.read_bytes()).hexdigest()
    assert digest == args.sha256 and re.fullmatch(r"[0-9a-f]{64}", digest)
    root = Path(__file__).resolve().parents[2]
    expected_head = ScriptDirectory(str(root / "tender-app/backend/migrations")).get_current_head()
    name = "bidwb_restore_" + uuid.uuid4().hex
    assert re.fullmatch(r"bidwb_restore_[0-9a-f]{32}", name)
    container = "bidwb-fusion-check-pg"
    remote = f"/tmp/{name}.dump"
    admin = create_engine("postgresql+psycopg://postgres@127.0.0.1:15432/postgres?connect_timeout=2", isolation_level="AUTOCOMMIT")
    url = f"postgresql+psycopg://postgres@127.0.0.1:15432/{name}?connect_timeout=2"
    engine = None
    created = False
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        created = True
        subprocess.run(["docker", "cp", str(backup), f"{container}:{remote}"], capture_output=True, text=True, check=True, timeout=60)
        restored = subprocess.run(
            ["docker", "exec", container, "pg_restore", "--exit-on-error", "--no-owner", "-U", "postgres", "-d", name, remote],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if restored.returncode:
            raise RuntimeError(restored.stderr[-2000:])
        engine = create_engine(url)

        def counts(conn):
            names = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")).scalars().all()
            assert all(re.fullmatch(r"[a-z_0-9]+", n) for n in names)
            return {n: conn.execute(text(f'SELECT count(*) FROM "{n}"')).scalar_one() for n in names}

        with engine.connect() as conn:
            before = counts(conn)
            snapshots_before = [list(row) for row in conn.execute(text("SELECT id,sha256,contract_version FROM input_snapshots ORDER BY id"))]
            revision_before = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            foreign_keys = conn.execute(text("SELECT count(*) FROM pg_constraint WHERE contype='f' AND connamespace='public'::regnamespace")).scalar_one()
        migration = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=root / "tender-app/backend",
            env={**os.environ, "BIDWB_MIGRATION_DATABASE_URL": url, "PYTHONUTF8": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
            check=False,
        )
        if migration.returncode:
            raise RuntimeError(migration.stderr[-2000:])
        with engine.connect() as conn:
            after = counts(conn)
            foreign_keys_after = conn.execute(text("SELECT count(*) FROM pg_constraint WHERE contype='f' AND connamespace='public'::regnamespace")).scalar_one()
            unvalidated = conn.execute(text("SELECT count(*) FROM pg_constraint WHERE connamespace='public'::regnamespace AND NOT convalidated")).scalar_one()
            assert unvalidated == 0 and foreign_keys_after >= foreign_keys
            revision_after = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            assert all(after[n] == count for n, count in before.items())
            assert revision_after == expected_head
            assert [list(row) for row in conn.execute(text("SELECT id,sha256,contract_version FROM input_snapshots ORDER BY id"))] == snapshots_before
            legacy_unknown = conn.execute(
                text("SELECT count(*) FROM input_snapshots WHERE parse_inputs_json IS NULL AND coverage_gaps_json IS NULL")
            ).scalar_one()
            assert legacy_unknown == before["input_snapshots"]
            assert {"word_conversions", "word_pdf_links", "word_pdf_adoptions"}.issubset(after)
            assert after["projects"] > 0 and after["file_versions"] > 0 and after["audit_events"] > 0
        blocked = []
        for table, column in (("file_versions", "original_name"), ("audit_events", "action")):
            with engine.begin() as conn:
                nested = conn.begin_nested()
                try:
                    conn.execute(text(f'UPDATE "{table}" SET "{column}"="{column}" WHERE id=(SELECT id FROM "{table}" LIMIT 1)'))
                except DBAPIError as exc:
                    assert getattr(exc.orig, "sqlstate", None) == "23000" and f"append_only:{table}" in str(exc.orig)
                    blocked.append(table)
                    nested.rollback()
                else:
                    raise AssertionError(f"只追加保护未生效：{table}")
        role_checks = []
        with engine.begin() as conn:
            conn.execute(text("SET LOCAL ROLE app_api"))
            assert conn.execute(text("SELECT count(*) FROM projects")).scalar_one() == after["projects"]
            assert conn.execute(text("SELECT count(*) FROM word_conversions")).scalar_one() == after["word_conversions"]
            role_checks.append("app_api_read_business_and_new_word_tables")
        with engine.begin() as conn:
            conn.execute(text("SET LOCAL ROLE app_worker"))
            assert conn.execute(text("SELECT count(*) FROM file_versions")).scalar_one() == after["file_versions"]
            role_checks.append("app_worker_read_archives")
        result = {
            "backup_sha256": digest,
            "backup_bytes": backup.stat().st_size,
            "isolated_container": container,
            "revision_before": revision_before,
            "revision_after": revision_after,
            "counts_before": before,
            "counts_after": after,
            "existing_row_counts_preserved": True,
            "foreign_keys_before": foreign_keys,
            "foreign_keys_after": foreign_keys_after,
            "unvalidated_constraints": unvalidated,
            "append_only_updates_blocked": blocked,
            "role_checks": role_checks,
            "ownership_restored": False,
            "oss_objects_restored": False,
            "production_database_modified": False,
            "temporary_database_removed": True,
        }
        result.update(
            {"expected_head": expected_head, "snapshot_id_sha_contract_preserved": True, "legacy_snapshots_without_invented_identity": legacy_unknown}
        )
    finally:
        if engine is not None:
            engine.dispose()
        if created:
            with admin.connect() as conn:
                conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
        subprocess.run(["docker", "exec", container, "rm", "-f", remote], capture_output=True, timeout=30, check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "revision_before": revision_before,
                "revision_after": revision_after,
                "existing_row_counts_preserved": True,
                "append_only_checks": len(blocked),
                "temporary_database_removed": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
