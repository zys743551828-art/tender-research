"""本地 SQLite＋内容寻址产物的版本闭环；不承担生产身份认证。"""
from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class GateError(ValueError):
    pass


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, root: Path):
        self.root = root
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "records.sqlite", timeout=5, isolation_level=None)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS no_object_update BEFORE UPDATE ON objects BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS no_object_delete BEFORE DELETE ON objects BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TABLE IF NOT EXISTS routes (environment TEXT PRIMARY KEY, bundle_id TEXT NOT NULL REFERENCES objects(id));
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, idem TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, environment TEXT NOT NULL,
          bundle_id TEXT NOT NULL REFERENCES objects(id), snapshot_id TEXT NOT NULL REFERENCES objects(id),
          state TEXT NOT NULL, token TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0, result_id TEXT REFERENCES objects(id));
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
        INSERT OR IGNORE INTO schema_version VALUES (1);
        """)

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def close(self):
        self.db.close()

    def put_artifact(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / "artifacts" / digest
        if path.exists() and path.read_bytes() != data:
            raise GateError("内容寻址产物损坏")
        if not path.exists():
            path.write_bytes(data)
        return digest

    def read_artifact(self, digest: str) -> bytes:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise GateError("无效产物校验值")
        data = (self.root / "artifacts" / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise GateError("产物校验失败")
        return data

    def freeze(self, kind: str, payload: dict) -> str:
        text = canonical(payload)
        oid = kind + ":" + hashlib.sha256(text.encode()).hexdigest()
        self.db.execute("INSERT OR IGNORE INTO objects VALUES (?,?,?)", (oid, kind, text))
        if self.db.execute("SELECT kind,payload FROM objects WHERE id=?", (oid,)).fetchone() != (kind, text):
            raise GateError("对象校验冲突")
        return oid

    def get(self, oid: str, kind: str | None = None) -> dict:
        row = self.db.execute("SELECT kind,payload FROM objects WHERE id=?", (oid,)).fetchone()
        if not row or (kind and row[0] != kind):
            raise GateError("对象不存在或类型不符")
        return json.loads(row[1])

    def file_version(self, project: str, role: str, data: bytes, obtained_at: str, supersedes: str | None = None) -> str:
        from datetime import datetime
        if datetime.fromisoformat(obtained_at).tzinfo is None:
            raise GateError("资料取得时间须含时区")
        if supersedes:
            previous = self.get(supersedes, "file")
            if previous["project"] != project or previous["role"] != role:
                raise GateError("文件更正须保持项目与资料角色")
        return self.freeze("file", {"project": project, "role": role, "sha256": self.put_artifact(data),
                                    "obtained_at": obtained_at, "supersedes": supersedes})

    def snapshot(self, project: str, files: list[str], as_of: str) -> str:
        from datetime import datetime
        cutoff = datetime.fromisoformat(as_of)
        if cutoff.tzinfo is None:
            raise GateError("截止时间须含时区")
        if not files or len(files) != len(set(files)):
            raise GateError("快照文件必须非空且不重复")
        for oid in files:
            obj = self.get(oid, "file")
            acquired = datetime.fromisoformat(obj["obtained_at"])
            if acquired.tzinfo is None or obj["project"] != project or acquired > cutoff:
                raise GateError("跨项目或事后资料不能进入事前快照")
            if obj["role"] not in ("tender", "supplement", "submitted_body", "drawing", "boq"):
                raise GateError("官方成绩及结果不得作为事前预测输入")
            self.read_artifact(obj["sha256"])
        return self.freeze("snapshot", {"project": project, "files": sorted(files), "as_of": as_of, "contract": "input-v1"})

    def bundle(self, components: dict, experiment: str, compatibility: str = "input-v1") -> str:
        if set(components) != {"parser", "rules", "features", "strategy", "code", "dependencies"}:
            raise GateError("模型组合缺少具体组件或包含未声明组件")
        if not all(isinstance(v, str) and v and v.lower() not in ("latest", "current") for v in components.values()):
            raise GateError("组件须锁定具体版本")
        self.get(experiment, "experiment")
        return self.freeze("bundle", {"components": components, "experiment": experiment, "compatibility": compatibility})

    def release(self, environment: str, bundle: str, validation: str, actor_role: str) -> str:
        if actor_role != "release_owner":
            raise GateError("发布角色无权限")
        obj, report = self.get(bundle, "bundle"), self.get(validation, "validation")
        if report.get("bundle") != bundle or report.get("status") != "passed":
            raise GateError("发布包没有对应通过的验证")
        if environment != "fixture" or report.get("scope") != "synthetic_contract_fixture":
            raise GateError("本模块只允许夹具环境发布；没有生产正式验证")
        if obj["compatibility"] != "input-v1":
            raise GateError("输入契约不兼容")
        with self.transaction():
            previous = self.db.execute("SELECT bundle_id FROM routes WHERE environment=?", (environment,)).fetchone()
            rid = self.freeze("release", {"environment": environment, "bundle": bundle,
                                          "validation": validation, "actor_role": actor_role,
                                          "previous_bundle": previous[0] if previous else None,
                                          "nonce": secrets.token_hex(8)})
            self.db.execute("INSERT INTO routes VALUES (?,?) ON CONFLICT(environment) DO UPDATE SET bundle_id=excluded.bundle_id", (environment, bundle))
        return rid

    def enqueue(self, environment: str, snapshot: str, idem: str, payload: dict) -> str:
        input_obj = self.get(snapshot, "snapshot")
        text = canonical(payload)
        with self.transaction():
            existing = self.db.execute("SELECT id,snapshot_id,payload,environment FROM jobs WHERE idem=?", (idem,)).fetchone()
            if existing:
                if existing[1:] != (snapshot, text, environment):
                    raise GateError("幂等键对应不同输入")
                return existing[0]
            route = self.db.execute("SELECT bundle_id FROM routes WHERE environment=?", (environment,)).fetchone()
            if not route:
                raise GateError("没有可用发布版本")
            if self.get(route[0], "bundle")["compatibility"] != input_obj["contract"]:
                raise GateError("任务输入与发布包不兼容")
            job = "job:" + secrets.token_hex(16)
            self.db.execute("INSERT INTO jobs(id,idem,payload,environment,bundle_id,snapshot_id,state) VALUES (?,?,?,?,?,?,'pending')",
                            (job, idem, text, environment, route[0], snapshot))
        return job

    def claim(self, now: float, lease_seconds: float = 30) -> dict | None:
        if lease_seconds <= 0:
            raise GateError("租约时长须为正")
        with self.transaction():
            row = self.db.execute("SELECT id,bundle_id,snapshot_id,attempts FROM jobs WHERE state='pending' OR (state='running' AND lease_until<=?) ORDER BY rowid LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            token = secrets.token_hex(16)
            self.db.execute("UPDATE jobs SET state='running',token=?,lease_until=?,attempts=attempts+1 WHERE id=?", (token, now + lease_seconds, row[0]))
            return {"job_id": row[0], "bundle_id": row[1], "snapshot_id": row[2], "token": token, "attempt": row[3] + 1}

    def finish(self, job: str, token: str, result: dict, now: float) -> str:
        with self.transaction():
            row = self.db.execute("SELECT token,state,lease_until,bundle_id,snapshot_id FROM jobs WHERE id=?", (job,)).fetchone()
            if not row or row[0] != token or row[1] != "running" or row[2] <= now:
                raise GateError("旧 worker 或过期租约不得写入结果")
            oid = self.freeze("run_result", {"job": job, "bundle": row[3], "snapshot": row[4], "result": result})
            self.db.execute("UPDATE jobs SET state='completed',result_id=?,token=NULL WHERE id=?", (oid, job))
        return oid

    def outcome(self, submission: str, label: dict, previous: str | None = None) -> str:
        from .contracts import OfficialLabel
        parsed = OfficialLabel.model_validate_json(canonical(label))
        if parsed.submission_id != submission:
            raise GateError("结果须对应实际提交")
        if previous and self.get(previous, "outcome")["submission"] != submission:
            raise GateError("标签更正关联错误提交")
        return self.freeze("outcome", {"submission": submission, "label": label, "previous": previous})

    def backup(self, destination: Path):
        if destination.exists():
            raise GateError("备份目标必须为新目录")
        destination.mkdir(parents=True)
        target = sqlite3.connect(destination / "records.sqlite")
        with target:
            self.db.backup(target)
        target.close()
        shutil.copytree(self.root / "artifacts", destination / "artifacts")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (destination / "artifacts").iterdir()}
        (destination / "backup_manifest.json").write_text(canonical({"schema_version": 1, "artifacts": hashes,
            "database_sha256": hashlib.sha256((destination / "records.sqlite").read_bytes()).hexdigest()}) + "\n")

    @classmethod
    def restore(cls, backup: Path, destination: Path):
        manifest = json.loads((backup / "backup_manifest.json").read_text())
        if hashlib.sha256((backup / "records.sqlite").read_bytes()).hexdigest() != manifest["database_sha256"]:
            raise GateError("备份数据库校验失败")
        actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (backup / "artifacts").iterdir()}
        if actual != manifest["artifacts"] or any(name != digest for name, digest in actual.items()):
            raise GateError("备份产物缺失或损坏")
        if destination.exists():
            raise GateError("恢复目标必须为新目录")
        shutil.copytree(backup, destination)
        restored = cls(destination)
        try:
            restored.verify_restored()
        except BaseException:
            restored.close()
            raise
        return restored

    def verify_restored(self):
        if self.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise GateError("数据库完整性检查失败")
        if self.db.execute("PRAGMA foreign_key_check").fetchall():
            raise GateError("外键检查失败")
        for oid, kind, text in self.db.execute("SELECT id,kind,payload FROM objects"):
            if oid != kind + ":" + hashlib.sha256(canonical(json.loads(text)).encode()).hexdigest():
                raise GateError("冻结记录内容校验失败")
        for payload, in self.db.execute("SELECT payload FROM objects WHERE kind='file'"):
            self.read_artifact(json.loads(payload)["sha256"])
