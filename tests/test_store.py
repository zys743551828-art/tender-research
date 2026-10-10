"""对象存储与缓存测试；用内存假客户端代替 S3／OSS，数据均为合成夹具。"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest

from tender_research import intake, store
from tender_research.store import ObjectStore, StoreError, storage_key
from tests.test_intake import body_pages, make_pdf, pairs, row, write_sheet

STORE_ENV = {"TENDER_STORE_ENDPOINT": "https://oss-cn-hangzhou.aliyuncs.com", "TENDER_STORE_REGION": "cn-hangzhou",
             "TENDER_STORE_BUCKET": "tender-originals", "TENDER_STORE_ACCESS_KEY_ID": "AKID",
             "TENDER_STORE_SECRET_ACCESS_KEY": "SECRET"}


class FakeError(Exception):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3:
    """实现 head_object／put_object／get_object 的内存桶；conditional=False 模拟不支持 If-None-Match 的服务。"""

    def __init__(self, conditional: bool = True):
        self.objects: dict[tuple[str, str], dict] = {}
        self.conditional = conditional
        self.puts: list[dict] = []
        self.before_put = None          # 测试钩子：模拟上传前的并发写入

    def head_object(self, Bucket, Key):
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise FakeError(404, "404")
        return {"ContentLength": len(obj["body"]), "Metadata": dict(obj["meta"]), "ETag": f'"{obj["etag"]}"'}

    def put_object(self, Bucket, Key, Body, ContentMD5, Metadata, ContentType=None, IfNoneMatch=None):
        self.puts.append({"Key": Key, "IfNoneMatch": IfNoneMatch, "ContentType": ContentType})
        if self.before_put:
            self.before_put(Bucket, Key)
        if base64.b64encode(hashlib.md5(Body).digest()).decode() != ContentMD5:
            raise FakeError(400, "BadDigest")
        if IfNoneMatch == "*" and self.conditional and (Bucket, Key) in self.objects:
            raise FakeError(412, "PreconditionFailed")
        self.store(Bucket, Key, Body, Metadata)
        return {}

    def get_object(self, Bucket, Key):
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise FakeError(404, "NoSuchKey")
        return {"Body": io.BytesIO(obj["body"])}

    def store(self, bucket, key, body, meta):
        self.objects[(bucket, key)] = {"body": bytes(body), "meta": dict(meta), "etag": hashlib.md5(body).hexdigest()}


@pytest.fixture
def fake(monkeypatch):
    client = FakeS3()
    monkeypatch.setattr(store, "client_factory", lambda config: client)
    for k, v in STORE_ENV.items():
        monkeypatch.setenv(k, v)
    return client


@pytest.fixture
def library(tmp_path, monkeypatch):
    """数据集仓库：1.2 版 CASE-001（原件在仓库工作区）、公开规则资料（relative_path）、一个 1.3 版案例（原件在缓存）。"""
    datasets, cache = tmp_path / "datasets", tmp_path / "cache"
    monkeypatch.setenv("TENDER_DATASETS_CACHE", str(cache))
    legacy = datasets / "CASE-001_广德_示例"
    make_pdf(legacy / "01-招标资料" / "CASE-001_招标文件_20260820.pdf", ["legacy tender text " * 20])
    make_pdf(legacy / "02-投标文件" / "CASE-001_技术标_示例_20260910.pdf", body_pages(1))
    records = []
    for i, rel in enumerate(("01-招标资料/CASE-001_招标文件_20260820.pdf", "02-投标文件/CASE-001_技术标_示例_20260910.pdf"), 1):
        data = (legacy / rel).read_bytes()
        records.append({"file_id": f"FILE-00{i}", "path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (legacy / "资料清单.json").write_text(json.dumps({"schema_version": "1.2", "case_id": "CASE-001",
                                                     "file_records": records}), encoding="utf-8")
    rules = datasets / "公开规则资料"
    make_pdf(rules / "政策" / "规则.pdf", ["public rule text " * 20])
    data = (rules / "政策" / "规则.pdf").read_bytes()
    (rules / "资料清单.json").write_text(json.dumps({"collection_id": "PUB", "file_records": [
        {"file_id": "PUB-001", "relative_path": "政策/规则.pdf", "bytes": len(data),
         "sha256": hashlib.sha256(data).hexdigest()}]}), encoding="utf-8")
    new = cache / "CASE-002_合肥_示例" / "02-投标文件" / "CASE-002_技术标_甲_20260420.pdf"
    make_pdf(new, body_pages(2))
    data = new.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    case2 = datasets / "CASE-002_合肥_示例"
    case2.mkdir()
    (case2 / "资料清单.json").write_text(json.dumps({"schema_version": "1.3", "case_id": "CASE-002", "file_records": [
        {"file_id": "FILE-001", "path": "02-投标文件/CASE-002_技术标_甲_20260420.pdf", "bytes": len(data),
         "sha256": sha, "storage": {"key": storage_key(sha, ".pdf")}}]}), encoding="utf-8")
    return {"datasets": datasets, "cache": cache, "tmp": tmp_path}


def intake_inbox(library, tmp_path, bodies: dict[str, list[str]]) -> Path:
    """入库测试用收件目录；去掉夹具里只为存储测试准备的最简 CASE-002 清单。"""
    shutil.rmtree(library["datasets"] / "CASE-002_合肥_示例")
    inbox = tmp_path / "收件"
    make_pdf(inbox / "A招标.pdf", ["tender " * 30])
    for name, pages in bodies.items():
        make_pdf(inbox / name, pages)
    return inbox


def test_storage_key_and_config(tmp_path):
    sha = "ab" + "0" * 62
    assert storage_key(sha, ".PDF") == f"originals/sha256/ab/{sha}.pdf"
    with pytest.raises(ValueError):
        storage_key("xyz", ".pdf")
    secret = tmp_path / "secret"
    secret.write_text("from-file\n", encoding="utf-8")
    env = dict(STORE_ENV, TENDER_STORE_SECRET_ACCESS_KEY="")
    env["TENDER_STORE_SECRET_ACCESS_KEY_FILE"] = str(secret)
    config = store.load_config(env)
    assert config.secret_access_key == "from-file" and "from-file" not in repr(config)
    assert config.addressing == "virtual" and config.guard == "oss"
    assert store.load_config(dict(STORE_ENV, TENDER_STORE_ENDPOINT="https://s3.example.com")).guard == "if-none-match"
    with pytest.raises(store.ConfigError, match="TENDER_STORE_BUCKET"):
        store.load_config({k: v for k, v in STORE_ENV.items() if k != "TENDER_STORE_BUCKET"})


def test_push_is_idempotent(library, fake):
    s = store.open_store()
    result = store.push(library["datasets"], library["cache"], s)
    assert len(result["uploaded"]) == 4 and not result["errors"] and not result["missing_local"]
    for o in store.originals(library["datasets"]):
        obj = fake.objects[("tender-originals", o.key)]
        assert hashlib.sha256(obj["body"]).hexdigest() == o.sha256 and obj["meta"] == {"sha256": o.sha256}
    assert all(p["ContentType"] == "application/pdf" for p in fake.puts)
    puts = len(fake.puts)
    again = store.push(library["datasets"], library["cache"], s)
    assert len(again["exists"]) == 4 and not again["uploaded"] and len(fake.puts) == puts
    assert store.main(["--datasets", str(library["datasets"]), "push"]) == 0


def test_push_reports_missing_local_source(library, fake):
    shutil.rmtree(library["cache"])
    result = store.push(library["datasets"], library["cache"], store.open_store())
    assert [m["owner"] for m in result["missing_local"]] == ["CASE-002"] and len(result["uploaded"]) == 3


def test_overwrite_protection(library, fake, tmp_path):
    o = next(o for o in store.originals(library["datasets"]) if o.owner == "PUB")
    fake.store("tender-originals", o.key, b"other bytes", {"sha256": "f" * 64})
    result = store.push(library["datasets"], library["cache"], store.open_store())
    assert [e["key"] for e in result["errors"]] == [o.key]
    assert fake.objects[("tender-originals", o.key)]["body"] == b"other bytes"
    assert store.main(["--datasets", str(library["datasets"]), "push"]) == 1

    # 上传前 HEAD 未见对象、写入时被并发写入：If-None-Match 拒绝，相同对象视为已存在，不同对象报错
    path = library["datasets"] / "公开规则资料" / "政策" / "规则.pdf"
    data = path.read_bytes()
    key = "originals/sha256/race/same.pdf"
    fake.before_put = lambda b, k: fake.store(b, k, data, {"sha256": o.sha256})
    s = ObjectStore(fake, "tender-originals", "if-none-match")
    assert s.put(key, path, o.sha256) == "exists"
    fake.before_put = lambda b, k: fake.store(b, k, b"x", {"sha256": "e" * 64})
    with pytest.raises(StoreError, match="拒绝覆盖"):
        s.put("originals/sha256/race/diff.pdf", path, o.sha256)
    assert fake.objects[("tender-originals", "originals/sha256/race/diff.pdf")]["body"] == b"x"
    fake.before_put = None
    head_only = ObjectStore(fake, "tender-originals", "head")
    head_only.put("originals/sha256/plain.pdf", path, o.sha256)
    assert fake.puts[-1]["IfNoneMatch"] is None
    with pytest.raises(StoreError, match="校验值不符"):
        head_only.put("originals/sha256/bad.pdf", path, "0" * 64)


def test_fetch_and_verify_failure(library, fake, tmp_path):
    s = store.open_store()
    store.push(library["datasets"], library["cache"], s)
    dest = tmp_path / "fresh-cache"
    result = store.fetch([o for o in store.originals(library["datasets"]) if o.owner == "CASE-001"], dest, s)
    assert len(result["downloaded"]) == 2 and not result["errors"]
    for o in store.originals(library["datasets"]):
        if o.owner == "CASE-001":
            assert store.sha256_file(dest / o.rel) == o.sha256
    assert len(store.fetch(store.originals(library["datasets"]), dest, s)["present"]) == 2
    # 桶中对象被篡改：下载校验失败，不留下文件
    victim = next(o for o in store.originals(library["datasets"]) if o.owner == "PUB")
    fake.objects[("tender-originals", victim.key)]["body"] = b"tampered"
    other = tmp_path / "other-cache"
    result = store.fetch([victim], other, s)
    assert result["errors"] and "校验失败" in result["errors"][0]["error"]
    assert not (other / victim.rel).exists() and not list((other / victim.rel).parent.glob(".*part"))
    del fake.objects[("tender-originals", victim.key)]
    assert store.fetch([victim], other, s)["errors"][0]["error"] == "桶中没有该对象"
    code = store.main(["--datasets", str(library["datasets"]), "fetch", "--case", "PUB", "--dest", str(other)])
    assert code == 1


def test_verify_report(library, fake, capsys):
    s = store.open_store()
    store.push(library["datasets"], library["cache"], s)
    assert store.verify(library["datasets"], s) == {"checked": 4, "ok": 4, "missing": [], "mismatch": []}
    items = store.originals(library["datasets"])
    del fake.objects[("tender-originals", items[0].key)]
    fake.objects[("tender-originals", items[1].key)]["meta"] = {"sha256": "0" * 64}
    result = store.verify(library["datasets"], s)
    assert [m["key"] for m in result["missing"]] == [items[0].key]
    assert [m["key"] for m in result["mismatch"]] == [items[1].key] and result["ok"] == 2
    capsys.readouterr()
    assert store.main(["--datasets", str(library["datasets"]), "verify"]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"] == {"checked": 4, "ok": 2, "missing": 1, "mismatch": 1}


def test_status(library, fake, monkeypatch):
    out = store.status(library["datasets"], library["cache"])
    assert out["records"] == 4 and out["in_cache"] == 1 and out["in_datasets_worktree"] == 3
    assert out["store"]["configured"] and out["store"]["in_bucket"] == 0 and "SECRET" not in json.dumps(out)
    monkeypatch.delenv("TENDER_STORE_BUCKET")
    assert store.status(library["datasets"], library["cache"])["store"]["configured"] is False
    assert store.main(["--datasets", str(library["datasets"]), "status"]) == 0


def test_missing_config_fails_fast(library, monkeypatch, capsys, tmp_path):
    for k in STORE_ENV:
        monkeypatch.delenv(k, raising=False)
    assert store.main(["--datasets", str(library["datasets"]), "push"]) == 2
    assert "对象存储未配置" in capsys.readouterr().err
    inbox = intake_inbox(library, tmp_path, {"A甲.pdf": body_pages(5)})
    sheet = tmp_path / "样本登记表.csv"
    write_sheet(sheet, [row()])
    before = sorted(p for p in tmp_path.rglob("*"))
    code = intake.main(["--sheet", str(sheet), "--inbox", str(inbox), "--datasets", str(library["datasets"]), "--upload"])
    err = capsys.readouterr().err
    assert code == 2 and "对象存储未配置" in err and "TENDER_STORE_ENDPOINT" in err
    assert sorted(p for p in tmp_path.rglob("*")) == before


def test_intake_upload_and_fetch_by_pairs(library, fake, tmp_path):
    inbox = intake_inbox(library, tmp_path, {"A甲.pdf": body_pages(5), "A乙.pdf": body_pages(6)})
    sheet = tmp_path / "样本登记表.csv"
    write_sheet(sheet, [row(), row(技术标正文="A乙.pdf", 投标企业="乙")])
    env = {"datasets": library["datasets"]}
    out = intake.run(sheet, inbox, library["datasets"], upload=True)
    assert len(out.upload["uploaded"]) == 3 and out.upload["exists"] == []   # 共用的招标文件只传一次
    report = (library["datasets"] / "入库报告.md").read_text(encoding="utf-8")
    assert "本次上传 3 个对象" in report and "待上传" not in report
    for s in pairs(env)["samples"]:
        for f in [*s["tender_files"], s["body_file"]]:
            assert ("tender-originals", f["storage_key"]) in fake.objects
    out = intake.run(sheet, inbox, library["datasets"], upload=True)
    assert out.upload == {"uploaded": [], "exists": [], "failed": []}   # 全部已入库，无新原件

    # 另一台机器：只有仓库与桶，按配对集取回原件
    fresh = tmp_path / "fresh"
    result = store.fetch(store.from_pairs(library["datasets"] / "评分配对集.json"), fresh, store.open_store())
    assert len(result["downloaded"]) == 4 and not result["errors"]
    sample = pairs(env)["samples"][0]
    assert store.sha256_file(fresh / sample["body_file"]["path"]) == sample["body_file"]["sha256"]


def test_pairs_without_cache_use_stored_signatures(library, fake, tmp_path):
    inbox = intake_inbox(library, tmp_path, {"A甲.pdf": body_pages(5), "近似.pdf": body_pages(5, mutate=0.1)})
    sheet = tmp_path / "样本登记表.csv"
    write_sheet(sheet, [row(), row(招标编号="TEST-Z", 技术标正文="近似.pdf", 投标企业="乙")])
    intake.run(sheet, inbox, library["datasets"])
    env = {"datasets": library["datasets"]}
    first = pairs(env)["samples"]
    assert first[0]["template_group"] == first[1]["template_group"]
    shutil.rmtree(library["cache"])
    out = intake.run(sheet, inbox, library["datasets"])
    assert pairs(env)["samples"] == first
    assert any(i.category == "not_cached" for i in out.issues)
    # 缓存中文件大小不符：记为 missing_files
    body = Path(first[0]["body_file"]["path"])
    damaged = library["cache"] / body
    damaged.parent.mkdir(parents=True)
    damaged.write_bytes(b"short")
    intake.run(sheet, inbox, library["datasets"])
    assert pairs(env)["samples"][0]["exclusion"] == "missing_files"


def test_intake_refuses_different_file_in_cache(library, tmp_path):
    inbox = intake_inbox(library, tmp_path, {"A甲.pdf": body_pages(5)})
    sheet = tmp_path / "样本登记表.csv"
    write_sheet(sheet, [row()])
    planned = intake.run(sheet, inbox, library["datasets"], dry_run=True).rows[0]
    target = library["cache"] / planned.dir_name / planned.manifest["file_records"][1]["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different")
    with pytest.raises(intake.InputError, match="缓存中已有不同文件"):
        intake.run(sheet, inbox, library["datasets"])
    assert target.read_bytes() == b"different" and not (library["datasets"] / planned.dir_name).exists()
