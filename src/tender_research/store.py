"""原件对象存储：内容寻址上传、取回到本地缓存、核验与状态。

原件按校验值存放在 ``originals/sha256/<前两位>/<sha256><扩展名>``；git 中的数据集只保存资料清单、校验值与存储键，
桶名属于部署配置，不写入清单。本地缓存（``TENDER_DATASETS_CACHE``，默认 ``~/.cache/tender-datasets``）与数据集同样按
案例目录排列真实文件。boto3 为可选依赖：``pip install -e ".[store]"``。

配置（环境变量；密钥类可改用 ``<名称>_FILE`` 指向只含该值的文件）：

- ``TENDER_STORE_ENDPOINT``、``TENDER_STORE_REGION``、``TENDER_STORE_BUCKET``
- ``TENDER_STORE_ACCESS_KEY_ID``、``TENDER_STORE_SECRET_ACCESS_KEY``
- ``TENDER_STORE_ADDRESSING``：virtual（默认，阿里云 OSS 要求）或 path
- ``TENDER_STORE_SIGNATURE``：auto（默认；阿里云端点用 s3，其余用 s3v4）、s3、s3v4
- ``TENDER_STORE_WRITE_GUARD``：auto（默认；OSS 端点用 x-oss-forbid-overwrite，其余用 If-None-Match）、
  if-none-match、oss、head（只做上传前 HEAD 检查）
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .paths import DATASETS

KEY_PREFIX = "originals/sha256"
MANIFEST = "资料清单.json"
REQUIRED = {"endpoint": "TENDER_STORE_ENDPOINT", "region": "TENDER_STORE_REGION", "bucket": "TENDER_STORE_BUCKET",
            "access_key_id": "TENDER_STORE_ACCESS_KEY_ID", "secret_access_key": "TENDER_STORE_SECRET_ACCESS_KEY"}
GUARDS = ("auto", "if-none-match", "oss", "head")
NOT_FOUND = {"404", "NoSuchKey", "NotFound"}
CONFLICT = {"409", "412", "PreconditionFailed", "ConditionalRequestConflict", "FileAlreadyExists"}

# 测试注入点：接收 StoreConfig，返回实现 head_object／put_object／get_object 的客户端。
client_factory: Callable[[StoreConfig], object] | None = None


class StoreError(RuntimeError):
    """对象存储操作失败。"""


class ConfigError(StoreError):
    """对象存储未配置或依赖缺失。"""


@dataclass(frozen=True)
class StoreConfig:
    endpoint: str
    region: str
    bucket: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    addressing: str = "virtual"
    write_guard: str = "auto"
    signature_version: str = "auto"

    @property
    def signature(self) -> str:
        if self.signature_version != "auto":
            return self.signature_version
        host = urlsplit(self.endpoint).hostname or ""
        return "s3" if host.endswith(".aliyuncs.com") else "s3v4"

    @property
    def guard(self) -> str:
        if self.write_guard != "auto":
            return self.write_guard
        return "oss" if "aliyuncs.com" in self.endpoint else "if-none-match"


def _env_value(env: Mapping[str, str], name: str) -> str | None:
    if env.get(name):
        return env[name].strip()
    path = env.get(name + "_FILE")
    if path:
        try:
            return Path(path).expanduser().read_text(encoding="utf-8").strip() or None
        except OSError as exc:
            raise ConfigError(f"{name}_FILE 无法读取：{exc}") from None
    return None


def load_config(env: Mapping[str, str] | None = None) -> StoreConfig:
    env = os.environ if env is None else env
    values = {k: _env_value(env, name) for k, name in REQUIRED.items()}
    missing = [REQUIRED[k] for k, v in values.items() if not v]
    if missing:
        raise ConfigError("对象存储未配置：缺少 " + "、".join(missing)
                          + "。在环境变量中设置，或用对应的 *_FILE 变量指向只含该值的文件。")
    addressing = env.get("TENDER_STORE_ADDRESSING", "virtual")
    guard = env.get("TENDER_STORE_WRITE_GUARD", "auto")
    signature = env.get("TENDER_STORE_SIGNATURE", "auto")
    if signature not in ("auto", "s3", "s3v4"):
        raise ConfigError("TENDER_STORE_SIGNATURE 须为 auto、s3 或 s3v4")
    if addressing not in ("virtual", "path"):
        raise ConfigError(f"TENDER_STORE_ADDRESSING 须为 virtual 或 path：{addressing!r}")
    if guard not in GUARDS:
        raise ConfigError(f"TENDER_STORE_WRITE_GUARD 须为 {'、'.join(GUARDS)} 之一：{guard!r}")
    return StoreConfig(**values, addressing=addressing, write_guard=guard, signature_version=signature)


def cache_root(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("TENDER_DATASETS_CACHE") or "~/.cache/tender-datasets").expanduser()


def storage_key(sha256: str, suffix: str) -> str:
    """内容寻址存储键；扩展名取原件的扩展名（小写）。"""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError(f"不是 SHA-256：{sha256!r}")
    return f"{KEY_PREFIX}/{sha256[:2]}/{sha256}{suffix.lower()}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _forbid_overwrite(request, **_):
    request.headers["x-oss-forbid-overwrite"] = "true"


def make_client(config: StoreConfig):
    if client_factory is not None:
        return client_factory(config)
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ConfigError('未安装对象存储依赖：pip install -e ".[store]"') from None
    client = boto3.client(
        "s3", endpoint_url=config.endpoint, region_name=config.region,
        aws_access_key_id=config.access_key_id, aws_secret_access_key=config.secret_access_key,
        config=Config(signature_version=config.signature, s3={"addressing_style": config.addressing},
                      request_checksum_calculation="when_required", response_checksum_validation="when_required"))
    if config.guard == "oss":
        client.meta.events.register("before-sign.s3.PutObject", _forbid_overwrite)
    return client


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return ""
    code = response.get("Error", {}).get("Code")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return str(code or status or "")


class ObjectStore:
    """内容寻址的原件桶：同一键只写一次，已有不同对象时拒绝覆盖。"""

    def __init__(self, client, bucket: str, guard: str = "head"):
        self.client, self.bucket, self.guard = client, bucket, guard

    def head(self, key: str) -> dict | None:
        try:
            r = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _error_code(exc) in NOT_FOUND:
                return None
            raise
        meta = {k.lower(): v for k, v in (r.get("Metadata") or {}).items()}
        return {"bytes": r.get("ContentLength"), "sha256": meta.get("sha256"),
                "etag": str(r.get("ETag", "")).strip('"').lower()}

    @staticmethod
    def same(obj: dict, sha256: str, size: int | None) -> bool:
        return obj["sha256"] == sha256 and (size is None or obj["bytes"] == size)

    def put(self, key: str, path: Path, sha256: str) -> str:
        """上传一个原件；返回 uploaded 或 exists（桶中已有相同对象）。"""
        size = path.stat().st_size
        existing = self.head(key)
        if existing is not None:
            if self.same(existing, sha256, size):
                return "exists"
            raise StoreError(f"{key} 已有不同对象（{existing['bytes']} 字节，sha256 元数据 {existing['sha256']}），拒绝覆盖")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise StoreError(f"{path} 与清单校验值不符，未上传")
        md5 = hashlib.md5(data)
        args = {"Bucket": self.bucket, "Key": key, "Body": data, "ContentMD5": base64.b64encode(md5.digest()).decode(),
                "ContentType": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "Metadata": {"sha256": sha256}}
        if self.guard == "if-none-match":
            args["IfNoneMatch"] = "*"
        try:
            self.client.put_object(**args)
        except Exception as exc:
            if _error_code(exc) not in CONFLICT:
                raise
            existing = self.head(key)   # 并发写入：桶中对象相同即视为已存在
            if existing is not None and self.same(existing, sha256, size):
                return "exists"
            raise StoreError(f"{key} 已被写入不同对象，拒绝覆盖") from None
        after = self.head(key)
        etag_is_md5 = after is not None and re.fullmatch(r"[0-9a-f]{32}", after["etag"] or "")
        if after is None or not self.same(after, sha256, size) or (etag_is_md5 and after["etag"] != md5.hexdigest()):
            raise StoreError(f"{key} 上传后核对失败")
        return "uploaded"

    def get(self, key: str, dest: Path, sha256: str):
        """下载到 dest；校验值不符时删除临时文件并报错，不留下错误文件。"""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        h = hashlib.sha256()
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
            with tmp.open("wb") as f:
                for chunk in iter(lambda: body.read(1 << 20), b""):
                    h.update(chunk)
                    f.write(chunk)
            if h.hexdigest() != sha256:
                raise StoreError(f"{key} 下载内容校验失败（得到 {h.hexdigest()[:12]}，应为 {sha256[:12]}）")
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()


def open_store(env: Mapping[str, str] | None = None) -> ObjectStore:
    config = load_config(env)
    return ObjectStore(make_client(config), config.bucket, config.guard)


# ---------- 清单中的原件 ----------

@dataclass(frozen=True)
class Original:
    owner: str        # 案例编号或资料集编号
    rel: str          # 相对数据集根（也是相对缓存根）的路径
    sha256: str
    bytes: int | None
    key: str


def manifests(datasets: Path) -> list[Path]:
    return sorted(p for p in datasets.rglob(MANIFEST)
                  if not any(part.startswith(".") or part == "入库模板" for part in p.relative_to(datasets).parts))


def originals(datasets: Path) -> list[Original]:
    """数据集中全部资料清单登记的原件；1.2 版等未写存储键的记录按校验值推出键。"""
    out = []
    for manifest in manifests(datasets):
        m = json.loads(manifest.read_text(encoding="utf-8"))
        base = manifest.parent.relative_to(datasets)
        owner = m.get("case_id") or m.get("collection_id") or base.as_posix()
        for rec in m.get("file_records", []):
            path, sha = rec.get("path") or rec.get("relative_path"), rec.get("sha256")
            if not path or not sha:
                continue
            key = (rec.get("storage") or {}).get("key") or storage_key(sha, Path(path).suffix)
            out.append(Original(owner, (base / path).as_posix(), sha, rec.get("bytes"), key))
    return out


def from_pairs(pairs: Path) -> list[Original]:
    data = json.loads(pairs.read_text(encoding="utf-8"))
    out = []
    for s in data["samples"]:
        for f in [*s["tender_files"], s["body_file"]]:
            key = f.get("storage_key") or storage_key(f["sha256"], Path(f["path"]).suffix)
            out.append(Original(s["sample_id"], f["path"], f["sha256"], None, key))
    return out


def unique_keys(items: Iterable[Original]) -> dict[str, Original]:
    out = {}
    for o in items:
        out.setdefault(o.key, o)
    return out


def local_source(o: Original, roots: Iterable[Path]) -> Path | None:
    for root in roots:
        p = root / o.rel
        if p.is_file() and (o.bytes is None or p.stat().st_size == o.bytes) and sha256_file(p) == o.sha256:
            return p
    return None


# ---------- 子命令 ----------

def push(datasets: Path, cache: Path, store: ObjectStore, items: Iterable[Original] | None = None) -> dict:
    """上传桶中尚无的原件；来源依次取缓存与数据集工作区。已有相同对象跳过，已有不同对象报错。"""
    result = {"uploaded": [], "exists": [], "missing_local": [], "errors": []}
    for key, o in unique_keys(items if items is not None else originals(datasets)).items():
        try:
            existing = store.head(key)
            if existing is not None:
                if store.same(existing, o.sha256, o.bytes):
                    result["exists"].append(key)
                else:
                    result["errors"].append({"key": key, "path": o.rel, "error": "桶中已有不同对象，拒绝覆盖"})
                continue
            source = local_source(o, (cache, datasets))
            if source is None:
                result["missing_local"].append({"key": key, "path": o.rel, "owner": o.owner})
                continue
            result[store.put(key, source, o.sha256)].append(key)
        except StoreError as exc:
            result["errors"].append({"key": key, "path": o.rel, "error": str(exc)})
    return result


def fetch(items: Iterable[Original], dest: Path, store: ObjectStore) -> dict:
    """把缺少的原件下载到 dest（默认缓存根）的案例目录结构中并核对校验值。"""
    result = {"downloaded": [], "present": [], "errors": []}
    for o in items:
        target = dest / o.rel
        if target.is_file() and sha256_file(target) == o.sha256:
            result["present"].append(o.rel)
            continue
        try:
            store.get(o.key, target, o.sha256)
            result["downloaded"].append(o.rel)
        except StoreError as exc:
            result["errors"].append({"path": o.rel, "key": o.key, "error": str(exc)})
        except Exception as exc:
            if _error_code(exc) not in NOT_FOUND:
                raise
            result["errors"].append({"path": o.rel, "key": o.key, "error": "桶中没有该对象"})
    return result


def verify(datasets: Path, store: ObjectStore) -> dict:
    """逐个存储键核对：存在、字节数与 sha256 元数据一致。"""
    result = {"checked": 0, "ok": 0, "missing": [], "mismatch": []}
    for key, o in unique_keys(originals(datasets)).items():
        result["checked"] += 1
        obj = store.head(key)
        if obj is None:
            result["missing"].append({"key": key, "path": o.rel, "owner": o.owner})
        elif not store.same(obj, o.sha256, o.bytes):
            result["mismatch"].append({"key": key, "path": o.rel, "bytes": obj["bytes"], "sha256": obj["sha256"]})
        else:
            result["ok"] += 1
    return result


def status(datasets: Path, cache: Path, env: Mapping[str, str] | None = None) -> dict:
    items = originals(datasets)
    keys = unique_keys(items)
    out = {"datasets": str(datasets), "cache": str(cache), "records": len(items), "unique_objects": len(keys),
           "in_cache": sum((cache / o.rel).is_file() for o in items),
           "in_datasets_worktree": sum((datasets / o.rel).is_file() for o in items)}
    try:
        config = load_config(env)
    except ConfigError as exc:
        out["store"] = {"configured": False, "message": str(exc)}
        return out
    out["store"] = {"configured": True, "endpoint": config.endpoint, "bucket": config.bucket,
                    "addressing": config.addressing, "write_guard": config.guard}
    store = ObjectStore(make_client(config), config.bucket, config.guard)
    out["store"]["in_bucket"] = sum(store.head(k) is not None for k in keys)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=Path, default=DATASETS, help="数据集仓库根目录，默认同 TENDER_DATASETS")
    parser.add_argument("--cache", type=Path, default=None, help="本地缓存根目录，默认同 TENDER_DATASETS_CACHE")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("push", help="上传清单登记、本地有而桶中没有的原件")
    f = sub.add_parser("fetch", help="把缺少的原件下载到缓存并核对校验值")
    scope = f.add_mutually_exclusive_group()
    scope.add_argument("--case", help="只取一个案例，如 CASE-012")
    scope.add_argument("--pairs", type=Path, help="只取评分配对集中样本的招标文件与正文")
    f.add_argument("--dest", type=Path, help="下载目标根目录，默认缓存根；指向数据集仓库可恢复工作区中的原件")
    sub.add_parser("verify", help="核对每个登记的原件在桶中存在且字节数、校验值一致")
    sub.add_parser("status", help="清单、缓存与桶的计数")
    args = parser.parse_args(argv)
    cache = args.cache or cache_root()
    try:
        if args.command == "status":
            result, failed = status(args.datasets, cache), False
        else:
            store = open_store()
            if args.command == "push":
                result = push(args.datasets, cache, store)
                failed = bool(result["errors"] or result["missing_local"])
            elif args.command == "verify":
                result = verify(args.datasets, store)
                failed = bool(result["missing"] or result["mismatch"])
            else:
                if args.pairs:
                    items = from_pairs(args.pairs)
                else:
                    items = [o for o in originals(args.datasets) if not args.case or o.owner == args.case]
                    if args.case and not items:
                        raise StoreError(f"数据集中没有 {args.case} 的原件记录")
                result = fetch(items, args.dest or cache, store)
                failed = bool(result["errors"])
    except StoreError as exc:
        print(f"对象存储操作未执行：{exc}", file=sys.stderr)
        return 2
    summary = {k: (len(v) if isinstance(v, list) else v) for k, v in result.items()}
    print(json.dumps({"command": args.command, "summary": summary, "detail": result}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
