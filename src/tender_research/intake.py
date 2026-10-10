"""批量入库：样本登记表 + 收件目录 → 案例目录、资料清单 1.3、评分配对集与入库报告。

一行登记表对应一份样本（某项目／标段下某企业的一次实际提交）。原件复制到本地缓存的案例目录
（``TENDER_DATASETS_CACHE``），``--upload`` 时再按内容寻址写入对象存储；数据集仓库只写资料清单、README、
评分配对集与入库报告。收件目录中的原件不改动，已有案例目录不覆盖，同一样本重复运行跳过。
``--dry-run`` 只校验，报告打印到标准输出，不写任何文件。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

from .paths import DATASETS
from .store import ObjectStore, StoreError, cache_root, open_store, storage_key

SCHEMA_VERSION = "1.3"
PAIRS_SCHEMA = "score-pairs/1"
PAIRS_FILE = "评分配对集.json"
REPORT_FILE = "入库报告.md"
ITEMS_FILE = "分项得分.csv"
MANIFEST = "资料清单.json"

COLUMNS = ("样本编号", "项目名称", "招标编号", "标段", "地区", "工程类型", "评分办法", "投标企业", "本公司",
           "提交日期", "招标文件", "技术标正文", "技术分", "技术满分", "成绩口径", "聚合方式", "成绩来源文件",
           "成绩来源页码", "成绩来源类型", "结果", "成绩取得日期", "备注")
TAIL_COLUMNS = ("同场技术分", "入围分数线")
REQUIRED = ("项目名称", "招标编号", "地区", "工程类型", "评分办法", "投标企业", "本公司", "提交日期", "招标文件",
            "技术标正文", "技术满分", "成绩口径", "成绩来源类型")
ITEM_COLUMNS = ("样本编号或登记表行号", "评分项", "得分", "满分", "评委")

ENGINEERING_TYPES = ("市政道路", "排水管网", "综合市政", "园林", "房建", "其他")
TARGETS = {"技术总分": "technical_total", "综合总分": "overall_total"}
SOURCE_KINDS = {"官方原件": "official_document", "公司台账": "company_ledger", "口头报送": "user_statement"}
STAGES = {"入围": "shortlisted", "未入围": "not_shortlisted", "被否决": "rejected", "中标": "awarded",
          "未中标": "not_awarded", "未知": "unknown", "": "unknown"}
UNKNOWN_MARKS = ("/", "—", "-", "未公布")
ADDENDUM_NAME = re.compile("补遗|答疑|澄清|补充|修改|变更")
CASE_DIR = re.compile(r"^(CASE-(\d{3,}))_")
CASE_ID = re.compile(r"^CASE-\d{3,}$")

TEXT_LAYER_MIN_CHARS = 50          # 正文平均每页字符数下限
SHINGLE = 5                        # 字符 n-gram 长度
PERMUTATIONS = 128
MINHASH_SEED = 20260101
TEMPLATE_JACCARD = 0.5
MINHASH_SCHEME = f"char{SHINGLE}-perm{PERMUTATIONS}-seed{MINHASH_SEED}"
_PRIME = np.uint64(4294967311)     # 大于 2^32 的素数
_MASK32 = np.uint64(0xFFFFFFFF)

# 问题类别：标题、是否拒收该行、默认处理建议。报告按此顺序列出。
CATEGORIES = {
    "field_invalid": ("必填缺失或格式错误", True, "按模板说明补齐或改正后重跑"),
    "missing_file": ("缺文件", True, "把原件放进收件目录，或改正登记表中的文件名后重跑"),
    "ambiguous_file": ("文件名歧义", True, "登记表改写为相对收件目录的路径（如 子目录/文件名.pdf），或删去重复的原件"),
    "unreadable_file": ("文件无法读取", True, "确认原件未损坏、未加密后重跑"),
    "id_conflict": ("编号冲突或重复登记", True, "核对样本编号；同一样本只登记一行，编号留空可自动分配"),
    "field_conflict": ("同场成绩或入围分数线不一致", True, "同一项目／标段各行填写相同的同场技术分与入围分数线，或只在一行填写"),
    "items_invalid": ("分项得分问题", False, "按分项得分表说明改正；该条分项未写入"),
    "items_ignored": ("分项未写入", False, "已入库样本的资料清单不改写，分项需人工追加并复核"),
    "no_text_layer": ("无文字层", False, "正文需 OCR 或取得带文字层的原件；样本保留，不进入监督集"),
    "score_out_of_range": ("分数超出满分范围", False, "核对技术分与技术满分的口径；样本保留，不进入监督集"),
    "composite": ("综合分不作技术标签", False, "取得技术分后另行登记；样本可用于跑预测，不进入监督集"),
    "missing_label": ("技术分未公布", False, "保留未知，不记为零；样本可用于跑预测，不进入监督集"),
    "items_sum": ("分项合计与技术分不符", False, "核对分项是否完整、是否为最终得分"),
    "own_score_not_in_field": ("本企业分数不在同场成绩中", False, "核对同场技术分与本企业技术分"),
    "duplicate_body": ("重复正文", False, "同一正文已有样本；后入库者不形成独立样本，核对是否误登记"),
    "max_inconsistent": ("项目内满分不一致", False, "同一项目／标段的技术满分应一致，核对招标文件评分办法"),
    "project_inconsistent": ("项目内信息不一致", False, "核对同一招标编号各行的项目名称、地区、评分办法与招标文件"),
    "date_order": ("日期先后可疑", False, "成绩取得日期早于提交日期，核对日期"),
    "pending_verification": ("成绩待核验", False, "补充官方原件及页码后另行核验"),
    "archive_damaged": ("已入库文件缺失", False, "从原件恢复该文件；恢复前样本标为 missing_files"),
    "not_cached": ("缓存中缺少原件", False, "运行 python -m tender_research.store fetch 取回；模板分组使用资料清单中的签名"),
    "upload_failed": ("上传失败", False, "检查对象存储配置与网络后运行 python -m tender_research.store push"),
    "legacy_case": ("未纳入配对集的已有案例", False, "该案例资料清单不是 1.3 版，未进入评分配对集；需要时人工补登"),
}


class InputError(ValueError):
    """登记表结构或参数错误，整批不处理。"""


@dataclass
class Issue:
    category: str
    message: str
    row: int | None = None
    sample: str | None = None
    advice: str | None = None

    @property
    def blocking(self) -> bool:
        return CATEGORIES[self.category][1]


@dataclass
class FileInfo:
    path: Path
    sha256: str
    bytes: int
    pages: int | None
    page_chars: list[int] = field(default_factory=list)
    text: str = ""

    @property
    def chars(self) -> int:
        return sum(self.page_chars)

    @property
    def text_layer(self) -> bool | None:
        if not self.pages:
            return None
        return self.chars / self.pages >= TEXT_LAYER_MIN_CHARS


@dataclass
class Row:
    line: int
    cells: dict[str, str]
    issues: list[Issue] = field(default_factory=list)
    tender: list[tuple[str, FileInfo, str]] = field(default_factory=list)   # (原名, 信息, role)
    body: tuple[str, FileInfo] | None = None
    score_source: tuple[str, FileInfo] | None = None
    parsed: dict = field(default_factory=dict)
    items: list[dict] = field(default_factory=list)
    sample_id: str | None = None
    status: str = "pending"        # pending / new / skipped / rejected
    manifest: dict | None = None
    dir_name: str | None = None
    copies: list[tuple[Path, str]] = field(default_factory=list)   # (源路径, 案例内相对路径)

    def fail(self, category: str, message: str, advice: str | None = None):
        self.issues.append(Issue(category, message, self.line, self.cells.get("样本编号") or None, advice))

    @property
    def rejected(self) -> bool:
        return any(i.blocking for i in self.issues)


# ---------- 基础解析 ----------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_name(name: str) -> str:
    """文件名比较口径：全角转半角、忽略大小写与空白。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name)).casefold()


def norm_header(name: str) -> str:
    return re.sub(r"[（(].*?[）)]$", "", unicodedata.normalize("NFKC", name or "").strip()).strip()


def parse_number(raw: str) -> Decimal | None:
    """数值；未公布标记返回 None，无法解析抛 ValueError。"""
    value = unicodedata.normalize("NFKC", raw or "").strip()
    if value in UNKNOWN_MARKS:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"不是数值：{raw!r}") from None
    if not number.is_finite():
        raise ValueError(f"不是有限数值：{raw!r}")
    return number


def parse_date(raw: str) -> str:
    """接受 YYYY-MM-DD 及 Excel 常见的 YYYY/M/D、YYYY.M.D，统一为 ISO 日期。"""
    m = re.fullmatch(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", unicodedata.normalize("NFKC", raw).strip())
    if not m:
        raise ValueError(f"日期格式应为 YYYY-MM-DD：{raw!r}")
    return date(*map(int, m.groups())).isoformat()


def parse_score_list(raw: str) -> list[Decimal | None]:
    parts = [p.strip() for p in re.split(r"[;；]", unicodedata.normalize("NFKC", raw)) if p.strip()]
    return [parse_number(p) for p in parts]


def as_float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def read_csv(path: Path) -> tuple[list[str], list[tuple[int, list[str]]], str]:
    """读取 CSV；UTF-8（带或不带 BOM）优先，Excel 另存的 GB18030 也接受。行号与 Excel 行号一致。"""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise InputError(f"{path.name} 编码无法识别，请另存为 CSV UTF-8")
    records = list(csv.reader(io.StringIO(text, newline="")))
    if not records:
        raise InputError(f"{path.name} 为空")
    header = [norm_header(c) for c in records[0]]
    while header and not header[-1]:
        header.pop()
    rows = [(i + 2, [c.strip() for c in r]) for i, r in enumerate(records[1:]) if any(c.strip() for c in r)]
    return header, rows, encoding


# ---------- PDF 检查与 MinHash ----------

def inspect_file(path: Path) -> FileInfo:
    """SHA-256、字节数；PDF 另取页数与每页非空白字符数（pdfium 文字层）。"""
    info = FileInfo(path, sha256_file(path), path.stat().st_size, None)
    if path.suffix.lower() != ".pdf":
        return info
    try:
        pdf = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as exc:
        raise ValueError(f"PDF 无法打开：{exc}") from None
    try:
        texts = []
        for page in pdf:
            textpage = page.get_textpage()
            texts.append(textpage.get_text_range())
            textpage.close()
            page.close()
        info.pages = len(texts)
        info.page_chars = [len(re.sub(r"\s+", "", t)) for t in texts]
        info.text = "\n".join(texts)
    finally:
        pdf.close()
    return info


_rng = np.random.default_rng(MINHASH_SEED)
_A = _rng.integers(1, 2**31, PERMUTATIONS, dtype=np.uint64)
_B = _rng.integers(0, 2**31, PERMUTATIONS, dtype=np.uint64)


def minhash(text: str) -> np.ndarray | None:
    """字符 5-gram shingle 的 MinHash 签名（128 个置换，固定种子）；文本过短返回 None。"""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    if len(compact) < SHINGLE:
        return None
    cps = np.frombuffer(compact.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    n = len(cps) - SHINGLE + 1
    h = np.zeros(n, dtype=np.uint64)
    for k in range(SHINGLE):
        h = (h * np.uint64(1000003) + cps[k:k + n]) & _MASK32
    shingles = np.unique(h)
    sig = np.full(PERMUTATIONS, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(shingles), 4096):
        chunk = shingles[start:start + 4096, None]
        sig = np.minimum(sig, ((chunk * _A + _B) % _PRIME).min(axis=0))
    return sig


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


# ---------- 命名 ----------

_ADMIN = r"(?:省|自治区|特别行政区|自治州|地区|市|州|盟|县|区|旗)"


def region_units(region: str) -> list[str]:
    return re.findall(r".+?" + _ADMIN, region) or [region]


def region_short(region: str) -> str:
    """最末一级行政区去掉“市／县／区”等后缀：安徽省合肥市 → 合肥。"""
    last = region_units(region.strip())[-1]
    return re.sub(_ADMIN + "$", "", last) or last


def safe_name(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|_\s]+', "", text)


def project_short(name: str, region: str, limit: int = 16) -> str:
    """工程简称：去括注、地区名与“工程项目”等尾词，取主名称前段。仅用于目录名，元数据保留全名。"""
    text = re.sub(r"[（(][^）)]*[）)]", "", name)
    text = re.split(r"[-—–]", text)[0] or text
    for unit in region_units(region):
        for token in (unit, re.sub(_ADMIN + "$", "", unit)):
            if len(token) >= 2:
                text = text.replace(token, "")
    while True:
        stripped = re.sub(r"(工程项目|建设项目|施工项目|项目|工程|施工|招标|标段|一期|二期)$", "", text)
        if stripped == text or not stripped:
            break
        text = stripped
    return safe_name(text)[:limit] or "未命名工程"


def bidder_short(name: str, limit: int = 12) -> str:
    """企业简称：去公司组织形式及尾部“工程／建设”：安徽秉一市政工程有限公司 → 安徽秉一市政。"""
    text = re.sub(r"(集团股份有限公司|股份有限公司|有限责任公司|集团有限公司|有限公司|公司)$", "", name.strip())
    text = re.sub(r"(建设工程|工程|建设)$", "", text) or text
    return safe_name(text)[:limit] or "企业"


# ---------- 数据集 ----------

@dataclass
class Case:
    case_id: str
    number: int
    dir: Path
    manifest: dict | None


def scan_library(datasets: Path) -> dict[str, Case]:
    cases = {}
    if not datasets.is_dir():
        return cases
    for d in sorted(datasets.iterdir()):
        m = CASE_DIR.match(d.name)
        if not d.is_dir() or not m:
            continue
        path = d / MANIFEST
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        cases[m.group(1)] = Case(m.group(1), int(m.group(2)), d, manifest)
    return cases


def body_record(manifest: dict) -> dict | None:
    for rec in manifest.get("file_records", []):
        if rec.get("role") == "body":
            return rec
    return None


def project_key(tender_id: str, section: str | None) -> str:
    return f"{tender_id}|{section or ''}"


# ---------- 收件目录 ----------

class Inbox:
    def __init__(self, root: Path):
        self.root = root
        self.by_name: dict[str, list[Path]] = {}
        self.by_stem: dict[str, list[Path]] = {}
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.name.startswith((".", "~$")):
                self.by_name.setdefault(norm_name(p.name), []).append(p)
                self.by_stem.setdefault(norm_name(p.stem), []).append(p)
        self._info: dict[Path, FileInfo] = {}

    def locate(self, name: str) -> tuple[Path | None, str | None, str | None]:
        """按文件名在收件目录递归查找；返回 (路径, 问题类别, 说明)。同名多份且内容不同为歧义。"""
        if "/" in name or "\\" in name:
            p = (self.root / name.replace("\\", "/")).resolve()
            if p.is_file() and p.is_relative_to(self.root.resolve()):
                return p, None, None
            return None, "missing_file", f"收件目录中没有 {name}"
        key = norm_name(name)
        found = self.by_name.get(key) or self.by_stem.get(key) or []
        if not found:
            return None, "missing_file", f"收件目录中没有 {name}"
        if len(found) > 1 and len({sha256_file(p) for p in found}) > 1:
            rel = "、".join(str(p.relative_to(self.root)) for p in found)
            return None, "ambiguous_file", f"{name} 对应多份不同文件：{rel}"
        return found[0], None, None

    def info(self, path: Path) -> FileInfo:
        if path not in self._info:
            self._info[path] = inspect_file(path)
        return self._info[path]


# ---------- 逐行校验 ----------

def validate_row(row: Row, inbox: Inbox):
    c, p = row.cells, row.parsed
    for col in REQUIRED:
        if not c.get(col):
            row.fail("field_invalid", f"必填列“{col}”为空")
    if c.get("样本编号") and not CASE_ID.fullmatch(c["样本编号"]):
        row.fail("field_invalid", f"样本编号应为 CASE-三位以上数字：{c['样本编号']!r}")
    if c.get("工程类型") and c["工程类型"] not in ENGINEERING_TYPES:
        row.fail("field_invalid", f"工程类型须为 {'／'.join(ENGINEERING_TYPES)} 之一：{c['工程类型']!r}",
                 "不在列表中的填“其他”，原文写入备注")
    if c.get("本公司") and c["本公司"] not in ("是", "否"):
        row.fail("field_invalid", f"本公司须填 是／否：{c['本公司']!r}")
    for col, mapping in (("成绩口径", TARGETS), ("成绩来源类型", SOURCE_KINDS), ("结果", STAGES)):
        if c.get(col) and c[col] not in mapping:
            row.fail("field_invalid", f"{col}须为 {'／'.join(k for k in mapping if k)} 之一：{c[col]!r}")
    for col in ("提交日期", "成绩取得日期"):
        if c.get(col):
            try:
                p[col] = parse_date(c[col])
            except ValueError as exc:
                row.fail("field_invalid", f"{col}：{exc}")
    try:
        p["技术满分"] = parse_number(c.get("技术满分", "")) if c.get("技术满分") else None
        if c.get("技术满分") and (p["技术满分"] is None or p["技术满分"] <= 0):
            row.fail("field_invalid", "技术满分须为正数")
    except ValueError as exc:
        row.fail("field_invalid", f"技术满分：{exc}")
    p["技术分"] = None
    if c.get("技术分"):
        try:
            p["技术分"] = parse_number(c["技术分"])
        except ValueError as exc:
            row.fail("field_invalid", f"技术分：{exc}")
    if c.get("成绩来源页码"):
        if not re.fullmatch(r"\d+", c["成绩来源页码"]) or int(c["成绩来源页码"]) < 1:
            row.fail("field_invalid", f"成绩来源页码须为正整数（物理页）：{c['成绩来源页码']!r}")
        else:
            p["成绩来源页码"] = int(c["成绩来源页码"])
    for col in TAIL_COLUMNS:
        if c.get(col):
            try:
                values = parse_score_list(c[col]) if col == "同场技术分" else [parse_number(c[col])]
            except ValueError as exc:
                row.fail("field_invalid", f"{col}：{exc}")
                continue
            limit = p.get("技术满分")
            if limit and any(v is not None and not (0 <= v <= limit) for v in values):
                row.fail("field_invalid", f"{col}有超出 0～技术满分 的值：{c[col]}")
            p[col] = values if col == "同场技术分" else values[0]
    if p.get("提交日期") and p.get("成绩取得日期") and p["成绩取得日期"] < p["提交日期"]:
        row.issues.append(Issue("date_order", f"成绩取得日期 {p['成绩取得日期']} 早于提交日期 {p['提交日期']}",
                                row.line, c.get("样本编号") or None))
    locate_files(row, inbox)


def locate_files(row: Row, inbox: Inbox):
    c = row.cells

    def find(name: str) -> FileInfo | None:
        path, category, message = inbox.locate(name)
        if category:
            row.fail(category, message)
            return None
        try:
            return inbox.info(path)
        except (OSError, ValueError) as exc:
            row.fail("unreadable_file", f"{name}：{exc}")
            return None

    names = [n.strip() for n in re.split(r"[;；]", c.get("招标文件", "")) if n.strip()]
    for i, name in enumerate(names):
        info = find(name)
        if info:
            role = "addendum" if i > 0 and ADDENDUM_NAME.search(name) else "tender"
            row.tender.append((Path(name).name, info, role))
    if c.get("技术标正文"):
        if re.search(r"[;；]", c["技术标正文"]):
            row.fail("field_invalid", "技术标正文只能填一个文件名")
        else:
            info = find(c["技术标正文"])
            if info and info.pages is None:
                row.fail("unreadable_file", f"技术标正文须为 PDF：{c['技术标正文']}")
            elif info:
                row.body = (Path(c["技术标正文"]).name, info)
    if c.get("成绩来源文件"):
        info = find(c["成绩来源文件"])
        if info:
            row.score_source = (Path(c["成绩来源文件"]).name, info)


# ---------- 分项得分 ----------

def read_items(path: Path, rows: list[Row], issues: list[Issue]):
    header, records, _ = read_csv(path)
    if header[:len(ITEM_COLUMNS)] != list(ITEM_COLUMNS):
        raise InputError(f"{path.name} 表头应为：{'，'.join(ITEM_COLUMNS)}")
    by_line = {r.line: r for r in rows}
    by_id = {r.cells["样本编号"]: r for r in rows if r.cells.get("样本编号")}
    for line, cells in records:
        cells = (cells + [""] * len(ITEM_COLUMNS))[:len(ITEM_COLUMNS)]
        ref, item, score, maximum, evaluator = cells
        target = by_id.get(ref) or (by_line.get(int(ref)) if ref.isdigit() else None)

        def bad(message, line=line, target=target, ref=ref):
            issue = Issue("items_invalid", f"分项得分表第 {line} 行：{message}", None, ref or None)
            if target:   # 挂到对应行，编号分配后随行更新
                issue.row = target.line
                target.issues.append(issue)
            else:
                issues.append(issue)

        if target is None:
            bad(f"找不到样本 {ref!r}（填登记表中的样本编号或行号）")
            continue
        if not item:
            bad("评分项为空")
            continue
        try:
            value, top = parse_number(score), parse_number(maximum)
        except ValueError as exc:
            bad(str(exc))
            continue
        if top is None or top <= 0:
            bad("满分须为正数")
            continue
        if value is not None and not (0 <= value <= top):
            bad(f"得分 {score} 超出 0～{maximum}")
            continue
        target.items.append({"item": item, "value": as_float(value), "maximum": float(top),
                             "evaluator": evaluator or None})


def check_items(row: Row):
    final = [i for i in row.items if i["evaluator"] is None]
    total, top = row.parsed.get("技术分"), row.parsed.get("技术满分")
    if not final or total is None or not top or any(i["value"] is None for i in final):
        return
    s = sum(Decimal(str(i["value"])) for i in final)
    if abs(s - total) > Decimal("0.05") * top:
        row.issues.append(Issue("items_sum", f"最终分项合计 {s} 与技术分 {total} 相差超过满分的 5%",
                                row.line, row.sample_id))


# ---------- 资料清单 ----------

def build_manifest(row: Row, case_id: str, today: str, sheet_sha: str, code_sha: str) -> tuple[str, dict, list]:
    """返回 (目录名, 资料清单, [(源路径, 案例内相对路径)])。"""
    c, p = row.cells, row.parsed
    dir_name = f"{case_id}_{region_short(c['地区'])}_{project_short(c['项目名称'], c['地区'])}"
    records, copies = [], []

    def add(kind, role, original, info, folder, stem, document_date=None):
        file_id = f"FILE-{len(records) + 1:03d}"
        rel = f"{folder}/{case_id}_{stem}{info.path.suffix.lower()}"
        rec = {"file_id": file_id, "kind": kind, "role": role, "original_filename": original, "path": rel,
               "document_date": document_date, "bytes": info.bytes, "pages": info.pages, "sha256": info.sha256,
               "storage": {"key": storage_key(info.sha256, info.path.suffix)}}
        if role == "body":
            rec.update(text_chars=info.chars, text_layer=info.text_layer)
            sig = minhash(info.text) if info.text_layer else None
            if sig is not None:
                rec["minhash"] = {"scheme": MINHASH_SCHEME, "signature": sig.astype(">u8").tobytes().hex()}
        records.append(rec)
        copies.append((info.path, rel))
        return rec

    tender_no = addendum_no = 0
    for original, info, role in row.tender:
        if role == "tender":
            tender_no += 1
            stem = "招标文件" + (f"_分册{tender_no}" if tender_no > 1 else "") + "_日期未注明"
            add("招标文件", role, original, info, "01-招标资料", stem)
        else:
            addendum_no += 1
            add("补遗", role, original, info, "01-招标资料", f"补遗{addendum_no}_日期未注明")
    submitted = p["提交日期"]
    body = add("技术标", "body", row.body[0], row.body[1], "02-投标文件",
               f"技术标_{bidder_short(c['投标企业'])}_{submitted.replace('-', '')}", submitted)
    source = None
    if row.score_source:
        known = p.get("成绩取得日期")
        source = add("成绩来源", "score_source", row.score_source[0], row.score_source[1], "03-评审结果",
                     f"成绩来源_{known.replace('-', '') if known else '日期未注明'}", known)
    target = TARGETS[c["成绩口径"]]
    kind = SOURCE_KINDS[c["成绩来源类型"]]
    verified = kind == "official_document" and source is not None and p.get("成绩来源页码") is not None
    result = {"result_id": f"{case_id}-OFF-1", "target": target, "value": as_float(p.get("技术分")),
              "maximum": float(p["技术满分"]) if target == "technical_total" else None,
              "aggregation": c.get("聚合方式") or None, "source_kind": kind,
              "source_file_id": source["file_id"] if source else None, "source_pdf_page": p.get("成绩来源页码"),
              "known_at": p.get("成绩取得日期"), "stage_result": STAGES[c.get("结果", "")],
              "verification_status": "verified" if verified else "pending",
              "submission_file_id": body["file_id"], "submission_sha256": body["sha256"]}
    if row.items:
        result["items"] = row.items
    assessment = {"method": c["评分办法"], "max_scores": {"technical": float(p["技术满分"])},
                  "reported_results": [], "official_results": [result]}
    if p.get("同场技术分") is not None or p.get("入围分数线") is not None:
        assessment["project_field"] = {
            "technical_scores": [as_float(v) for v in p["同场技术分"]] if p.get("同场技术分") is not None else None,
            "shortlist_cutoff": as_float(p.get("入围分数线")),
            "source_kind": kind, "source_file_id": result["source_file_id"],
            "source_pdf_page": result["source_pdf_page"]}
    manifest = {
        "schema_version": SCHEMA_VERSION, "case_id": case_id, "project_name": c["项目名称"],
        "tender_id": c["招标编号"], "section": c.get("标段") or None, "region": c["地区"],
        "engineering_type": c["工程类型"], "bidder": c["投标企业"], "own_company": c["本公司"] == "是",
        "submitted_at": submitted, "registered_date": today, "remarks": c.get("备注") or None,
        "intake": {"sheet_sha256": sheet_sha, "sheet_row": row.line, "code_sha256": code_sha},
        "file_records": records, "assessment": assessment}
    return dir_name, manifest, copies


def fmt(value) -> str:
    if value is None:
        return "未公布"
    return f"{value:g}" if isinstance(value, float) else str(value)


def case_readme(dir_name: str, m: dict) -> str:
    r = m["assessment"]["official_results"][0]
    title = dir_name.split("_", 1)[1].replace("_", "")
    label = {"technical_total": "技术总分", "overall_total": "综合总分"}[r["target"]]
    source = {v: k for k, v in SOURCE_KINDS.items()}[r["source_kind"]]
    stage = {v: k for k, v in STAGES.items() if k}[r["stage_result"]]
    status = "已核验" if r["verification_status"] == "verified" else "待核验"
    max_text = fmt(m["assessment"]["max_scores"]["technical"])
    lines = [f"# {m['case_id']}｜{title}", "",
             f"状态：批量入库；{label} {fmt(r['value'])}（技术满分 {max_text}），{source}，{status}", "",
             "## 项目与文件", "", "| 字段 | 内容 |", "| --- | --- |",
             f"| 项目名称 | {m['project_name']} |",
             f"| 招标编号／标段 | {m['tender_id']}／{m['section'] or '未分标段'} |",
             f"| 地区／工程类型 | {m['region']}／{m['engineering_type']} |",
             f"| 评分办法 | {m['assessment']['method']} |",
             f"| 投标企业 | {m['bidder']}（{'本公司' if m['own_company'] else '非本公司'}） |",
             f"| 提交日期 | {m['submitted_at']} |", "",
             "| 文件编号 | 文件 | 页数 |", "| --- | --- | --- |"]
    for rec in m["file_records"]:
        lines.append(f"| {rec['file_id']} | [{rec['kind']}]({rec['path']}) | {rec['pages'] if rec['pages'] else '—'} |")
    body = next(rec for rec in m["file_records"] if rec["role"] == "body")
    lines += ["", "来源名称、校验值及结构化记录见[资料清单](资料清单.json)。"]
    if body.get("text_layer") is False:
        lines += ["", f"技术标正文无文字层（共 {body['text_chars']} 字），内容分析前需 OCR。"]
    lines += ["", "## 实际结果", "", "| 口径 | 分值 | 聚合方式 | 来源 | 出处页 | 阶段结果 | 取得日期 |",
              "| --- | --- | --- | --- | --- | --- | --- |",
              (f"| {label} | {fmt(r['value'])} | {r['aggregation'] or '—'} | {source} | {r['source_pdf_page'] or '—'} "
               f"| {stage} | {r['known_at'] or '未登记'} |"), "",
              "结果唯一结构化记录见资料清单的 `assessment.official_results`。"
              + ("综合总分不作技术标签。" if r["target"] != "technical_total" else "")]
    if m.get("remarks"):
        lines += ["", f"备注：{m['remarks']}"]
    lines += ["", f"由样本登记表第 {m['intake']['sheet_row']} 行入库（登记表 SHA-256 {m['intake']['sheet_sha256'][:12]}）。", ""]
    return "\n".join(lines)


# ---------- 评分配对集 ----------

@dataclass
class Entry:
    dir_name: str
    manifest: dict
    root: Path | None                # 已入库为缓存中的案例目录；本次入库为 None，正文取 body_info
    body_info: FileInfo | None = None
    row: int | None = None

    @property
    def case_id(self) -> str:
        return self.manifest["case_id"]

    @property
    def number(self) -> int:
        return int(self.case_id.split("-")[1])


EXCLUSION_ISSUES = {"no_text_layer": "no_text_layer", "score_out_of_range": "score_out_of_range",
                    "composite_not_technical": "composite", "missing_label": "missing_label"}


def exclusion_message(exclusion: str, m: dict, body: dict) -> str:
    r = (m["assessment"].get("official_results") or [{}])[-1]
    top = m["assessment"]["max_scores"].get("technical")
    return {"no_text_layer": f"正文 {body['pages']} 页共 {body.get('text_chars', 0)} 字，平均每页不足 {TEXT_LAYER_MIN_CHARS} 字",
            "score_out_of_range": f"技术分 {fmt(r.get('value'))} 超出 0～{fmt(top)}",
            "composite_not_technical": f"登记的是综合总分 {fmt(r.get('value'))}",
            "missing_label": "技术分登记为“/”或留空"}[exclusion]


def pick_label(manifest: dict) -> tuple[dict | None, str | None]:
    results = manifest["assessment"].get("official_results", [])
    tech = [r for r in results if r.get("target") == "technical_total"]
    if not tech:
        return None, "composite_not_technical" if results else "missing_label"
    r = tech[-1]
    if r.get("value") is None:
        return None, "missing_label"
    label = {"target": "technical_total", "value": r["value"], "maximum": r["maximum"],
             "verification_status": r["verification_status"], "source_kind": r["source_kind"],
             "known_at": r.get("known_at"), "stage_result": r.get("stage_result", "unknown"),
             "items": r.get("items", [])}
    if not 0 <= r["value"] <= (r["maximum"] or 0):
        return label, "score_out_of_range"
    return label, None


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def stored_signature(body: dict) -> np.ndarray | None:
    m = body.get("minhash") or {}
    if m.get("scheme") != MINHASH_SCHEME:
        return None
    return np.frombuffer(bytes.fromhex(m["signature"]), dtype=">u8").astype(np.uint64)


def build_pairs(entries: list[Entry], cache: Path, now: str, code_sha: str, issues: list[Issue],
                legacy_files: dict[str, str] | None = None) -> dict:
    """路径相对缓存根。legacy_files：未纳入配对集的已有案例的文件 SHA → 案例编号，正文与之相同同样记为重复。
    缓存中缺少的已入库原件不影响样本资格；缓存中存在但大小不符的记为 missing_files。"""
    entries = sorted(entries, key=lambda e: e.number)
    ids = [e.case_id for e in entries]
    uf = UnionFind(ids)
    samples, signatures, projects = {}, {}, {}
    first_sha = dict(legacy_files or {})
    not_cached = []
    for e in entries:
        m = e.manifest
        key = project_key(m["tender_id"], m.get("section"))
        body = body_record(m)
        tenders = [r for r in m["file_records"] if r.get("role") in ("tender", "addendum")]
        exclusion = None
        if e.root is not None:
            for rec in [body, *tenders]:
                path = e.root / rec["path"]
                if not path.is_file():
                    not_cached.append(f"{e.dir_name}/{rec['path']}")
                elif path.stat().st_size != rec["bytes"]:
                    exclusion = "missing_files"
                    issues.append(Issue("archive_damaged", f"文件缺失或大小不符：{e.dir_name}/{rec['path']}",
                                        e.row, e.case_id))
        label, label_exclusion = pick_label(m)
        duplicate_of = first_sha.setdefault(body["sha256"], e.case_id)
        if duplicate_of != e.case_id:
            if duplicate_of in uf.parent:
                uf.union(duplicate_of, e.case_id)
            exclusion = exclusion or "duplicate_body"
            issues.append(Issue("duplicate_body", f"技术标正文与 {duplicate_of} 相同（SHA-256 {body['sha256'][:12]}）",
                                e.row, e.case_id))
        if not body.get("text_layer"):
            exclusion = exclusion or "no_text_layer"
        exclusion = exclusion or label_exclusion
        if e.row is not None and exclusion in EXCLUSION_ISSUES:
            issues.append(Issue(EXCLUSION_ISSUES[exclusion], exclusion_message(exclusion, m, body), e.row, e.case_id))
        projects.setdefault(key, []).append(e)
        samples[e.case_id] = {
            "sample_id": e.case_id, "project_key": key, "template_group": None, "region": m["region"],
            "engineering_type": m["engineering_type"], "scoring_scheme": m["assessment"]["method"],
            "bidder": m["bidder"], "own_company": m["own_company"], "submitted_at": m["submitted_at"],
            "tender_files": [{"file_id": r["file_id"], "path": f"{e.dir_name}/{r['path']}", "sha256": r["sha256"],
                              "pages": r["pages"], "role": r["role"], "storage_key": file_key(r)} for r in tenders],
            "body_file": {"file_id": body["file_id"], "path": f"{e.dir_name}/{body['path']}",
                          "sha256": body["sha256"], "pages": body["pages"], "text_layer": bool(body.get("text_layer")),
                          "storage_key": file_key(body)},
            "label": label, "eligible": exclusion is None, "exclusion": exclusion}
        if duplicate_of != e.case_id:
            samples[e.case_id]["duplicate_of"] = duplicate_of
        if body.get("text_layer") and duplicate_of == e.case_id:
            sig = stored_signature(body)
            if sig is None:
                path = None if e.root is None else e.root / body["path"]
                info = e.body_info or (inspect_file(path) if path is not None and path.is_file() else None)
                sig = minhash(info.text) if info else None
            if sig is not None:
                signatures[e.case_id] = sig
    project_out = {}
    for key, members in projects.items():
        for other in members[1:]:
            uf.union(members[0].case_id, other.case_id)
        project_out[key] = project_summary(key, members, issues)
    keys = list(signatures)
    if keys:
        matrix = np.stack([signatures[k] for k in keys])
        for i, k in enumerate(keys[:-1]):
            similar = (matrix[i + 1:] == matrix[i]).mean(axis=1) >= TEMPLATE_JACCARD
            for j in np.nonzero(similar)[0]:
                uf.union(k, keys[i + 1 + j])
    roots = {}
    for cid in ids:   # 按编号顺序给模板组编号，组内最小编号决定组号先后
        roots.setdefault(uf.find(cid), f"TG-{len(roots) + 1:03d}")
        samples[cid]["template_group"] = roots[uf.find(cid)]
    if not_cached:
        issues.append(Issue("not_cached", f"{len(not_cached)} 个已入库原件不在本地缓存，如 {not_cached[0]}"))
    return {"schema": PAIRS_SCHEMA, "generated_at": now, "datasets_root": str(cache.resolve()),
            "code_sha256": code_sha, "samples": [samples[cid] for cid in ids], "projects": project_out}


def file_key(rec: dict) -> str:
    return (rec.get("storage") or {}).get("key") or storage_key(rec["sha256"], Path(rec["path"]).suffix)


def project_summary(key: str, members: list[Entry], issues: list[Issue]) -> dict:
    """项目级汇总：同场技术分、入围分数线及一致性检查。"""
    first = members[0].manifest
    out = {"project_name": first["project_name"], "tender_id": first["tender_id"], "section": first.get("section"),
           "sample_ids": [e.case_id for e in members], "field_scores": None, "shortlist_cutoff": None,
           "field_source": None}

    def flag(category, message):
        issues.append(Issue(category, f"{key}：{message}", next((e.row for e in members if e.row), None),
                            "、".join(e.case_id for e in members)))

    maxima = {e.manifest["assessment"]["max_scores"].get("technical") for e in members}
    if len(maxima) > 1:
        flag("max_inconsistent", f"技术满分有 {sorted(maxima)} 多种")
    for name, attr in (("评分办法", lambda m: m["assessment"]["method"]), ("项目名称", lambda m: m["project_name"]),
                       ("地区", lambda m: m["region"]),
                       ("招标文件", lambda m: tuple(sorted(r["sha256"] for r in m["file_records"]
                                                        if r.get("role") in ("tender", "addendum"))))):
        if len({attr(e.manifest) for e in members}) > 1:
            flag("project_inconsistent", f"同一项目各样本的{name}不同")
    fields = [(e, e.manifest["assessment"].get("project_field")) for e in members]
    fields = [(e, f) for e, f in fields if f]
    for attr, out_key in (("technical_scores", "field_scores"), ("shortlist_cutoff", "shortlist_cutoff")):
        given = [(e, f) for e, f in fields if f.get(attr) is not None]
        if not given:
            continue
        values = {json.dumps(sort_scores(f[attr]) if attr == "technical_scores" else f[attr]) for _, f in given}
        if len(values) > 1:
            flag("field_conflict", f"{'同场技术分' if out_key == 'field_scores' else '入围分数线'}在各样本间不一致")
            continue
        out[out_key] = given[0][1][attr]
        if out["field_source"] is None:
            f = given[0][1]
            out["field_source"] = {"sample_id": given[0][0].case_id, "source_kind": f["source_kind"],
                                   "source_file_id": f["source_file_id"], "source_pdf_page": f["source_pdf_page"]}
    if out["field_scores"] is not None:
        for e in members:
            label, _ = pick_label(e.manifest)
            if label and not any(v is not None and abs(v - label["value"]) < 0.005 for v in out["field_scores"]):
                issues.append(Issue("own_score_not_in_field", f"技术分 {fmt(label['value'])} 不在同场技术分中",
                                    e.row, e.case_id))
    return out


def sort_scores(values: list) -> list:
    return sorted(values, key=lambda v: (v is None, v or 0))


# ---------- 主流程 ----------

@dataclass
class Outcome:
    rows: list[Row]
    issues: list[Issue]
    pairs: dict
    report: str
    legacy: list[str]
    upload: dict | None = None      # 上传结果；未上传时为 None


def code_sha256() -> str:
    return sha256_file(Path(__file__))


def run(sheet: Path, inbox: Path, datasets: Path, dry_run: bool = False, items: Path | None = None,
        now: datetime | None = None, cache: Path | None = None, upload: bool = False,
        store: ObjectStore | None = None) -> Outcome:
    now = now or datetime.now(UTC).astimezone()
    cache = cache or cache_root()
    if upload and store is None:
        store = open_store()          # 未配置时在任何写入之前报错
    if not sheet.is_file():
        raise InputError(f"登记表不存在：{sheet}")
    if not inbox.is_dir():
        raise InputError(f"收件目录不存在：{inbox}")
    if items is None and (sheet.parent / ITEMS_FILE).is_file():
        items = sheet.parent / ITEMS_FILE
    header, records, _ = read_csv(sheet)
    expected = list(COLUMNS)
    if header[:len(COLUMNS)] != expected or header[len(COLUMNS):] != list(TAIL_COLUMNS[:len(header) - len(COLUMNS)]):
        raise InputError("登记表表头与模板不一致；应依次为：" + "，".join(COLUMNS + TAIL_COLUMNS) + "（最后两列可省略）")
    sheet_sha, code_sha = sha256_file(sheet), code_sha256()
    library = scan_library(datasets)
    box = Inbox(inbox)
    rows = [Row(line, dict(zip(header, cells + [""] * (len(header) - len(cells))))) for line, cells in records]
    issues: list[Issue] = []
    for row in rows:
        validate_row(row, box)
    if items:
        read_items(items, rows, issues)
    resolve_identity(rows, library)
    check_fields(rows, library)
    assign_numbers(rows, library, now.date().isoformat(), sheet_sha, code_sha)
    for row in rows:
        if row.items and row.status == "skipped":
            row.issues.append(Issue("items_ignored", f"{row.sample_id} 已入库，分项得分 {len(row.items)} 条未写入",
                                    row.line, row.sample_id))
    entries, legacy, legacy_files = [], [], {}
    for case in library.values():
        if case.manifest and case.manifest.get("schema_version") == SCHEMA_VERSION:
            row = next((r.line for r in rows if r.sample_id == case.case_id), None)
            entries.append(Entry(case.dir.name, case.manifest, cache / case.dir.name, row=row))
        else:
            legacy.append(case.dir.name)
            for rec in (case.manifest or {}).get("file_records", []):
                if rec.get("sha256"):
                    legacy_files.setdefault(rec["sha256"], case.case_id)
            issues.append(Issue("legacy_case", f"{case.dir.name}（资料清单版本 "
                                f"{(case.manifest or {}).get('schema_version', '缺失')}）", sample=case.case_id))
    for row in rows:
        if row.status == "new":
            entries.append(Entry(row.dir_name, row.manifest, None, row.body[1], row.line))
    for row in rows:
        issues.extend(row.issues)
    pairs = build_pairs(entries, cache, now.isoformat(timespec="seconds"), code_sha, issues, legacy_files)
    for row in rows:
        if row.status == "new":
            r = row.manifest["assessment"]["official_results"][0]
            if r["verification_status"] == "pending" and r["value"] is not None:
                issues.append(Issue("pending_verification", "成绩来源不是带页码的官方原件", row.line, row.sample_id))
    outcome = Outcome(rows, issues, pairs, "", legacy)
    if not dry_run:
        datasets.mkdir(parents=True, exist_ok=True)
        for row in rows:
            if row.status == "new":
                archive(row, datasets, cache)
        if upload:
            outcome.upload = upload_new(rows, cache, store, issues)
    outcome.report = render_report(outcome, sheet, sheet_sha, items, inbox, datasets, cache, dry_run, now)
    if not dry_run:
        write_atomic(datasets / PAIRS_FILE, json.dumps(pairs, ensure_ascii=False, indent=2) + "\n")
        write_atomic(datasets / REPORT_FILE, outcome.report)
    return outcome


def new_files(rows: list[Row]) -> list[tuple[Row, dict]]:
    return [(r, rec) for r in rows if r.status == "new" for rec in r.manifest["file_records"]]


def upload_new(rows: list[Row], cache: Path, store: ObjectStore, issues: list[Issue]) -> dict:
    """上传本次入库的原件；同一存储键只处理一次，桶中已有相同对象跳过。"""
    result, done = {"uploaded": [], "exists": [], "failed": []}, set()
    for row, rec in new_files(rows):
        key = rec["storage"]["key"]
        if key in done:
            continue
        done.add(key)
        try:
            result[store.put(key, cache / row.dir_name / rec["path"], rec["sha256"])].append(key)
        except StoreError as exc:
            result["failed"].append(key)
            issues.append(Issue("upload_failed", f"{rec['path']}：{exc}", row.line, row.sample_id))
    return result


def resolve_identity(rows: list[Row], library: dict[str, Case]):
    """幂等判定：同编号＋同正文 SHA，或编号留空时同项目＋同企业＋同正文 SHA，视为已入库。"""
    seen_ids, seen_keys = {}, {}
    for row in rows:
        if row.rejected:
            row.status = "rejected"
            continue
        c, sha = row.cells, row.body[1].sha256
        identity = (c["招标编号"], c.get("标段", ""), c["投标企业"], sha)
        sid = c.get("样本编号")
        if sid:
            if sid in seen_ids:
                row.fail("id_conflict", f"样本编号 {sid} 与第 {seen_ids[sid]} 行重复")
            elif sid in library:
                m = library[sid].manifest
                rec = body_record(m) if m and m.get("schema_version") == SCHEMA_VERSION else None
                if rec and rec["sha256"] == sha:
                    row.status, row.sample_id = "skipped", sid
                else:
                    row.fail("id_conflict", f"{sid} 已存在（{library[sid].dir.name}），正文不同或不是批量入库的案例",
                             "编号留空自动分配，或核对是否登记错行")
            seen_ids.setdefault(sid, row.line)
        else:
            for case in library.values():
                m = case.manifest
                if not m or m.get("schema_version") != SCHEMA_VERSION:
                    continue
                rec = body_record(m)
                if rec and (m["tender_id"], m.get("section") or "", m["bidder"], rec["sha256"]) == identity:
                    row.status, row.sample_id = "skipped", case.case_id
                    break
        if row.status == "pending" and not row.rejected:
            if identity in seen_keys:
                row.fail("id_conflict", f"与第 {seen_keys[identity]} 行是同一项目、同一企业、同一正文的重复登记")
            seen_keys.setdefault(identity, row.line)
        if row.rejected:
            row.status = "rejected"


def check_fields(rows: list[Row], library: dict[str, Case]):
    """同场技术分与入围分数线：同一项目内须一致；与库中已有值或本表其他行冲突的行拒收。"""
    for col, attr in (("同场技术分", "technical_scores"), ("入围分数线", "shortlist_cutoff")):
        def canon(v, col=col):
            return json.dumps(sort_scores([as_float(x) for x in v]) if col == "同场技术分" else as_float(v))
        known = {}
        for case in library.values():
            m = case.manifest
            f = (m or {}).get("assessment", {}).get("project_field") if m and m.get("schema_version") == SCHEMA_VERSION else None
            if f and f.get(attr) is not None:
                value = f[attr]
                known.setdefault(project_key(m["tender_id"], m.get("section")),
                                 json.dumps(sort_scores(value) if col == "同场技术分" else value))
        pending = {}
        for row in rows:
            if row.status == "pending" and row.parsed.get(col) is not None:
                key = project_key(row.cells["招标编号"], row.cells.get("标段"))
                pending.setdefault(key, []).append(row)
        for key, group in pending.items():
            values = {canon(r.parsed[col]) for r in group}
            for row in group:
                if key in known and canon(row.parsed[col]) != known[key]:
                    row.fail("field_conflict", f"{col}与库中同项目已登记的值不同")
                elif key not in known and len(values) > 1:
                    row.fail("field_conflict", f"{col}与本表同项目其他行不同（第 {'、'.join(str(r.line) for r in group)} 行）")
                if row.rejected:
                    row.status = "rejected"


def assign_numbers(rows: list[Row], library: dict[str, Case], today: str, sheet_sha: str, code_sha: str):
    explicit = [int(r.cells["样本编号"].split("-")[1]) for r in rows if r.cells.get("样本编号") and not r.rejected]
    next_no = max([c.number for c in library.values()] + explicit + [0]) + 1
    for row in rows:
        if row.status != "pending":
            continue
        if row.cells.get("样本编号"):
            row.sample_id = row.cells["样本编号"]
        else:
            row.sample_id, next_no = f"CASE-{next_no:03d}", next_no + 1
        check_items(row)
        row.dir_name, row.manifest, row.copies = build_manifest(row, row.sample_id, today, sheet_sha, code_sha)
        row.status = "new"
        for issue in row.issues:
            issue.sample = row.sample_id


def archive(row: Row, datasets: Path, cache: Path):
    """原件逐个复制到缓存的案例目录并核对 SHA（已有相同文件保留，不同则报错）；
    资料清单与 README 先写入临时目录，再整体改名为数据集仓库中的案例目录。已有案例目录不动。"""
    final = datasets / row.dir_name
    if final.exists() or any(datasets.glob(f"{row.sample_id}_*")):
        raise InputError(f"案例目录已存在，不覆盖：{final.name}")
    for (source, rel), rec in zip(row.copies, row.manifest["file_records"]):
        target = cache / row.dir_name / rel
        if target.exists():
            if sha256_file(target) != rec["sha256"]:
                raise InputError(f"缓存中已有不同文件，不覆盖：{target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(f".{target.name}.part")
        try:
            shutil.copyfile(source, part)
            if sha256_file(part) != rec["sha256"]:
                raise OSError(f"复制后校验失败：{rel}")
            os.replace(part, target)
        finally:
            if part.exists():
                part.unlink()
    tmp = datasets / f".{row.dir_name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        tmp.mkdir()
        (tmp / MANIFEST).write_text(json.dumps(row.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (tmp / "README.md").write_text(case_readme(row.dir_name, row.manifest), encoding="utf-8")
        os.rename(tmp, final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def write_atomic(path: Path, text: str):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------- 报告 ----------

def cell(text) -> str:
    return str(text if text not in (None, "") else "—").replace("|", "／").replace("\n", " ")


def render_report(o: Outcome, sheet: Path, sheet_sha: str, items: Path | None, inbox: Path, datasets: Path,
                  cache: Path, dry_run: bool, now: datetime) -> str:
    rows, pairs = o.rows, o.pairs
    by_status = {s: [r for r in rows if r.status == s] for s in ("new", "skipped", "rejected")}
    samples = pairs["samples"]
    excl = {}
    for s in samples:
        if s["exclusion"]:
            excl[s["exclusion"]] = excl.get(s["exclusion"], 0) + 1
    lines = ["# 入库报告", "", "| 项目 | 内容 |", "| --- | --- |",
             f"| 生成时间 | {now.isoformat(timespec='seconds')} |",
             f"| 模式 | {'试运行（--dry-run），未写入任何文件' if dry_run else '正式入库'} |",
             f"| 登记表 | {sheet}（SHA-256 {sheet_sha[:12]}） |",
             f"| 分项得分表 | {items if items else '无'} |",
             f"| 收件目录 | {inbox} |", f"| 数据集 | {datasets} |", f"| 原件缓存 | {cache} |", "",
             "## 汇总", "",
             "| 登记行 | 通过 | 其中本次入库 | 其中已入库跳过 | 拒收 |", "| --- | --- | --- | --- | --- |",
             (f"| {len(rows)} | {len(by_status['new']) + len(by_status['skipped'])} | {len(by_status['new'])} "
              f"| {len(by_status['skipped'])} | {len(by_status['rejected'])} |"), "",
             (f"评分配对集共 {len(samples)} 个样本，进入监督集 {sum(s['eligible'] for s in samples)} 个，"
              f"项目 {len(pairs['projects'])} 个，模板组 {len({s['template_group'] for s in samples})} 个。")]
    if excl:
        lines.append("未进入监督集：" + "，".join(f"{k} {v}" for k, v in sorted(excl.items())) + "。")
    if by_status["new"]:
        status = {s["sample_id"]: s for s in samples}
        lines += ["", "## 本次入库" + ("（计划）" if dry_run else ""), "",
                  "| 行号 | 样本编号 | 目录 | 投标企业 | 技术分／满分 | 模板组 | 监督集 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in by_status["new"]:
            s = status[r.sample_id]
            res = r.manifest["assessment"]["official_results"][0]
            score = f"{fmt(res['value'])}／{fmt(r.manifest['assessment']['max_scores']['technical'])}"
            lines.append(f"| {r.line} | {r.sample_id} | {cell(r.dir_name)} | {cell(r.cells['投标企业'])} | {score} "
                         f"| {s['template_group']} | {'是' if s['eligible'] else '否（' + s['exclusion'] + '）'} |")
    if by_status["skipped"]:
        lines += ["", "## 已入库（跳过）", "", "编号留空的行建议回填下列样本编号。", "", "| 行号 | 样本编号 |", "| --- | --- |"]
        lines += [f"| {r.line} | {r.sample_id} |" for r in by_status["skipped"]]
    files = new_files(rows)
    if files:
        lines += ["", "## 对象存储", ""]
        if dry_run:
            lines.append(f"试运行不复制、不上传；正式入库后有 {len(files)} 个原件需上传。")
        elif o.upload is not None:
            lines.append(f"本次上传 {len(o.upload['uploaded'])} 个对象，桶中已有相同对象 {len(o.upload['exists'])} 个，"
                         f"失败 {len(o.upload['failed'])} 个（见问题清单）。")
        else:
            lines += [(f"以下 {len(files)} 个原件只在本地缓存，待上传：运行 `python -m tender_research.store push`，"
                       "再用 `python -m tender_research.store verify` 核对。"), "",
                      "| 样本编号 | 文件 | 存储键 |", "| --- | --- | --- |"]
            lines += [f"| {r.sample_id} | {cell(r.dir_name + '/' + rec['path'])} | {rec['storage']['key']} |"
                      for r, rec in files]
    lines += ["", "## 问题清单", ""]
    if not o.issues:
        lines.append("无。")
    for category, (title, blocking, advice) in CATEGORIES.items():
        found = [i for i in o.issues if i.category == category]
        if not found:
            continue
        lines += [f"### {title}（{len(found)}{'，拒收' if blocking else ''}）", "",
                  "| 行号 | 样本 | 说明 | 处理建议 |", "| --- | --- | --- | --- |"]
        for i in sorted(found, key=lambda i: (i.row or 0, i.sample or "")):
            lines.append(f"| {cell(i.row)} | {cell(i.sample)} | {cell(i.message)} | {cell(i.advice or advice)} |")
        lines.append("")
    if lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, required=True, help="样本登记表.csv")
    parser.add_argument("--inbox", type=Path, required=True, help="收件目录（原件平铺或任意子目录）")
    parser.add_argument("--datasets", type=Path, default=DATASETS, help="数据集根目录，默认同 TENDER_DATASETS")
    parser.add_argument("--items", type=Path, help="分项得分.csv；缺省时取登记表同目录下的同名文件")
    parser.add_argument("--cache", type=Path, help="原件缓存根目录，默认同 TENDER_DATASETS_CACHE（~/.cache/tender-datasets）")
    parser.add_argument("--upload", action="store_true", help="把本次入库的原件上传到对象存储（需 TENDER_STORE_* 配置）")
    parser.add_argument("--dry-run", action="store_true", help="只校验并打印报告，不写任何文件")
    args = parser.parse_args(argv)
    try:
        outcome = run(args.sheet, args.inbox, args.datasets, args.dry_run, args.items,
                      cache=args.cache, upload=args.upload)
    except (InputError, StoreError, OSError) as exc:
        print(f"入库未执行：{exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(outcome.report)
    else:
        new = sum(r.status == "new" for r in outcome.rows)
        print(f"入库 {new} 个样本；报告：{args.datasets / REPORT_FILE}；配对集：{args.datasets / PAIRS_FILE}")
    failed_upload = bool(outcome.upload and outcome.upload["failed"])
    return 1 if failed_upload or any(r.status == "rejected" for r in outcome.rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
