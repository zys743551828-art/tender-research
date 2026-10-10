"""本地 PDF 证据检索基线；不生成评分、规则结论或工程意见。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pypdfium2
from pypdf import PdfReader

from . import paths

VERSION = "1.0.0"
PROJECT_ID = re.compile(r"(?<![A-Za-z0-9])20\d{2}[A-Z]{3,10}\d{4,}(?:-\d+)?(?![A-Za-z0-9])")


class InputError(ValueError):
    """输入原件与清单不符或无法安全解析。"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    # 仅用于检索和摘录比对；输出证据始终保留抽取的原文。
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def terms(text: str) -> Counter:
    """中文二字片段＋完整英数词；不包含项目专属答案或页码。"""
    value = normalized(text)
    tokens = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", value)
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            tokens.append(run)
    return Counter(tokens)


@dataclass(frozen=True)
class Document:
    file_id: str
    path: Path
    source_path: str
    sha256: str
    source_ids: tuple[str, ...]
    pages: tuple[str, ...]
    project_ids: tuple[str, ...]
    image_pages: tuple[int, ...] | None = None
    rendered_blank_pages: tuple[int, ...] = ()

    def summary(self) -> dict:
        sparse = [i + 1 for i, page in enumerate(self.pages) if len(normalized(page)) < 24]
        return {
            "file_id": self.file_id,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "pages": len(self.pages),
            "project_ids": list(self.project_ids),
            "identity_status": "not_identified" if not self.project_ids else (
                "identified" if len(self.project_ids) == 1 else "ambiguous"
            ),
            "text_sparse_pdf_pages": sparse,
            "ocr_required_candidate_pages": None if self.image_pages is None else [p for p in sparse if p in self.image_pages],
            "rendered_blank_pages": list(self.rendered_blank_pages),
        }


@dataclass(frozen=True)
class Chunk:
    document: Document
    pdf_page: int
    start_char: int
    end_char: int
    text: str

    def evidence(self, score: float) -> dict:
        return {
            "file_id": self.document.file_id,
            "source_ids": list(self.document.source_ids),
            "source_path": self.document.source_path,
            "source_sha256": self.document.sha256,
            "pdf_page": self.pdf_page,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote": self.text,
            "retrieval_score": round(score, 6),
            "kind": "source_excerpt",
        }


def load_documents(manifest_path: Path, extractor: str = "pdfplumber-render-guard") -> list[Document]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("file_records", [])
    if not records:
        raise InputError("资料清单没有文件记录")
    docs = []
    seen = set()
    for row in records:
        fid = row["file_id"]
        relative_path = row.get("relative_path", row.get("path"))
        if not isinstance(relative_path, str) or not relative_path:
            raise InputError(f"文件路径缺失：{fid}")
        if fid in seen:
            raise InputError(f"文件编号重复：{fid}")
        seen.add(fid)
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise InputError(f"文件路径超出原件库：{fid}")
        try:
            path = paths.original(root, relative_path)
        except FileNotFoundError as exc:
            raise InputError(str(exc)) from exc
        data = path.read_bytes()
        if digest(data) != row["sha256"] or len(data) != row["bytes"]:
            raise InputError(f"原件校验不符：{fid}")
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise InputError(f"加密 PDF 尚未支持：{fid}")
        if len(reader.pages) != row["pages"]:
            raise InputError(f"原件页数不符：{fid}")
        image_pages, rendered_blank = None, []
        if extractor == "pypdf":
            pages = tuple(page.extract_text() or "" for page in reader.pages)
        elif extractor in ("pdfplumber-crop", "pdfplumber-render-guard"):
            visible_pages = []
            image_pages = []
            rendered = pypdfium2.PdfDocument(path) if extractor == "pdfplumber-render-guard" else None
            with pdfplumber.open(path) as pdf:
                for number, page in enumerate(pdf.pages, 1):
                    clipped = page.crop(page.bbox)
                    text = clipped.extract_text() or ""
                    if clipped.images:
                        image_pages.append(number)
                    # 扫描页、短标题页和空白页不混为一类；只对稀疏页检查完全无墨迹。
                    # 不能据此声称常规密集页上的全部字符都像素可见。
                    if rendered is not None and len(normalized(text)) < 24:
                        image = rendered[number - 1].render(scale=1).to_pil().convert("L")
                        if sum(image.histogram()[:200]) == 0:
                            text = ""
                            rendered_blank.append(number)
                    visible_pages.append(text)
                    page.close()
            if rendered is not None:
                rendered.close()
            pages = tuple(visible_pages)
            if len(pages) != len(reader.pages):
                raise InputError(f"抽取页数与 PDF 不符：{fid}")
        else:
            raise InputError(f"未知抽取方式：{extractor}")
        # 只从前三个 PDF 页读取项目编号，避免合同示例／后文历史项目污染关联。
        header = unicodedata.normalize("NFKC", "\n".join(pages[:3])).upper()
        ids = tuple(sorted(set(PROJECT_ID.findall(header))))
        docs.append(Document(fid, path, relative_path, row["sha256"],
                             tuple(row.get("source_ids", [])), pages, ids,
                             None if image_pages is None else tuple(image_pages), tuple(rendered_blank)))
    return docs


def chunks_for(docs: list[Document], size: int = 900, overlap: int = 180) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("要求窗口大小为正且 0≤重叠<窗口大小")
    chunks = []
    for doc in docs:
        for page_no, text in enumerate(doc.pages, 1):
            for start in range(0, len(text), size - overlap):
                end = min(start + size, len(text))
                if normalized(text[start:end]):
                    chunks.append(Chunk(doc, page_no, start, end, text[start:end]))
                if end == len(text):
                    break
    return chunks


class EvidenceIndex:
    def __init__(self, docs: list[Document], size: int = 900, overlap: int = 180):
        self.documents = docs
        self.chunks = chunks_for(docs, size, overlap)
        self.counts = [terms(c.text) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = Counter(token for row in self.counts for token in row)

    def search(self, query: str, *, project_id: str | None = None,
               file_ids: list[str] | None = None, top_k: int = 5) -> list[dict]:
        if not 1 <= top_k <= 30:
            raise ValueError("top_k 必须在 1—30 之间")
        q = terms(query)
        if not q:
            return []
        project = project_id.upper() if project_id else None
        candidates = []
        n = len(self.chunks)
        for i, chunk in enumerate(self.chunks):
            if project and chunk.document.project_ids != (project,):
                continue
            if file_ids is not None and chunk.document.file_id not in file_ids:
                continue
            score = 0.0
            for token in q:
                tf = self.counts[i].get(token, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self.df[token] + 0.5) / (self.df[token] + 0.5))
                denom = tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / max(self.avg_length, 1))
                score += idf * tf * 2.5 / denom
            if score > 0:
                candidates.append((score, i))
        candidates.sort(key=lambda x: (-x[0], self.chunks[x[1]].document.file_id,
                                        self.chunks[x[1]].pdf_page, self.chunks[x[1]].start_char))
        # 同页最多返回一个片段，避免一个页面的重复窗口挤掉其他资料版本。
        hits, seen = [], set()
        for score, i in candidates:
            chunk = self.chunks[i]
            key = (chunk.document.file_id, chunk.pdf_page)
            if key not in seen:
                hits.append(chunk.evidence(score)); seen.add(key)
            if len(hits) == top_k:
                break
        return hits


def check_evidence(hit: dict, docs: list[Document]) -> tuple[bool, str]:
    doc = next((d for d in docs if d.file_id == hit.get("file_id")), None)
    if not doc or hit.get("source_sha256") != doc.sha256:
        return False, "原件身份或校验不符"
    if hit.get("source_path") != doc.source_path:
        return False, "原件路径不符"
    p, start, end = hit.get("pdf_page"), hit.get("start_char"), hit.get("end_char")
    if any(type(x) is not int for x in (p, start, end)) or not 1 <= p <= len(doc.pages):
        return False, "页码或字符位置无效"
    text = doc.pages[p - 1]
    if not 0 <= start < end <= len(text) or not hit.get("quote"):
        return False, "字符位置或摘录为空"
    if hit["quote"] != text[start:end]:
        return False, "摘录与该页原文不一致"
    return True, "抽取文本一致；尚未证明语义结论成立"


def runtime_info() -> dict:
    return {"version": VERSION, "python": sys.version.split()[0],
            "pypdf": importlib.metadata.version("pypdf"),
            "pdfplumber": importlib.metadata.version("pdfplumber"),
            "pypdfium2": importlib.metadata.version("pypdfium2"),
            "pdfminer.six": importlib.metadata.version("pdfminer.six")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extractor", choices=["pypdf", "pdfplumber-crop", "pdfplumber-render-guard"], default="pdfplumber-render-guard")
    parser.add_argument("--query")
    parser.add_argument("--project")
    parser.add_argument("--file", action="append", dest="file_ids")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    started = time.perf_counter()
    docs = load_documents(args.manifest, args.extractor)
    result = {"runtime": runtime_info(), "extractor": args.extractor,
              "documents": [d.summary() for d in docs]}
    if args.query:
        index = EvidenceIndex(docs)
        hits = index.search(args.query, project_id=args.project,
                            file_ids=args.file_ids, top_k=args.top_k)
        result.update({"query": args.query, "status": "evidence_candidates" if hits else "not_found",
                       "evidence": hits, "semantic_rule_status": "not_evaluated"})
    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InputError, OSError, ValueError) as exc:
        print(json.dumps({"status": "input_error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc
