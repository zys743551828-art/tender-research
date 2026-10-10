"""技术标正文文本臂：抽取正文、字符 n-gram 哈希 TF-IDF、余弦相似度与训练折内近邻目标。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

CHUNK_BITS = 14


def body_text(pdf: Path, sha256: str, cache: Path) -> tuple[str | None, str]:
    """按 SHA-256 缓存正文文字；原件缺失或校验值不符时不抽取。"""
    cached = cache / f"{sha256}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8"), "cached"
    if not pdf.is_file():
        return None, "body_pdf_missing"
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != sha256:
        return None, "body_sha256_mismatch"
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
    cache.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text, "extracted"


def ngram_counts(text: str, ngrams=(2, 3), bits: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """去空白后的字符 n-gram 计数，确定性多项式哈希到 2^bits 维（与进程哈希种子无关）。"""
    chars = np.frombuffer(re.sub(r"\s+", "", text).encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    mask, parts = np.uint64((1 << bits) - 1), []
    with np.errstate(over="ignore"):
        for n in ngrams:
            if len(chars) < n:
                continue
            h = np.full(len(chars) - n + 1, np.uint64(0x9E3779B97F4A7C15 ^ n))
            for j in range(n):
                h = (h * np.uint64(1000003)) ^ chars[j:len(chars) - n + 1 + j]
            h ^= h >> np.uint64(29)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(32)
            parts.append(h & mask)
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0)
    idx, counts = np.unique(np.concatenate(parts), return_counts=True)
    return idx.astype(np.int64), counts.astype(float)


def similarity_matrix(docs: list[tuple[np.ndarray, np.ndarray] | None], bits: int = 18) -> np.ndarray:
    """对数词频 × IDF、L2 归一后的余弦相似度；无文字的样本整行为 NaN。

    IDF 只用正文文字（不含任何标签），近邻目标在训练折内另行计算。
    """
    present = [i for i, d in enumerate(docs) if d is not None and len(d[0])]
    n = len(docs)
    sim = np.full((n, n), np.nan)
    if not present:
        return sim
    df = np.zeros(1 << bits)
    for i in present:
        df[docs[i][0]] += 1
    idf = np.log((1 + len(present)) / (1 + df)) + 1
    vectors = []
    for i in present:
        idx, tf = docs[i]
        w = (1 + np.log(tf)) * idf[idx]
        vectors.append((idx, w / np.linalg.norm(w)))
    gram = np.zeros((len(present), len(present)))
    width = 1 << min(CHUNK_BITS, bits)
    for start in range(0, 1 << bits, width):
        block = np.zeros((len(present), width), dtype=np.float32)
        for row, (idx, w) in enumerate(vectors):
            lo, hi = np.searchsorted(idx, [start, start + width])
            block[row, idx[lo:hi] - start] = w[lo:hi]
        gram += (block @ block.T).astype(float)
    sim[np.ix_(present, present)] = np.clip(gram, -1, 1)
    return sim


def knn_targets(sim: np.ndarray, y: np.ndarray, query: np.ndarray, pool: np.ndarray, groups: np.ndarray,
                k: int) -> np.ndarray:
    """查询样本取池内相似度最高的 k 个近邻，按相似度加权平均其目标；同组近邻一律排除。

    pool 只含训练折样本；训练折自身的特征按“留本组外”计算，与测试折口径一致。无文字时为 NaN。
    """
    out = np.full(len(query), np.nan)
    for row, i in enumerate(query):
        if np.isnan(sim[i, i]):
            continue
        candidates = pool[(groups[pool] != groups[i]) & ~np.isnan(sim[i, pool])]
        if not len(candidates):
            continue
        s = sim[i, candidates]
        top = np.argsort(-s, kind="stable")[:k]
        w = np.clip(s[top], 0, None)
        out[row] = float(np.average(y[candidates[top]], weights=w)) if w.sum() > 0 else float(y[candidates[top]].mean())
    return out
