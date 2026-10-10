"""仓库与数据集位置。

数据集默认取与本仓库同级的 ``tender-datasets``，可用环境变量 ``TENDER_DATASETS`` 覆盖。
原件不在 git 中：数据集工作目录或本地缓存（``TENDER_DATASETS_CACHE``）里有原件时才能读取，
没有时先 ``python -m tender_research.store fetch`` 取回。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
DATASETS = Path(os.environ.get("TENDER_DATASETS", REPO.parent / "tender-datasets"))
CACHE = Path(os.environ.get("TENDER_DATASETS_CACHE") or "~/.cache/tender-datasets").expanduser()

PUBLIC_RULES = DATASETS / "公开规则资料"
RULES_MANIFEST = PUBLIC_RULES / "资料清单.json"
RULES_REFERENCE = DATASETS / "评测集" / "RULE-001.json"
CASE_001 = DATASETS / "CASE-001_广德_污水管网改造及新建"
CASE_001_MANIFEST = CASE_001 / "资料清单.json"


def original(base: Path, relative: str) -> Path:
    """资料清单里一份原件的本地位置：先找数据集工作目录，再找本地缓存（同样的相对位置）。都没有时给出取回办法。"""
    path = base / relative
    if path.is_file():
        return path
    try:
        cached = CACHE / path.relative_to(DATASETS)
    except ValueError:
        cached = None
    if cached is not None and cached.is_file():
        return cached
    raise FileNotFoundError(
        f"原件不在本地：{path}。原件存放在对象存储，先运行 python -m tender_research.store fetch --dest {DATASETS} 取回"
    )


@dataclass(frozen=True)
class Workspace:
    """一次运行使用的三个位置：数据集、结果目录、含 ``.secrets/`` 的凭据目录。"""

    datasets: Path = DATASETS
    results: Path = RESULTS
    secrets: Path = REPO
