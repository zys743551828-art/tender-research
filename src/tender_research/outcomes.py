"""有线表格的保守成绩导入；保留斜杠和阶段，不制造正文配对。"""
from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pdfplumber

from . import paths
from .rule_evidence import normalized


def score_cell(raw: str | None, maximum: float) -> dict:
    value = normalized(raw or "")
    if value in ("", "/", "—", "-", "未公布"):
        return {"raw": raw, "status": "not_disclosed", "value": None}
    try:
        number = Decimal(value)
    except InvalidOperation:
        return {"raw": raw, "status": "needs_review", "value": None}
    if not number.is_finite() or not Decimal(0) <= number <= Decimal(str(maximum)):
        return {"raw": raw, "status": "needs_review", "value": None}
    return {"raw": raw, "status": "disclosed", "value": float(number)}


def classify_stage(first_page: str) -> str:
    # 长项目名可能跨两行；只读标题区，不从正文提及候选推断公告阶段。
    lines = first_page.splitlines()
    title = normalized(lines[0]) if lines else ""
    if "公告" not in title and "公示" not in title:
        title += normalized(lines[1]) if len(lines) > 1 else ""
    if "中标候选人" in title:
        return "candidate_publication"
    if "中标结果" in title or "中标公告" in title:
        return "final_award_publication_requires_identity_review"
    return "unknown"


def extract_assessment(path: Path, file_id: str, expected_sha: str, maximum: float) -> dict:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("评标原件校验失败")
    records, errors, excluded_tables, columns = [], [], [], None
    with pdfplumber.open(path) as pdf:
        for page_no, raw_page in enumerate(pdf.pages, 1):
            page = raw_page.crop(raw_page.bbox)
            for table in page.find_tables():
                rows = table.extract()
                if rows and max(len(row) for row in rows) != 13:
                    excluded_tables.append({"pdf_page": page_no, "table_bbox": list(table.bbox),
                                            "reason": "not_supported_13_column_score_table", "column_count": max(len(row) for row in rows)})
                    continue
                for row in rows:
                    compact = [normalized(cell or "") for cell in row]
                    if "技术文件详细评审得分" in compact:
                        needed = ("投标单位", "技术文件详细评审得分", "入围报价文件评审是/否")
                        if not all(c in compact for c in needed):
                            errors.append({"page": page_no, "reason": "header_not_supported"}); continue
                        columns = {key: compact.index(key) for key in needed}
                    if not row or not re.fullmatch(r"\d+", (row[0] or "").strip()):
                        continue
                    if columns is None or len(row) <= max(columns.values()):
                        errors.append({"page": page_no, "reason": "missing_or_short_header", "row": row}); continue
                    cell = score_cell(row[columns["技术文件详细评审得分"]], maximum)
                    company = re.sub(r"\s+", "", row[columns["投标单位"]] or "")
                    if not company:
                        errors.append({"page": page_no, "reason": "empty_bidder"}); continue
                    records.append({"row_number": int(row[0]), "bidder": company,
                                    "technical_total": cell, "maximum": maximum,
                                    "quote_review_admitted_raw": row[columns["入围报价文件评审是/否"]],
                                    "source": {"file_id": file_id, "sha256": expected_sha, "pdf_page": page_no,
                                               "table_bbox": list(table.bbox), "raw_cells": row},
                                    "body_pair_status": "not_available", "pure_ai_score": None,
                                    "aggregation": "not_inferred_from_expert_columns"})
            # 后页只有4列的否决表，不能沿用13列表头强行导入成绩。
            page.close()
    numbers = [r["row_number"] for r in records]
    if len(numbers) != len(set(numbers)):
        errors.append({"reason": "duplicate_row_numbers"})
    return {"records": records, "errors": errors, "excluded_tables": excluded_tables,
            "summary": {"rows": len(records), "disclosed_scores": sum(r["technical_total"]["status"] == "disclosed" for r in records),
                        "unknown_scores": sum(r["technical_total"]["status"] == "not_disclosed" for r in records)}}


def run(datasets: Path, output: Path, experiment_id="outcome-import"):
    archive = datasets / "公开规则资料"
    manifest = json.loads((archive / "资料清单.json").read_text())
    record = next(r for r in manifest["file_records"] if r["file_id"] == "PUB-006")
    assessment = extract_assessment(paths.original(archive, record["relative_path"]), record["file_id"], record["sha256"], 100.0)
    stages = []
    for fid in ("PUB-007", "PUB-012"):
        row = next(r for r in manifest["file_records"] if r["file_id"] == fid)
        path = paths.original(archive, row["relative_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("公示原件校验失败")
        with pdfplumber.open(path) as pdf:
            text = pdf.pages[0].crop(pdf.pages[0].bbox).extract_text() or ""
        stages.append({"file_id": fid, "sha256": row["sha256"], "pdf_page": 1,
                       "stage": classify_stage(text), "counted_as_won": False})
    disclosed = [r["technical_total"]["value"] for r in assessment["records"] if r["technical_total"]["status"] == "disclosed"]
    checks = {"all_208_rows_unique": len(assessment["records"]) == 208 and len({r["row_number"] for r in assessment["records"]}) == 208,
              "five_disclosed_203_unknown": assessment["summary"] == {"rows":208,"disclosed_scores":5,"unknown_scores":203},
              "disclosed_range_matches_reference": min(disclosed, default=-1) == 91.12 and max(disclosed, default=-1) == 91.33,
              "candidate_is_not_final_award": all(s["stage"] == "candidate_publication" and not s["counted_as_won"] for s in stages),
              "no_training_pairs_created": all(r["body_pair_status"] == "not_available" and r["pure_ai_score"] is None for r in assessment["records"])}
    result = {"experiment_id":experiment_id, "evaluation_kind":"known_layout_regression_not_generic_pdf_table_acceptance",
              "assessment":assessment, "stages":stages, "checks":[{"check_id":k,"passed":v} for k,v in checks.items()],
              "limits":["只验证一个已知13列评标表", "没有完整正文与实际提交关联，不进入技术总分训练", "成绩导入须复核，不把专家分反推纯AI分"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"summary":assessment["summary"], "checks":checks, "table_errors":len(assessment["errors"])},ensure_ascii=False))
    return result
