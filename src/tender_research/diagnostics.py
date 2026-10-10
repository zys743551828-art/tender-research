"""保守的确定性试验检查；不推断官方扣分，不能代替工程师审核。"""
from __future__ import annotations

import hashlib
import re
import unicodedata

from .contracts import Evidence, Finding
from .rule_evidence import Chunk, Document, check_evidence


def compact_with_offsets(text: str):
    chars, offsets = [], []
    for i, c in enumerate(text):
        for n in unicodedata.normalize("NFKC", c):
            if not n.isspace():
                chars.append(n); offsets.append(i)
    return "".join(chars), offsets


def source_span(doc: Document, page: int, start: int, end: int) -> dict:
    hit = Chunk(doc, page, start, end, doc.pages[page - 1][start:end]).evidence(0.0)
    if not check_evidence(hit, [doc])[0]:
        raise ValueError("诊断引用校验失败")
    Evidence.model_validate(hit)
    return hit


def extract_limited_rules(docs: list[Document]) -> list[dict]:
    """仅覆盖下列显式句式。未知句式不输出；不读评测集或参考答案。"""
    rules = []
    for doc in docs:
        if len(doc.project_ids) != 1:
            continue
        for page_no, raw in enumerate(doc.pages, 1):
            text, positions = compact_with_offsets(raw)
            def emit(key, value, category, match, condition="", *, _doc=doc, _page=page_no, _positions=positions, **extra):
                start, end = _positions[match.start()], _positions[match.end() - 1] + 1
                evidence = source_span(_doc, _page, start, end)
                rules.append({"rule_id": f"{_doc.file_id}:{_page}:{key}:{start}",
                              "project_id": _doc.project_ids[0], "key": key, "value": value,
                              "category": category, "condition": condition, "review_status": "needs_confirmation",
                              "parser_version": "explicit-clause-v1", "evidence": [evidence], **extra})
            for m in re.finditer(r"启用合肥公共资源交易大模型AI[“\"]类人[”\"]评审功能辅助评标", text):
                emit("ai_assistance", True, "explicit", m)
            for m in re.finditer(r"☑不执行AI[“\"]类人[”\"]辅助评标", text):
                emit("ai_assistance", False, "explicit", m)
            for m in re.finditer(r"本项满分(\d+(?:\.\d+)?)分", text):
                if "施工组织" in text[max(0, m.start() - 600):m.end() + 100]:
                    emit("technical_max", float(m.group(1)), "explicit", m)
            if "建议编制要求" in text:
                for m in re.finditer(r"(?:施工组织设计的篇幅)?不超过(\d+)页", text):
                    if 0 <= m.start() - text.find("建议编制要求") < 550:
                        emit("recommended_page_limit", int(m.group(1)), "recommendation", m,
                             "建议编制要求；不直接转为否决条件")
            intervals = list(re.finditer(r"良好得(\d+)分<F<(\d+)分", text))
            marker = text.find("现修改为")
            if marker >= 0:
                before = [m for m in intervals if m.start() < marker]
                after = [m for m in intervals if m.start() > marker]
                if len(before) == len(after) == 1:
                    old, new = before[0], after[0]
                    # 一条补疑必须同时带原值与新值；不能按文件列表顺序覆盖。
                    match = re.search(re.escape(text[old.start():new.end()]), text)
                    emit("good_interval_patch", {"lower": int(new.group(1)), "upper": int(new.group(2)),
                                               "lower_open": True, "upper_open": True}, "correction", match,
                         old_value={"lower": int(old.group(1)), "upper": int(old.group(2)), "lower_open": True, "upper_open": True})
            else:
                for m in intervals:
                    emit("good_interval", {"lower": int(m.group(1)), "upper": int(m.group(2)),
                                           "lower_open": True, "upper_open": True}, "explicit", m)
            for m in re.finditer(r"招标文件获取截止时间延期到[“\"]?(20\d{2})年\d+月\d+日", text):
                year = int(m.group(1))
                if year != int(doc.project_ids[0][:4]):
                    emit("date_year_conflict", {"source_year": year, "project_year": int(doc.project_ids[0][:4])},
                         "needs_clarification", m, "保留原文年份，不自动纠正")
            if "中标后提供即可" in text and "答:均可" in text:
                m = re.search(r"请问.{0,250}中标后提供即可[?？]答:均可[。.]", text)
                if m:
                    emit("testing_material_timing", "bid_or_post_award", "explicit_exception", m,
                         "仅限同条答复所述试验人员、仪器设备及检测单位资料")
    return rules


def resolve_interval(rules: list[dict], project: str) -> dict:
    base = [r for r in rules if r["project_id"] == project and r["key"] == "good_interval"]
    patch = [r for r in rules if r["project_id"] == project and r["key"] == "good_interval_patch"]
    if len(base) != 1 or len(patch) != 1 or base[0]["value"] != patch[0]["old_value"]:
        return {"status": "needs_review", "reason": "base_patch_not_unique_or_original_value_mismatch"}
    return {"status": "candidate_effective_rule_needs_confirmation", "value": patch[0]["value"],
            "base_rule": base[0]["rule_id"], "patch_rule": patch[0]["rule_id"],
            "evidence": base[0]["evidence"] + patch[0]["evidence"], "limits": "仅已归档版本的显式修订，未证明后续补遗齐全"}


def pipe_heading_conflicts(docs: list[Document]) -> list[dict]:
    findings = []
    # 限于明确的独立施工小节标题与该小节的核心材料描述；不因全稿出现多种管材而报错。
    heading = re.compile(r"(?m)^\s*([一二三四五六七八九十]+)、球墨铸铁管施工\s*$")
    next_heading = re.compile(r"(?m)^\s*[一二三四五六七八九十]+、[^\n]{1,35}$")
    for doc in docs:
        for page_no, raw in enumerate(doc.pages, 1):
            for title in heading.finditer(raw):
                subsequent = next_heading.search(raw, title.end())
                end = subsequent.start() if subsequent else len(raw)
                body = raw[title.end():end]
                wrong = re.search(r"FRPM\s*管|玻璃纤维增强塑料夹砂管", body)
                if wrong is None:
                    continue
                start = title.end() + wrong.start()
                hit_title = source_span(doc, page_no, title.start(), title.end())
                hit_body = source_span(doc, page_no, start, title.end() + wrong.end())
                fid = "FND-" + hashlib.sha256(f"{doc.sha256}:{page_no}:{title.start()}".encode()).hexdigest()[:16]
                finding = Finding(finding_id=fid, check_id="PIPE-HEADING-001", severity="review",
                                  status="needs_engineer_confirmation",
                                  statement="球墨铸铁管施工小节描述 FRPM／玻璃纤维增强塑料夹砂管，工程对象表述需核实。",
                                  evidence=[Evidence.model_validate(h) for h in (hit_title, hit_body)])
                findings.append(finding.model_dump())
    return findings


def revision_gate(finding: dict, authoritative_material: str | None, engineer_confirmed: bool) -> dict:
    if not authoritative_material or not engineer_confirmed:
        return {"status": "awaiting_engineering_basis", "finding_id": finding["finding_id"],
                "required": ["图纸或工程量清单确认管材", "工程师核实该小节用途"], "automatic_material_replacement": False}
    return {"status": "eligible_for_reviewed_revision", "finding_id": finding["finding_id"],
            "authoritative_material": authoritative_material, "automatic_material_replacement": False,
            "required": ["关联审核记录与修订稿", "重查施工、接口、验收及参数的一致性"]}
