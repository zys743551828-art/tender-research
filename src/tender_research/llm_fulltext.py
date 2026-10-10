"""全文候选评审：页证据绑定、语境约束和思考模式对照；不训练或校准。"""
import argparse
import json
import statistics
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from .llm_common import CASE_DIR, SYSTEM, canonical, finite_score, freeze_input, invoke, save, sha
from .model_service import create_provider
from .paths import Workspace
from .rule_evidence import load_documents

PROMPT_VERSION = "DS-CASE-002-v1"
RATES = {"deepseek-flash": (.006, .3, 1.2), "deepseek-v4-pro": (.044, 1.32, 3.96)}


class StrictObject(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Topic(StrictObject):
    topic_id: StrictInt = Field(ge=1, le=8)
    assessment: str
    evidence_ids: list[str] = Field(min_length=1)
    limitations: str


class Finding(StrictObject):
    id: str
    kind: str
    claim: str
    scope_reason: str
    evidence_ids: list[str] = Field(min_length=1)
    needed_facts: list[str]
    action: str


class Review(StrictObject):
    status: str
    target: str
    maximum: float
    estimated_value: float | None
    rubric: dict
    rule_evidence_ids: list[str] = Field(min_length=1)
    topic_reviews: list[Topic] = Field(min_length=8, max_length=8)
    findings: list[Finding] = Field(max_length=6)
    reason: str
    limitations: list[str] = Field(min_length=1)


def prepare(docs):
    evidence, documents = {}, []
    for d in docs:
        pages = []
        for i, text in enumerate(d.pages, 1):
            eid = f"{d.file_id}-P{i:03d}"
            evidence[eid] = {"file_id": d.file_id, "pdf_page": i, "source_sha256": d.sha256,
                             "quote": text, "coverage": "extracted_text_only_not_full_visual_page"}
            pages.append({"evidence_id": eid, "pdf_page": i, "text": text,
                          "image_present": None if d.image_pages is None else i in d.image_pages})
        documents.append({"file_id": d.file_id, "sha256": d.sha256, "total_pdf_pages": len(d.pages), "pages": pages})
    instruction = """独立评审这份完整技术标，输出探索性技术总分及最多6个重要诊断疑点。所有原件的全部物理页已提供，但页中图片、图表和复杂表格可能没有完整抽取。必须逐页保留范围；每份文件给出的总页数可直接核实篇幅。
先根据招标确认技术满分、完整等级端点、内部是否有分项权重、建议与强制、报价入围全部条件。八类主题按招标顺序用topic_id=1至8定性复核；有章节不等于针对性、可行性或语言精练达到优秀。综合作未校准候选总分，不发明官方分项、扣分、概率或置信区间，不按主题平均得分。可以返回null并说明不能估计的原因。
诊断判断必须检查同一部位、对象、工序、时间和计量口径：局部与全项目、总体与某管径、安装与固化、人数与人日、更高内部目标与合格线、不同地质条件都不能仅因数字不同判成矛盾。若对象对应关系未明确，kind=待工程确认，不直接断言文本矛盾。禁止把缺图纸、未见投标期审批记录、抽取不到图像当作官方缺项；50页不等于超过50页。若指违反要求，须同时选招标和正文证据。数字结论说明计算及单位；同一证据不得充当两处不同内容的证明。不得用常识发明招标范围或实际设计。
不自拟逐字引文，只选择提供的页evidence_id，程序绑定原页完整抽取文本；无需复制原文。返回JSON对象且只有这些字段：
{"status":"experimental_candidate","target":"technical_total","maximum":5.0,"estimated_value":数字或null,
"rubric":{"grade_intervals":"完整端点","official_subweights_available":false,"page_limit_nature":"性质及是否超建议页数","shortlist_rule":"全条件"},
"rule_evidence_ids":["FILE-001-P000"],
"topic_reviews":[{"topic_id":1,"assessment":"定性意见","evidence_ids":["FILE-002-P000"],"limitations":"缺口"}],
"findings":[{"id":"F-001","kind":"文本矛盾或响应缺口或待工程确认","claim":"问题","scope_reason":"对象和口径为何可比或仍不确定","evidence_ids":["FILE-002-P000"],"needed_facts":["所需事实"],"action":"核实或修改方向"}],
"reason":"等级及候选总分的根据","limitations":["适用边界"]}。
样例P000只是占位符，实际只能选输入中的ID；不要输出样例ID。至少引用一页招标规则；每个主题及每条发现须引用正文。文本矛盾若比较两处内容须提供至少两个不同证据ID；同页不同语句可以用一个ID但须在scope_reason说明。最多六条，允许零条。不要生成施工参数或可提交修订稿。"""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": instruction + "\n资料：\n" + canonical(documents)}]
    if "4.26" in canonical(messages) or "4.39" in canonical(messages):
        raise ValueError("事后成绩或门槛出现在输入；须审计")
    return messages, evidence


def validate_bound(output, evidence):
    errors, selections = [], []
    try:
        review = Review.model_validate(output)
    except ValidationError as exc:
        return {"automatic_gate_passed": False, "errors": ["schema:" + ".".join(map(str, e["loc"])) for e in exc.errors()],
                "bound_sections": [], "semantic_review": "pending"}
    if review.status != "experimental_candidate" or review.target != "technical_total" or review.maximum != 5:
        errors.append("target_or_scale")
    if review.estimated_value is not None and not finite_score(review.estimated_value, 5):
        errors.append("score_domain")
    if review.rubric.get("official_subweights_available") is not False:
        errors.append("unsupported_official_subweights")
    if {t.topic_id for t in review.topic_reviews} != set(range(1, 9)):
        errors.append("topic_id_coverage")
    if len({f.id for f in review.findings}) != len(review.findings):
        errors.append("duplicate_finding_id")
    sections = [("rubric", review.rule_evidence_ids)] + [(f"topic:{t.topic_id}", t.evidence_ids) for t in review.topic_reviews] + [(f"finding:{f.id}", f.evidence_ids) for f in review.findings]
    for name, ids in sections:
        valid = bool(ids) and len(ids) == len(set(ids)) and all(eid in evidence for eid in ids)
        if not valid:
            errors.append(name + ":invalid_or_duplicate_id")
        rows = [evidence[eid] for eid in ids if eid in evidence]
        if name == "rubric" and not any(e["file_id"] == "FILE-001" for e in rows):
            errors.append("rubric_missing_tender")
        if name != "rubric" and not any(e["file_id"] == "FILE-002" for e in rows):
            errors.append(name + ":missing_body")
        selections.append({"section": name, "evidence_ids": ids, "bound_evidence": rows})
    for finding in review.findings:
        if finding.kind not in ("文本矛盾", "响应缺口", "待工程确认"):
            errors.append(finding.id + ":kind")
        if finding.kind == "响应缺口" and not any(evidence.get(eid, {}).get("file_id") == "FILE-001" for eid in finding.evidence_ids):
            errors.append(finding.id + ":missing_requirement")
    return {"automatic_gate_passed": not errors, "errors": errors, "bound_sections": selections,
            "semantic_review": "pending_not_implied_by_page_binding"}


def visual_facts(results, docs):
    """只读已完成的原页存在性复核，不读取其评分、发现分类或原模型长观察。"""
    probe = json.loads((results / "llm-visual-probe.json").read_text())
    review = json.loads((results / "review-fulltext.json").read_text())
    if review["visual_probes"]["sha256"] != sha(results / "llm-visual-probe.json"):
        raise ValueError("图表复核与冻结探针不匹配")
    versions = {d.file_id: d.sha256 for d in docs}
    observations = {}
    for check in review["visual_probes"]["checks"]:
        source = next(i for i in probe["images"] if i["evidence_id"] == check["evidence_id"])
        if source["file_id"] != "FILE-002":
            continue
        if not check["core_probe_passed"] or source["source_sha256"] != versions[source["file_id"]]:
            raise ValueError("图表事实未复核或来源版本不一致")
        observations[source["evidence_id"]] = {
            **{k: source[k] for k in ("evidence_id", "file_id", "pdf_page", "source_sha256", "rendered_png_sha256", "width", "height")},
            "evidence_kind": "developer_verified_visual_presence",
            "facts": {k: check["reference"][k] for k in ("has_visible_content", "has_network_diagram", "has_gantt_chart", "has_site_layout", "planning_horizon_days")},
            "coverage": "presence and visible horizon only; network logic and location feasibility unverified"}
    return observations


def experiment_name(budget_probe, with_visual_facts):
    if budget_probe and with_visual_facts:
        raise ValueError("图表事实与预算探针分别登记")
    return "llm-fulltext-budget-probe" if budget_probe else "llm-fulltext-visual-facts" if with_visual_facts else "llm-fulltext-id-bound"


def run(ws: Workspace, budget_probe=False, with_visual_facts=False):
    experiment = experiment_name(budget_probe, with_visual_facts)
    folder = ws.results / experiment
    folder.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    files = [here / name for name in ("llm_fulltext.py", "llm_common.py", "model_service.py", "deepseek_client.py", "rule_evidence.py")]
    jobs = [(f"BOUND-FLASH-{i}", "deepseek-flash", False) for i in (1, 2, 3)] + [("BOUND-PRO-1", "deepseek-v4-pro", False), ("THINK-FLASH-1", "deepseek-flash", True), ("THINK-PRO-1", "deepseek-v4-pro", True)]
    if budget_probe:
        jobs = [("THINK-FLASH-BUDGET-1", "deepseek-flash", True)]
    if with_visual_facts:
        jobs = [("VISUAL-FACTS-PRO-1", "deepseek-v4-pro", True)]
    token_limit = 32768 if budget_probe else 16000
    protocol = {"experiment_id": experiment, "prompt_version": PROMPT_VERSION,
                "code_sha256": {p.name: sha(p) for p in files}, "max_completion_calls": len(jobs),
                "jobs": jobs, "output_token_limit": token_limit, "thinking_effort": "high",
                "configuration": "same messages as llm-fulltext-id-bound; single Flash thinking call with larger output budget; no automatic retry" if budget_probe else "page evidence IDs + source-bound citations + scope checks; thinking additionally compared on identical messages",
                "known_to_analyst": True, "blind_evaluation": False, "label_used_to_tune_prompt": False,
                "excluded_inputs": ["reported score", "observed cutoff", "case analysis", "earlier model outputs"],
                "source": "all archived tender 219 pages and submitted body 50 pages as extracted text; no images sent",
                "external_transfer_authorization": {"source": "direct_user_message", "date": "2026-10-07", "destination": "https://api.deepseek.com/chat/completions", "scope": "CASE-001 full tender and technical body"},
                "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/", "pricing_verified_date": "2026-10-07", "pricing_per_million_peak": RATES,
                "thinking_source": "https://api-docs.deepseek.com/guides/thinking_mode/"}
    if with_visual_facts:
        protocol["configuration"] = "full llm-fulltext-id-bound messages plus verified original-page chart presence facts; Pro high thinking 16000 tokens; no label or earlier diagnosis supplied"
        protocol["visual_fact_sources"] = {name: sha(ws.results / name) for name in ("llm-visual-probe.json", "review-fulltext.json")}
    freeze_input(folder / "protocol.json", protocol)
    bundle = folder / "source.zip"
    if not bundle.exists():
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            for path in files:
                z.write(path, path.name)
    docs = load_documents(ws.datasets / CASE_DIR / "资料清单.json")
    messages, evidence = prepare(docs)
    if with_visual_facts:
        observations = visual_facts(ws.results, docs)
        evidence.update(observations)
        messages.append({"role": "user", "content": "补充同版原件的图表存在性复核。仅验证图表存在及可见计划120日历天，未验施工网络逻辑、全部节点或平面坐标。复核事实不是评分标签，不把存在误报缺失；仍以完整正文与规则评审。可选择下列新证据ID，原文抽取仍在前述资料。\n" + canonical(observations)})
    if "4.26" in canonical(messages) or "4.39" in canonical(messages):
        raise ValueError("补充事实后出现事后成绩或门槛；须审计")
    client = create_provider(ws.secrets)
    freeze_input(folder / "health.json", client.health()) if not (folder / "health.json").exists() else None
    health = json.loads((folder / "health.json").read_text())
    if health["status"] != "available":
        raise RuntimeError("模型服务不可用")
    rows, calls = [], []
    for call_id, model, thinking in jobs:
        record = invoke(client, folder, call_id, messages, model, max_tokens=token_limit, prompt_version=PROMPT_VERSION, thinking=thinking)
        calls.append(record)
        row = {"call_id": call_id, "model": model, "thinking": thinking, "status": record["status"]}
        if record["status"] == "received":
            output = record["response"]["output"]
            row.update(candidate_score=output.get("estimated_value"), validation=validate_bound(output, evidence))
        rows.append(row)
        if record.get("http_status") in (401, 402, 403, 429):
            break
    # 报送结果只在全部候选生成后读取，用于对照。
    actual = json.loads((ws.datasets / CASE_DIR / "资料清单.json").read_text())["assessment"]["reported_results"][0]["value"]
    for row in rows:
        if finite_score(row.get("candidate_score"), 5):
            row["difference_to_user_report"] = round(row["candidate_score"] - actual, 4)
    values = [r["candidate_score"] for r in rows if r["model"] == "deepseek-flash" and not r["thinking"] and finite_score(r.get("candidate_score"), 5)]
    usage = {k: sum(c.get("response", {}).get("usage", {}).get(k, 0) for c in calls) for k in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")}
    cost = sum(sum(c.get("response", {}).get("usage", {}).get(k, 0) * rate for k, rate in zip(("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens"), RATES[c.get("response", {}).get("requested_model", "deepseek-flash")])) for c in calls) / 1e6
    summary = {"experiment_id": experiment, "finished_at": datetime.now(UTC).isoformat(), "actual_completion_calls": len(calls), "successful_completions": sum(c["status"] == "received" for c in calls),
               "originals": [d.summary() for d in docs], "case_calls": rows,
               "flash_repeat": {"n": len(values), "values": values, "range": round(max(values)-min(values), 4) if values else None, "population_std": statistics.pstdev(values) if values else None},
               "usage_successful_calls_only": usage, "estimated_cost_usd_offpeak_to_peak_successful_calls_only": [round(cost/2, 6), round(cost, 6)],
               "billing_complete": all(c["status"] == "received" for c in calls), "payment_receipt_available": False,
               "official_verified_pairs": 0, "trained_on_real_cases": False, "reference_kind": "user_report_pending_source_single_known_case",
               "artifact_sha256": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
               "limits": ["page binding proves origin, not semantic support", "single-case exploration is not generalization", "thinking mode has one repeat per model", "extracted text omits image/diagram content", "output budget or timeout failure may still incur charges", "model aliases do not fix service weights"]}
    save(ws.results / (experiment + ".json"), summary)
    print(json.dumps({"event": "finished", "calls": len(calls), "success": summary["successful_completions"], "scores": [{k: r[k] for k in ("call_id", "candidate_score") if k in r} for r in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-probe", action="store_true", help="单次 Flash 思考模式，输出预算 32768")
    parser.add_argument("--visual-facts", action="store_true", help="全文 Pro 思考模式加入已复核图表存在性事实")
    args = parser.parse_args()
    run(Workspace(), args.budget_probe, args.visual_facts)
