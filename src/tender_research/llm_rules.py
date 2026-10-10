"""公开规则评测：单题隔离，模型选证据 ID，程序回填原文；只发送官方公开资料。"""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .llm_common import SYSTEM, freeze_input, invoke, save, sha
from .model_service import create_provider
from .paths import Workspace
from .rule_evidence import check_evidence, load_documents

PROMPT_VERSION = "DS-RULE-003-v1"
EXPERIMENTS = {"deepseek-flash": "llm-public-rules-flash", "deepseek-v4-pro": "llm-public-rules-pro"}


def bind_evidence(output, task_id, hits):
    expected = {"task_id", "status", "answer", "evidence_ids", "limitations"}
    ids = output.get("evidence_ids")
    valid = (set(output) == expected and output.get("task_id") == task_id
             and output.get("status") in ("supported", "insufficient", "conflict")
             and isinstance(output.get("answer"), str) and isinstance(output.get("limitations"), str)
             and isinstance(ids, list) and bool(ids)
             and all(isinstance(i, str) and i in hits for i in ids) and len(ids) == len(set(ids)))
    return {"contract_passed": valid, "bound_evidence": [hits[i] for i in ids] if valid else [],
            "semantic_review": "pending_not_implied_by_valid_evidence_ids"}


def run(ws: Workspace, model="deepseek-flash"):
    experiment = EXPERIMENTS[model]
    folder = ws.results / experiment
    folder.mkdir(parents=True, exist_ok=True)
    ref_path = ws.datasets / "评测集/RULE-001.json"
    retrieval_path = ws.results / "retrieval-rules.json"
    reference = json.loads(ref_path.read_text())
    retrieval = json.loads(retrieval_path.read_text())
    if sha(ref_path) != retrieval["reference_sha256"]:
        raise ValueError("参考版本不一致")
    candidate = next(c for c in retrieval["candidates"] if c["candidate_id"] == "pdfplumber-render-guard-900-180")
    manifest = ws.datasets / "公开规则资料/资料清单.json"
    documents = load_documents(manifest)
    source_records = {r["file_id"]: r for r in json.loads(manifest.read_text())["file_records"]}
    by_doc = {d.file_id: d for d in documents}
    here = Path(__file__).parent
    protocol = {"experiment_id": experiment, "prompt_version": PROMPT_VERSION, "context": "dossier",
                "code_sha256": {name: sha(here / name) for name in ("llm_rules.py", "llm_common.py", "deepseek_client.py", "model_service.py")},
                "reference_sha256": sha(ref_path), "retrieval_sha256": sha(retrieval_path),
                "max_completion_calls": 32, "max_output_tokens_per_call": 4000,
                "design": ["one question per request", "explicit project/file scope",
                           "evidence IDs selected by model and exact source excerpts filled by program",
                           "source manifest and first 3 physical pages establish document identity",
                           "each top-3 hit expanded to whole page plus next page, deduplicated"],
                "known_reference_not_blind": True, "reference_answers_excluded_from_input": True,
                "case_body_sent": False, "model": model, "thinking": False, "temperature": 0}
    freeze_input(folder / "protocol.json", protocol)
    client = create_provider(ws.secrets, "deepseek")
    tasks, records = [], []
    for task in reference["tasks"]:
        hits = next(t for t in candidate["tasks"] if t["task_id"] == task["task_id"])["hits"][:3]
        expanded, seen = [], set()
        for hit in hits:
            doc = by_doc[hit["file_id"]]
            for page in (hit["pdf_page"], hit["pdf_page"] + 1):
                if page > len(doc.pages) or (doc.file_id, page) in seen:
                    continue
                seen.add((doc.file_id, page))
                text = doc.pages[page - 1]
                if not text.strip():
                    continue
                expanded.append({**hit, "pdf_page": page, "start_char": 0, "end_char": len(text), "quote": text})
        hits = expanded
        if not all(check_evidence(h, documents)[0] for h in hits):
            raise ValueError("检索证据原件校验失败")
        numbered = {f"E{i + 1}": h for i, h in enumerate(hits)}
        inputs = {"task_id": task["task_id"], "query": task["query"],
                  "project_id": task.get("project_id"), "allowed_file_ids": task.get("file_ids"),
                  "evidence": [{"evidence_id": key, "file_id": h["file_id"], "pdf_page": h["pdf_page"], "quote": h["quote"]} for key, h in numbered.items()]}
        inputs["document_identity"] = [
            {"file_id": fid, "manifest_record": source_records[fid],
             "detected_project_ids": list(by_doc[fid].project_ids),
             "frontmatter": [{"pdf_page": i + 1, "text": text} for i, text in enumerate(by_doc[fid].pages[:3])],
             "identity_note": "文件身份需结合官方归档来源、封面和项目编号，示范文本页眉表示采用版式，不据此否定项目正文身份"}
            for fid in dict.fromkeys(h["file_id"] for h in hits)]
        prompt = """只回答这个问题，只依据本题允许的项目和文件。未提供的补疑及条件不得猜测；明确是原文件还是补疑。区分项目、例外、区间端点、建议与强制、技术入围与最终中标。回答简洁，只展开题目相关条件。证据不足标为 insufficient，条款本身矛盾标为 conflict，不能用另一项目解决冲突。
仅返回 JSON：{"task_id":"...","status":"supported或insufficient或conflict","answer":"答案与必要条件","evidence_ids":["E1"],"limitations":"所给证据的局限"}。
选择能够支持结论的原文证据编号，原文由程序回填；不输出自拟引用，不创造不存在的证据编号。不能按未提供的分项权重赋分。\n"""
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt + json.dumps(inputs, ensure_ascii=False)}]
        record = invoke(client, folder, task["task_id"], messages, model, max_tokens=4000, prompt_version=PROMPT_VERSION)
        records.append(record)
        row = {"task_id": task["task_id"], "status": record["status"]}
        if record["status"] == "received":
            row.update(bind_evidence(record["response"]["output"], task["task_id"], numbered))
        tasks.append(row)
        if record.get("http_status") in (401, 402, 403, 429):
            break
    usage = {k: sum(r.get("response", {}).get("usage", {}).get(k, 0) for r in records)
             for k in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")}
    rates = (.006, .3, 1.2) if model == "deepseek-flash" else (.044, 1.32, 3.96)
    cost_peak = sum(usage[k] * rate for k, rate in zip(("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens"), rates)) / 1e6
    summary = {"experiment_id": experiment, "finished_at": datetime.now(UTC).isoformat(),
               "actual_calls": len(records), "successful_calls": sum(r["status"] == "received" for r in records),
               "tasks": tasks, "usage": usage, "estimated_cost_usd_offpeak_to_peak": [round(cost_peak / 2, 6), round(cost_peak, 6)],
               "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/", "pricing_verified_date": "2026-10-07",
               "cost_is_estimate_not_payment_receipt": True,
               "limits": ["证据原文由程序绑定，不能据此声称模型自由引用准确率100%", "证据编号存在不证明语义支持", "开发者已知参考，非独立工程盲测"],
               "artifact_sha256": {p.name: sha(p) for p in sorted(folder.glob("*.json"))}}
    save(ws.results / (experiment + ".json"), summary)
    print(json.dumps({"event": "finished", "calls": len(records), "success": summary["successful_calls"], "usage": usage}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=tuple(EXPERIMENTS), default="deepseek-flash")
    run(Workspace(), p.parse_args().model)
