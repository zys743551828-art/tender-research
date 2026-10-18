"""按预先冻结的五对相同证据 ID 输入复测图注敏感性，保存修复响应与失败。"""

import argparse
import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from statistics import mean

from bidwb.ai import predict, prompts
from bidwb.ai.grounding import EvidenceText
from bidwb.ai.llm import MAX_HTTP_ATTEMPTS, MAX_REPAIRS, IncompleteGeneration, OpenAICompatClient, Usage, call_validated
from bidwb.config import Settings
from bidwb.errors import Unavailable
from dotenv import dotenv_values


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


class RecordedClient:
    def __init__(self, inner, destination):
        self.inner, self.destination = inner, destination
        self.model = inner.model
        self.responses = []

    def complete(self, **kwargs):
        record = {"request": kwargs}
        self.responses.append(record)
        try:
            result = self.inner.complete(**kwargs)
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            save(self.destination, self.responses)
            raise
        record.update({"response": result.text, "usage": asdict(result.usage)})
        save(self.destination, self.responses)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    cases = {name: source["cases"][name] for name in ("baseline", "unrelated_caption")}
    rule = source["rule"]
    bodies = {name: [EvidenceText(f"shared_body_{i}", quote) for i, quote in enumerate(texts)] for name, texts in cases.items()}
    users = {
        name: prompts.judgement_user(rule, [EvidenceText("synthetic_rule", rule)], "8", body, sections=[], truncated=False) for name, body in bodies.items()
    }
    schedule = [{"pair": pair, "case": name} for pair in range(1, 6) for name in (list(cases) if pair % 2 else list(reversed(cases)))]
    protocol = {
        "synthetic_only": True,
        "source_file_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "cases": cases,
        "rule": rule,
        "max_points": "8",
        "input_method": predict.PREDICTION_METHOD,
        "prompt_version": prompts.PROMPT_VERSION,
        "system": prompts.JUDGEMENT_SYSTEM,
        "users": users,
        "shared_evidence_ids": True,
        "schedule": schedule,
        "temperature": 0,
        "runs_per_case": 5,
        "max_format_repairs_per_run": MAX_REPAIRS,
        "max_transport_attempts_per_completion": MAX_HTTP_ATTEMPTS,
        "failure_policy": "Record each failed run; continue fixed schedule; no replacement or outcome-based stopping.",
        "comparison": "Per-arm scores, range, mean; paired caption-minus-baseline deltas for complete pairs; effectiveness/consistency and raw notes retained.",
        "repeat_penalty_criterion": "Only label repeated penalty in this fixture if all five pairs complete and every delta < 0; otherwise record no uniform repeat penalty, not invariance.",
        "limitations": [
            "No independent official score pairs, actual image pixels or real quality accuracy.",
            "Shared IDs remove the previous case-prefixed-ID confound; this is not an exact replication of those requests.",
            "One synthetic rule; provider alias, backend randomness and time are not controlled; no causal attribution or statistical significance claim.",
            "Recorded completions include schema repairs; adapter internal HTTP retries and unsuccessful-request tokens are not exposed.",
            "Do not change production prompts from this fixture alone.",
        ],
    }
    encoded = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    (args.output / "protocol.json").write_bytes(encoded + b"\n")
    root = Path(__file__).resolve().parents[2]
    config = dotenv_values(root / "tender-app/deploy/.env")
    settings = Settings(
        environment="test",
        llm_base_url=config.get("LLM_BASE_URL") or "https://api.deepseek.com",
        llm_model=config.get("LLM_MODEL") or "deepseek-chat",
        llm_timeout_seconds=60,
        llm_api_key=(root / "tender-app/deploy/secrets/llm_api_key").read_text(encoding="utf-8").strip(),
    )
    client, usage = OpenAICompatClient(settings), Usage()
    results = []
    try:
        for entry in schedule:
            name, pair = entry["case"], entry["pair"]
            recorded = RecordedClient(client, args.output / f"pair-{pair}-{name}-responses.json")
            result = dict(entry)
            try:
                output = call_validated(recorded, prompts.item_judgement(Decimal(8)), system=protocol["system"], user=users[name], usage=usage, temperature=0)
                grounded = predict.ground(output, {item.evidence_id for item in bodies[name]}, Decimal(8))
                result.update({"status": "completed", "raw": output.model_dump(), "grounded": asdict(grounded)})
            except (IncompleteGeneration, Unavailable, ValueError, KeyError, TypeError) as exc:
                result.update({"status": "failed", "error_type": type(exc).__name__})
            save(args.output / f"pair-{pair}-{name}.json", result)
            results.append(result)
            print(json.dumps({**entry, "status": result["status"], "score": str(result.get("grounded", {}).get("score"))}), flush=True)
    finally:
        client.close()
    scores = {name: [r["grounded"]["score"] for r in results if r["case"] == name and r["status"] == "completed"] for name in cases}
    deltas = []
    for pair in range(1, 6):
        paired = {r["case"]: r["grounded"]["score"] for r in results if r["pair"] == pair and r["status"] == "completed"}
        deltas.append({"pair": pair, "caption_minus_baseline": paired["unrelated_caption"] - paired["baseline"] if len(paired) == 2 else None})
    save(
        args.output / "receipt.json",
        {
            "protocol_sha256_without_trailing_newline": hashlib.sha256(encoded).hexdigest(),
            "requested_model": client.model,
            "usage": asdict(usage),
            "completed_runs": sum(r["status"] == "completed" for r in results),
            "arms": {
                name: {"scores": values, "range": [min(values), max(values)] if values else None, "mean": mean(values) if values else None}
                for name, values in scores.items()
            },
            "paired_deltas": deltas,
            "repeat_penalty_criterion_met": all(d["caption_minus_baseline"] is not None and d["caption_minus_baseline"] < 0 for d in deltas),
            "real_score_accuracy_evaluated": False,
            "actual_image_semantics_evaluated": False,
            "production_database_modified": False,
            "production_prompt_modified": False,
        },
    )


if __name__ == "__main__":
    main()
