"""冻结合成条款与受控变体，记录实际产品逐项判断，不作正式评分精度结论。"""

import argparse
import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from bidwb.ai import predict, prompts
from bidwb.ai.grounding import EvidenceText
from bidwb.ai.llm import OpenAICompatClient, Usage, call_validated
from bidwb.config import Settings
from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    rule = "本合成项目总工期不得超过180日历天。施工进度计划满分8分，根据阶段安排、关键线路及资源保障的具体性与可行性综合评分。"
    baseline = [
        "第一章 施工进度：本合成道路工程总工期180日历天；场地准备20天、地下管线60天、道路基层60天、面层及验收40天，依次衔接。",
        "关键线路为场地准备→地下管线→道路基层→面层验收；管线配置2个作业班组和2台挖掘机，基层配置1台摊铺机，周度核对完成量，延误时增加同类作业班组。进度表总工期180日历天。",
    ]
    cases = {
        "baseline": baseline,
        "numeric_conflict": [baseline[0], baseline[1].replace("进度表总工期180", "进度表总工期240")],
        "generic_expansion": ["第一章 施工进度：加强组织领导，科学组织施工，强化责任落实，做好施工协调，确保工程顺利完成。" * 30],
        "unrelated_caption": [*baseline, "插图说明：一张海滨度假风景装饰照片，与施工进度无关。本次未提供图像像素，不能判断图中工程内容。"],
    }
    protocol = {
        "synthetic_only": True,
        "input_method": predict.PREDICTION_METHOD,
        "prompt_version": prompts.PROMPT_VERSION,
        "rule": rule,
        "max_points": "8",
        "cases": cases,
        "temperature": 0,
        "checks": [
            "numeric_conflict requires two distinct offered citations",
            "generic and unrelated content must not be treated as actionable progress measures",
        ],
        "runs_per_case": 1,
        "real_score_accuracy_evaluated": False,
        "actual_image_semantics_evaluated": False,
    }
    encoded = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    (args.output / "protocol.json").write_bytes(encoded + b"\n")
    digest = hashlib.sha256(encoded).hexdigest()
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
    results = {}
    for name, texts in cases.items():
        body = [EvidenceText(f"{name}_body_{i}", quote) for i, quote in enumerate(texts)]
        output = call_validated(
            client,
            prompts.item_judgement(Decimal(8)),
            system=prompts.JUDGEMENT_SYSTEM,
            user=prompts.judgement_user(rule, [EvidenceText("synthetic_rule", rule)], "8", body, sections=[], truncated=False),
            usage=usage,
            temperature=0,
        )
        grounded = predict.ground(output, {item.evidence_id for item in body}, Decimal(8))
        results[name] = {"raw": output.model_dump(), "grounded": asdict(grounded)}
        (args.output / f"{name}.json").write_text(json.dumps(results[name], ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"case": name, "score": str(grounded.score), "effectiveness": grounded.effectiveness["status"], "consistency": grounded.consistency["status"]}
            ),
            flush=True,
        )
    receipt = {
        "protocol_sha256": digest,
        "requested_model": client.model,
        "usage": asdict(usage),
        "scores": {name: str(value["grounded"]["score"]) for name, value in results.items()},
        "numeric_conflict_detected": results["numeric_conflict"]["grounded"]["consistency"]["status"] == "contradicted",
        "generic_not_higher_than_baseline": results["generic_expansion"]["grounded"]["score"] <= results["baseline"]["grounded"]["score"],
        "caption_not_higher_than_baseline": results["unrelated_caption"]["grounded"]["score"] <= results["baseline"]["grounded"]["score"],
        "real_score_accuracy_evaluated": False,
        "actual_image_semantics_evaluated": False,
        "production_database_modified": False,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
