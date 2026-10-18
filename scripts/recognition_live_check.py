"""用真实表格 PDF 验证已部署候选核验，合成项目关闭自动评审，不调用业务模型。"""

import argparse
import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import httpx
from reportlab.pdfgen import canvas


def sample():
    output = io.BytesIO()
    document = canvas.Canvas(output)
    document.drawString(50, 770, "Synthetic recognition review - values are not a real bid")
    for x in (50, 250, 430):
        document.line(x, 600, x, 720)
    for y in (600, 640, 680, 720):
        document.line(50, y, 430, y)
    for y, first, second in ((700, "Item", "Amount"), (660, "Cost", "15.12"), (620, "Duration", "180")):
        document.drawString(60, y, first)
        document.drawString(260, y, second)
    document.save()
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("结果目录必须为新的路径")
    args.output.mkdir(parents=True)
    data = sample()
    (args.output / "synthetic-table.pdf").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    root = Path(__file__).resolve().parents[2]
    with httpx.Client(base_url="http://localhost:8080/api/v1/", timeout=60) as client:
        login = client.post(
            "auth/login",
            json={"username": "admin", "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip()},
        )
        login.raise_for_status()
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]

        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()

        def post(path, body, key=None):
            response = client.post(path, json=body, headers={"Idempotency-Key": key or uuid.uuid4().hex})
            response.raise_for_status()
            return response.json()

        pid = post("projects", {"title": "合成验证：识别候选核验与追加订正", "settings": {"auto_start": False}})["project_id"]
        upload = post(
            f"projects/{pid}/uploads",
            {"original_name": "synthetic-table.pdf", "role": "body", "media_type": "application/pdf", "sha256": digest, "byte_size": len(data)},
        )
        parts = []
        for part in upload["parts"]:
            number = part["part_number"]
            response = client.put(
                f"uploads/{upload['upload_id']}/parts/{number}",
                content=data[(number - 1) * upload["part_bytes"] : number * upload["part_bytes"]],
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            parts.append({"part_number": number, "etag": response.headers["etag"]})
        version = post(f"uploads/{upload['upload_id']}/complete", {"sha256": digest, "parts": parts})["version_id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            before = get(f"files/{version}/recognition")
            if before["parse_id"] and before["can_review"]:
                break
            time.sleep(0.5)
        else:
            raise AssertionError("解析未在限时内产生候选")
        item = next(item for item in before["items"] if item["kind"] == "table" and "15.12" in item["quote"])
        request = {
            "expected_parse_id": before["parse_id"],
            "evidence_id": item["evidence_id"],
            "decision": "accept",
            "reason": "合成表格逐格对照，金额为15.12",
            "reviewed_original": True,
            "reviewed_numbers": False,
        }
        response = client.post(f"files/{version}/recognition:review", json=request, headers={"Idempotency-Key": uuid.uuid4().hex})
        assert response.status_code == 422 and response.json()["code"] == "recognition_numbers_not_reviewed"
        request["reviewed_numbers"] = True
        key = uuid.uuid4().hex
        accepted = post(f"files/{version}/recognition:review", request, key)
        assert post(f"files/{version}/recognition:review", request, key) == accepted
        response = client.post(f"files/{version}/recognition:review", json=request, headers={"Idempotency-Key": uuid.uuid4().hex})
        assert response.status_code == 409 and response.json()["code"] == "recognition_parse_stale"
        current = get(f"files/{version}/recognition")
        selected = next(item for item in current["items"] if item["evidence_id"] == accepted["result_evidence_id"])
        corrected_quote = selected["quote"].replace(" | ", " ")
        assert corrected_quote != selected["quote"] and "15.12" in corrected_quote
        corrected = post(
            f"files/{version}/recognition:review",
            {
                **request,
                "expected_parse_id": current["parse_id"],
                "evidence_id": selected["evidence_id"],
                "decision": "correct",
                "corrected_quote": corrected_quote,
                "reason": "合成表格摘录去掉分隔符，金额保持15.12",
            },
        )
        final = get(f"files/{version}/recognition")
        assert len(final["history"]) == 2 and final["parse_id"] == corrected["result_parse_id"]
        original = get(f"evidence/{item['evidence_id']}")
        assert original["quote"] == item["quote"] and original["semantic_review"] == "pending"
        raw = client.get(f"files/{version}/content")
        raw.raise_for_status()
        assert hashlib.sha256(raw.content).hexdigest() == digest
        jobs = get(f"projects/{pid}/tasks")
        assert jobs and all(task["kind"] == "parse" for task in jobs)
        result = {
            "synthetic_only": True,
            "project_id": pid,
            "version_id": version,
            "source_sha256": digest,
            "initial_parse_id": before["parse_id"],
            "accepted": accepted,
            "corrected": corrected,
            "numeric_check_required": True,
            "idempotent_replay": True,
            "stale_parse_rejected": True,
            "original_evidence_unchanged": True,
            "original_pdf_unchanged": True,
            "final_pending_count": final["pending_count"],
            "coverage_gaps": final["coverage_gaps"],
            "model_calls": 0,
            "new_analysis_jobs": 0,
            "real_table_extraction": True,
            "ocr_accuracy_evaluated": False,
        }
    (args.output / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"project_id": pid, "real_table_extraction": True, "original_evidence_unchanged": True, "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
