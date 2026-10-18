"""上传纯栅格两页合成表格，验证正式 OCR、结构预览、可能续表及追加核验。"""

import argparse
import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import httpx
from bidwb.parsing import parse_pdf
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def sample():
    target = io.BytesIO()
    pdf = canvas.Canvas(target, pagesize=(595, 842))
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)
    for ys in ([1050, 1220, 1390, 1560], [80, 250, 420, 590]):
        image = Image.new("RGB", (1190, 1684), "white")
        draw = ImageDraw.Draw(image)
        xs = [100, 550, 1090]
        for x in xs:
            draw.line((x, ys[0], x, ys[-1]), fill="black", width=4)
        for y in ys:
            draw.line((xs[0], y, xs[-1], y), fill="black", width=4)
        for row, cells in enumerate((("Item", "Amount"), ("Cost", "15.12"), ("Duration", "180"))):
            for col, value in enumerate(cells):
                draw.text((xs[col] + 25, ys[row] + 50), value, font=font, fill="black")
        pdf.drawImage(ImageReader(image), 0, 0, width=595, height=842)
        pdf.showPage()
    pdf.save()
    return target.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    data = sample()
    original = parse_pdf(data)
    assert original.page_count == 2 and not original.chunks and original.page_text_chars == [0, 0]
    (args.output / "synthetic-scanned-tables.pdf").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    root = Path(__file__).resolve().parents[2]
    with httpx.Client(base_url="http://localhost:8080/api/v1/", timeout=60) as client:
        login = client.post(
            "auth/login",
            json={
                "username": "admin",
                "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip(),
            },
        )
        login.raise_for_status()
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]

        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()

        def post(path, value):
            response = client.post(path, json=value, headers={"Idempotency-Key": uuid.uuid4().hex})
            response.raise_for_status()
            return response.json()

        pid = post("projects", {"title": "合成验证：纯扫描表格结构与可能续表", "settings": {"auto_start": False}})["project_id"]
        upload = post(
            f"projects/{pid}/uploads",
            {
                "original_name": "synthetic-scanned-tables.pdf",
                "role": "body",
                "media_type": "application/pdf",
                "sha256": digest,
                "byte_size": len(data),
            },
        )
        parts = []
        for part in upload["parts"]:
            n = part["part_number"]
            response = client.put(f"uploads/{upload['upload_id']}/parts/{n}", content=data[(n - 1) * upload["part_bytes"] : n * upload["part_bytes"]])
            response.raise_for_status()
            parts.append({"part_number": n, "etag": response.headers["etag"]})
        version = post(f"uploads/{upload['upload_id']}/complete", {"sha256": digest, "parts": parts})["version_id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            candidates = get(f"files/{version}/recognition")
            if candidates["parse_id"] and candidates["can_review"]:
                break
            time.sleep(1)
        else:
            raise AssertionError("扫描表解析未在限时内完成")
        first = get(f"files/{version}/table-structures")
        second = get(f"files/{version}/table-structures?table_index=1")
        (args.output / "first-table.json").write_text(json.dumps(first, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output / "second-table.json").write_text(json.dumps(second, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        assert first["table_count"] == 2 and first["physical_page"] == 1 and second["physical_page"] == 2
        assert first["source"] == second["source"] == "raster_grid"
        assert first["row_count"] == second["row_count"] == 3 and all(len(row) == 2 for row in first["rows"] + second["rows"])
        assert any("15.12" in (cell or "") for row in first["rows"] for cell in row)
        assert first["continuations"] and first["continuations"][0]["review"] == "pending"
        assert "table_structure_not_verified" in candidates["coverage_gaps"] and "cross_page_tables_not_verified" in candidates["coverage_gaps"]
        assert "recognition_verification_required" in candidates["coverage_gaps"]
        item = next(row for row in candidates["items"] if row["kind"] == "table" and "15.12" in row["quote"])
        reviewed = post(
            f"files/{version}/recognition:review",
            {
                "expected_parse_id": candidates["parse_id"],
                "evidence_id": item["evidence_id"],
                "decision": "accept",
                "reason": "合成扫描原页逐格核对15.12",
                "reviewed_original": True,
                "reviewed_numbers": True,
            },
        )
        after = get(f"files/{version}/table-structures")
        assert after["rows"] == first["rows"] and after["continuations"] == first["continuations"] and after["parse_id"] != first["parse_id"]
        stale = client.get(f"files/{version}/table-structures?expected_parse_id={first['parse_id']}")
        assert stale.status_code == 409 and stale.json()["code"] == "recognition_parse_stale"
        assert get(f"evidence/{item['evidence_id']}")["semantic_review"] == "pending"
        current = get(f"files/{version}/recognition")
        assert "table_structure_not_verified" in current["coverage_gaps"] and "cross_page_tables_not_verified" in current["coverage_gaps"]
        assert "recognition_verification_required" in current["coverage_gaps"]
        content = client.get(f"files/{version}/content")
        content.raise_for_status()
        assert hashlib.sha256(content.content).hexdigest() == digest
        tasks = get(f"projects/{pid}/tasks")
        assert tasks and all(task["kind"] == "parse" for task in tasks)
        result = {
            "synthetic_only": True,
            "project_id": pid,
            "version_id": version,
            "source_sha256": digest,
            "native_text_characters": [0, 0],
            "real_local_ocr_and_raster_grid": True,
            "tables": 2,
            "rows_per_table": 3,
            "columns": 2,
            "repeated_header_continuation_pending": True,
            "numeric_cell_15_12_found": True,
            "review": reviewed,
            "original_pdf_and_evidence_preserved": True,
            "structure_unchanged_after_excerpt_review": True,
            "stale_structure_request_status": 409,
            "structure_and_partial_ocr_gaps_retained": True,
            "model_calls": 0,
            "new_business_analysis_jobs": 0,
            "real_bid_quality_evaluated": False,
        }
    (args.output / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
