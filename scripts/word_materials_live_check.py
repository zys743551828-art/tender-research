"""已部署 Word 上传、隔离转换、逐稿 PDF 补交及历史保留；仅创建合成项目，不调用业务模型。"""

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import httpx
from word_converter_check import word


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("结果必须使用新的目录")
    root = Path(__file__).resolve().parents[2]
    with httpx.Client(base_url="http://localhost:8080/api/v1/", timeout=60) as client:
        response = client.post(
            "auth/login",
            json={"username": "admin", "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip()},
        )
        response.raise_for_status()
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]

        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()

        def post(path, body):
            response = client.post(path, json=body, headers={"Idempotency-Key": str(uuid.uuid4())})
            response.raise_for_status()
            return response.json()

        pid = post("projects", {"title": "合成验证：Word 转换与逐稿 PDF 补交", "settings": {"auto_start": False}})["project_id"]

        def upload(data, name, media, source=None, version_of=None):
            sha = hashlib.sha256(data).hexdigest()
            request = {
                "original_name": name,
                "role": "body",
                "byte_size": len(data),
                "sha256": sha,
                "media_type": media,
                "source_word_version_id": source,
                "version_of_file_id": version_of,
            }
            session = post(f"projects/{pid}/uploads", request)
            parts = []
            for part in session["parts"]:
                index = part["part_number"]
                chunk = data[(index - 1) * session["part_bytes"] : index * session["part_bytes"]]
                response = client.put(f"uploads/{session['upload_id']}/parts/{index}", content=chunk, headers={"Content-Type": "application/octet-stream"})
                response.raise_for_status()
                parts.append({"part_number": index, "etag": response.headers["etag"]})
            return post(f"uploads/{session['upload_id']}/complete", {"sha256": sha, "parts": parts})

        def wait_word(identity, expected):
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                versions = get(f"projects/{pid}/files")
                entry = next(v for v in versions if v["version_id"] == identity)
                if entry["word"]["state"] == expected:
                    return entry, versions
                time.sleep(0.5)
            raise AssertionError({"word": identity, "expected": expected, "last": entry})

        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        first = upload(word(), "施组一.docx", media)
        converted, versions = wait_word(first["version_id"], "converted")
        adopted = converted["word"]["adopted_pdf_version_id"]
        assert adopted
        actual = next(v for v in versions if v["version_id"] == adopted)
        assert actual["file_id"] != first["file_id"] and actual["supersedes"] is None and actual["source_word_version_id"] == first["version_id"]
        response = client.get(f"files/{adopted}/content")
        response.raise_for_status()
        pdf = response.content
        assert hashlib.sha256(pdf).hexdigest() == actual["artifact"]["sha256"]
        post(f"files/{first['version_id']}/word:retry", {})
        retry_deadline = time.monotonic() + 150
        while time.monotonic() < retry_deadline:
            retried = next(v for v in get(f"projects/{pid}/files") if v["version_id"] == first["version_id"])
            if len(retried["word"]["candidates"]) >= 2:
                break
            time.sleep(0.5)
        assert len(retried["word"]["candidates"]) == 2 and retried["word"]["adopted_pdf_version_id"] == adopted
        blocked = upload(word('<w:p><w:fldSimple w:instr="PAGE"/></w:p>'), "外部改稿.docx", media)
        rejected, _ = wait_word(blocked["version_id"], "failed")
        assert not rejected["word"]["adopted_pdf_version_id"] and not rejected["word"]["candidates"]
        fallback = upload(pdf, "此稿实际PDF.pdf", "application/pdf", blocked["version_id"])
        listed = get(f"projects/{pid}/files")
        before = next(v for v in listed if v["version_id"] == blocked["version_id"])
        assert before["word"]["adopted_pdf_version_id"] == fallback["version_id"]
        updated = upload(
            word(
                '<w:p><w:r><w:rPr><w:rFonts w:ascii="Noto Sans CJK SC" w:eastAsia="Noto Sans CJK SC"/></w:rPr><w:t>第二版施工组织设计，工期190日历天。</w:t></w:r></w:p>'
            ),
            "外部改稿二.docx",
            media,
            version_of=blocked["file_id"],
        )
        after, versions = wait_word(updated["version_id"], "converted")
        old = next(v for v in versions if v["version_id"] == blocked["version_id"])
        assert not old["word"]["current"] and old["word"]["adopted_pdf_version_id"] == fallback["version_id"]
        response = client.get(f"files/{fallback['version_id']}/content")
        response.raise_for_status()
        assert response.content == pdf
        complex_word = upload(word("<w:tbl><w:tr><w:tc><w:p><w:r><w:t>工期180天</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"), "表格稿.docx", media)
        pending, _ = wait_word(complex_word["version_id"], "needs_review")
        assert pending["word"]["adopted_pdf_version_id"] is None
        candidate = pending["word"]["candidates"][0]
        adopted_status = post(
            f"files/{complex_word['version_id']}/word:adopt",
            {"link_id": candidate["link_id"], "reason": "合成验证：模拟人工核对确认接口，不代表真实标书版式验收", "reviewed_pdf": True},
        )
        assert adopted_status["adopted_pdf_version_id"] == candidate["pdf_version_id"]
        jobs = get(f"projects/{pid}/tasks")
        assert all(j["kind"] == "parse" for j in jobs), jobs
        output = {
            "synthetic": True,
            "project_id": pid,
            "first": converted,
            "actual_pdf": actual,
            "failed_word": rejected,
            "fallback_pdf": fallback,
            "updated_word": after,
            "old_word": old,
            "complex_word": pending,
            "adopted_complex_pdf": adopted_status,
            "conversion_retry_preserves_adoption": True,
            "model_calls": 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "project_id": pid,
                    "controlled_render": True,
                    "one_file_fallback": True,
                    "historical_pdf_preserved": True,
                    "complex_pdf_review": True,
                    "conversion_retry_preserves_adoption": True,
                    "model_calls": 0,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
