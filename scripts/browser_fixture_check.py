"""为内置浏览器验收准备合成 API 上传夹具，或只读保存验收后的产物。"""

import argparse
import hashlib
import io
import json
import time
import uuid
import zipfile
from pathlib import Path

import httpx
from outcome_identity_live_check import pdf


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    with httpx.Client(base_url="http://localhost:8080/api/v1/", timeout=120, trust_env=False) as client:
        login = client.post(
            "auth/login",
            json={"username": "admin", "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip()},
        )
        login.raise_for_status()
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]

        def request(method, path, body=None, headers=None):
            response = client.request(method, path, json=body, headers={"Idempotency-Key": str(uuid.uuid4()), **(headers or {})})
            response.raise_for_status()
            return response

        pid = args.project
        project = request("GET", f"projects/{pid}").json()
        assert project["title"].startswith("浏览器一期融合验收-合成资料")
        save(args.output / "project.json", project)
        if args.capture:
            for suffix in ("pipeline", "report", "rules", "tasks", "findings"):
                save(args.output / f"{suffix}.json", request("GET", f"projects/{pid}/{suffix}").json())
            state = json.loads((args.output / "pipeline.json").read_text(encoding="utf-8"))
            if state["snapshot_id"]:
                save(args.output / "snapshot.json", request("GET", f"snapshots/{state['snapshot_id']}").json())
            word = request("GET", f"projects/{pid}/report.docx").content
            with zipfile.ZipFile(io.BytesIO(word)) as archive:
                assert "word/document.xml" in archive.namelist()
            (args.output / "report.docx").write_bytes(word)
            annotated = request("GET", f"projects/{pid}/body.annotated.pdf").content
            assert annotated.startswith(b"%PDF-")
            (args.output / "body.annotated.pdf").write_bytes(annotated)
            save(
                args.output / "exports.json",
                {
                    "scope": "API content validation, not browser download completion or visual layout",
                    "word_bytes": len(word),
                    "word_sha256": hashlib.sha256(word).hexdigest(),
                    "pdf_bytes": len(annotated),
                    "pdf_sha256": hashlib.sha256(annotated).hexdigest(),
                },
            )
            print("验收产物与 API 导出内容已保存", flush=True)
            return
        assert request("GET", f"projects/{pid}/files").json() == []
        request("PATCH", f"projects/{pid}", {"settings": {**project["settings"], "auto_start": False}}, {"If-Match": str(project["revision"])})
        fixtures = {
            "tender": [
                "合成招标文件，仅用于浏览器验收，不是真实投标。",
                "第一章 强制性要求",
                "施工总工期不得超过180日历天，工程质量必须达到合格。",
                "第二章 技术评分标准",
                "施工进度计划合理可行的得8分，计划不合理的得0分。",
                "技术标满分为8分。",
            ],
            "body": [
                "合成施工组织设计，仅用于浏览器验收。",
                "第一章 施工进度计划",
                "施工总工期为240日历天，准备20天、管线100天、基层80天、面层40天。",
                "关键线路为准备、管线、基层、面层；配置2个班组和2台挖掘机。",
                "第二章 质量保证",
                "工程质量目标为合格，按设计及验收规范施工。",
            ],
        }
        for role, lines in fixtures.items():
            data = pdf(lines)
            (args.output / f"synthetic-{role}.pdf").write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            upload = request(
                "POST",
                f"projects/{pid}/uploads",
                {"original_name": f"浏览器验收-合成{role}.pdf", "role": role, "byte_size": len(data), "sha256": digest, "media_type": "application/pdf"},
            ).json()
            parts = []
            for part in upload["parts"]:
                n = part["part_number"]
                assert part["url"] == f"/api/v1/uploads/{upload['upload_id']}/parts/{n}"
                response = client.put(
                    f"uploads/{upload['upload_id']}/parts/{n}",
                    content=data[(n - 1) * upload["part_bytes"] : n * upload["part_bytes"]],
                    headers={"Content-Type": "application/octet-stream"},
                )
                response.raise_for_status()
                parts.append({"part_number": n, "etag": response.headers["etag"]})
            save(args.output / f"upload-{role}.json", request("POST", f"uploads/{upload['upload_id']}/complete", {"sha256": digest, "parts": parts}).json())
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = request("GET", f"projects/{pid}/pipeline").json()
            if state["stage"] in ("waiting_start", "failed"):
                break
            time.sleep(1)
        save(args.output / "pipeline.json", state)
        assert state["stage"] == "waiting_start", state
        save(
            args.output / "scope.json",
            {
                "synthetic_only": True,
                "project_created_through_browser": True,
                "files_uploaded_through_browser": False,
                "files_uploaded_through_same_origin_api": True,
                "analysis_started_by_script": False,
            },
        )
        print(json.dumps({"project_id": pid, "stage": state["stage"], "analysis_started_by_script": False}), flush=True)


if __name__ == "__main__":
    main()
