"""合成资料真实评审后，实测提交身份复用、未知成绩与旧稿拦截。"""

import argparse
import hashlib
import io
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


def pdf(text):
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    buffer = io.BytesIO()
    writer = canvas.Canvas(buffer)
    writer.setFont("STSong-Light", 12)
    for n, line in enumerate(text):
        writer.drawString(60, 760-24*n, line)
    writer.save()
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("使用新的结果路径")
    root = Path(__file__).resolve().parents[2]
    with httpx.Client(base_url="http://localhost:8080/api/v1/", timeout=120) as client:
        login = client.post("auth/login", json={"username": "admin", "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip()})
        login.raise_for_status()
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]

        def get(path):
            r = client.get(path)
            r.raise_for_status()
            return r.json()

        def post(path, body, key=None):
            r = client.post(path, json=body, headers={"Idempotency-Key": key or str(uuid.uuid4())})
            r.raise_for_status()
            return r.json()

        project = post("projects", {"title": "合成验证：成绩身份与未知状态", "lot": "合成测试标段", "settings": {"auto_start": True, "consistency_check": False, "scope_check": False}})
        pid = project["project_id"]
        for role, lines in (("tender", ["合成招标文件，仅用于系统验证。", "强制性要求：施工总工期不得超过180日历天。"]), ("body", ["合成施工组织设计，仅用于系统验证。", "施工总工期为180日历天，满足招标工期要求。"] )):
            data = pdf(lines)
            sha = hashlib.sha256(data).hexdigest()
            session = post(f"projects/{pid}/uploads", {"original_name": f"{role}-synthetic.pdf", "role": role, "media_type": "application/pdf", "byte_size": len(data), "sha256": sha})
            parts = []
            for part in session["parts"]:
                n = part["part_number"]
                r = client.put(f"uploads/{session['upload_id']}/parts/{n}", content=data[(n-1)*session["part_bytes"]:n*session["part_bytes"]], headers={"Content-Type": "application/octet-stream"})
                r.raise_for_status()
                parts.append({"part_number": n, "etag": r.headers["etag"]})
            post(f"uploads/{session['upload_id']}/complete", {"sha256": sha, "parts": parts})
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            state = get(f"projects/{pid}/pipeline")
            if state["stage"] in ("done", "failed"):
                break
            time.sleep(1)
        assert state["stage"] == "done", state
        package = get(f"projects/{pid}/rules")
        ids = [r["rule_id"] for r in package["rules"] if r["source_status"] == "active" and r["confirmation"] == "pending"]
        assert ids and all("180" in r["statement"] for r in package["rules"]), package
        post(f"projects/{pid}/rules:confirm", {"object_version": str(package["revision"]), "rule_ids": ids})
        report = get(f"projects/{pid}/report")
        assert report["qualification"]["qualification"]["violated"] == 0
        snapshot = get(f"snapshots/{state['snapshot_id']}")
        request = {"body_version_id": snapshot["body_version_id"], "snapshot_id": snapshot["snapshot_id"], "submitted_at": datetime.now(timezone.utc).isoformat(), "receipt_artifact_id": None}
        first = post(f"projects/{pid}/submissions", request)
        assert first["body_name"] == "body-synthetic.pdf" and len(first["body_sha256"]) == 64
        assert first["round"] == project["round"] and first["lot"] == project["lot"] and first["project_title"] == project["title"]
        project = get(f"projects/{pid}")
        draft = {"submission_id": first["submission_id"], "source_kind": "user_reported", "target": "technical_total", "value": None, "maximum": None, "score_unknown": True, "expected_project_revision": project["revision"], "aggregation": None, "stage_result": "pending", "source_page": None, "published_at": None, "supersedes": None, "source_artifact_id": None, "source_description": "合成验证：尚未公布成绩，无真实官方成绩或投标行为"}
        key = str(uuid.uuid4())
        unknown = post(f"projects/{pid}/outcomes", draft, key)
        assert unknown["value"] is None and unknown["maximum"] is None and unknown["verification_status"] == "pending" and unknown["target"] == "technical_total"
        assert post(f"projects/{pid}/outcomes", draft, key) == unknown
        second = post(f"projects/{pid}/submissions", request)
        assert second["supersedes"] == first["submission_id"]
        rejected = client.post(f"projects/{pid}/outcomes", json=draft, headers={"Idempotency-Key": str(uuid.uuid4())})
        assert rejected.status_code == 409 and rejected.json()["code"] == "submission_superseded", rejected.text
        assert get(f"projects/{pid}/outcomes")[0] == unknown
        result = {"synthetic": True, "project_id": pid, "report": report, "first_submission": first, "unknown_outcome": unknown, "corrected_submission": second, "old_identity_rejection": rejected.json(), "old_outcome_preserved": True, "idempotency_replay": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"project_id": pid, "unknown_preserved": True, "old_submission_rejected": True, "historical_outcome_preserved": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
