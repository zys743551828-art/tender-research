"""已部署项目问答与过期确认卡；合成资料、一次真实问答，不启动业务评审。"""

import argparse
import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import httpx
from reportlab.pdfgen import canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请使用新的结果路径")
    root = Path(__file__).resolve().parents[2]
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "Synthetic tender and construction plan. Duration: 180 days.")
    pdf.save()
    data = buffer.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    with httpx.Client(base_url="http://localhost:8080/api/v1/", timeout=180) as client:
        login = client.post("auth/login", json={"username": "admin", "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip()})
        login.raise_for_status()
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]

        def post(path, body):
            r = client.post(path, json=body, headers={"Idempotency-Key": str(uuid.uuid4())})
            r.raise_for_status()
            return r.json()

        project = post("projects", {"title": "合成验证：项目问答及过期确认卡", "lot": "测试标段", "settings": {"auto_start": False}})
        pid = project["project_id"]
        for role in ("tender", "body"):
            session = post(f"projects/{pid}/uploads", {"original_name": f"synthetic-{role}.pdf", "role": role, "byte_size": len(data), "sha256": sha, "media_type": "application/pdf"})
            parts = []
            for part in session["parts"]:
                n = part["part_number"]
                chunk = data[(n-1)*session["part_bytes"]:n*session["part_bytes"]]
                r = client.put(f"uploads/{session['upload_id']}/parts/{n}", content=chunk, headers={"Content-Type": "application/octet-stream"})
                r.raise_for_status()
                parts.append({"part_number": n, "etag": r.headers["etag"]})
            post(f"uploads/{session['upload_id']}/complete", {"sha256": sha, "parts": parts})
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            r = client.get(f"projects/{pid}/pipeline")
            r.raise_for_status()
            state = r.json()
            if state["stage"] == "waiting_start":
                break
            time.sleep(0.5)
        assert state["stage"] == "waiting_start", state
        reply = post(f"projects/{pid}/assistant", {"message": "检查新版：请说明当前缺少什么检查，不要实际执行操作。"})
        assert reply["mode"] == "model" and reply["cards"][0]["kind"] == "analysis", reply
        jobs = client.get(f"projects/{pid}/tasks").json()
        assert jobs and all(j["kind"] == "parse" for j in jobs), jobs
        current = client.get(f"projects/{pid}").json()
        changed = client.patch(f"projects/{pid}", json={"title": current["title"] + "（输入已变）"}, headers={"If-Match": str(current["revision"]), "Idempotency-Key": str(uuid.uuid4())})
        changed.raise_for_status()
        stale = client.post(f"projects/{pid}/analysis", json={"expected_context": reply["cards"][0]["expected_context"]}, headers={"Idempotency-Key": str(uuid.uuid4())})
        assert stale.status_code == 409 and stale.json()["code"] == "assistant_card_stale", stale.text
        jobs_after = client.get(f"projects/{pid}/tasks").json()
        assert {j["task_id"] for j in jobs_after} == {j["task_id"] for j in jobs}
        result = {"synthetic": True, "project_id": pid, "question_requests": 1, "reply": reply, "stale_response": stale.json(), "no_analysis_started": True, "jobs": jobs_after}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"project_id": pid, "model_reply": True, "stale_card_rejected": True, "no_analysis_started": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
