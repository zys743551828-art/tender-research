"""已部署同源上传、真实规则提取、投标条件冲突阻断与核实；仅用合成资料。"""
import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "tender-app/backend"))
from tests.pdfs import make_pdf  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("使用新的实验输出路径，不能覆盖原验证")
with httpx.Client(base_url="http://localhost:8080/api/v1", timeout=60) as client:
    login = client.post("/auth/login", json={"username": "admin", "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip()})
    login.raise_for_status()
    client.headers["X-CSRF-Token"] = login.json()["csrf_token"]

    def post(path, body):
        return client.post(path, json=body, headers={"Idempotency-Key": str(uuid.uuid4())})

    if args.project:
        pid = args.project
    else:
        project = post("/projects", {"title": "补遗投标条件核实验证（合成资料）", "tender_number": None, "lot": None, "round": 1,
                                     "region": "合肥", "engineering_type": "市政道路", "deadline": None, "settings": {"auto_start": False}})
        project.raise_for_status()
        pid = project.json()["project_id"]
    print(json.dumps({"project_id": pid, "stage": "created"}), flush=True)
    versions = []
    for role, quote in (("tender", "本项目技术标满分60分。施工组织设计应满足安全文明施工要求。"),
                        ("supplement", "本项目补遗说明：技术标满分80分，以本补遗为准。施工组织设计应满足安全文明施工要求。"),
                        ("body", "本项目施工组织设计：设置围挡，落实施工现场安全防护措施。")):
        data = make_pdf([[quote]])
        sha = hashlib.sha256(data).hexdigest()
        response = post(f"/projects/{pid}/uploads", {"original_name": f"合成{role}.pdf", "role": role, "byte_size": len(data), "sha256": sha, "media_type": "application/pdf"})
        response.raise_for_status()
        upload = response.json()
        headers = upload["parts"][0].get("headers", {})
        part = client.put(f"/uploads/{upload['upload_id']}/parts/1", content=data, headers={**headers, "Content-Type": "application/octet-stream"})
        part.raise_for_status()
        response = post(f"/uploads/{upload['upload_id']}/complete", {"sha256": sha, "parts": [{"part_number": 1, "etag": part.headers["etag"]}]})
        response.raise_for_status()
        versions.append(response.json()["version_id"])

    def wait_jobs(kind):
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            response = client.get(f"/projects/{pid}/tasks")
            response.raise_for_status()
            tasks = [task for task in response.json() if task["kind"] == kind]
            if tasks and all(task["state"] in ("succeeded", "failed", "cancelled") for task in tasks):
                assert all(task["state"] == "succeeded" for task in tasks), tasks
                return tasks
            time.sleep(1)
        raise TimeoutError(kind)

    wait_jobs("parse")
    response = post(f"/projects/{pid}/rules:extract", {})
    response.raise_for_status()
    wait_jobs("rules")
    response = client.get(f"/projects/{pid}/rules")
    response.raise_for_status()
    before = response.json()
    assert before["status"] == "conflict", before
    conflict = next(item for item in before["terms"]["conflicts"] if item["term"] == "technical_full_score")
    assert {item["values"]["value"] for item in conflict["alternatives"]} == {"60", "80"}, conflict
    snapshot = {"body_version_id": versions[2], "rule_set_id": before["rule_set_id"], "fact_set_id": None, "file_version_ids": versions, "reference_dataset_id": None}
    blocked = post(f"/projects/{pid}/snapshots", snapshot)
    assert blocked.status_code == 409 and blocked.json()["code"] == "rule_terms_conflict", blocked.text
    chosen = next(item for item in conflict["alternatives"] if item["values"]["value"] == "80")
    response = post(f"/projects/{pid}/rules/terms:resolve", {"object_version": str(before["revision"]), "term": "technical_full_score", "option_id": chosen["option_id"], "reason": "合成测试：已核对补遗明示技术标满分80分，本次仅核实满分条件。"})
    response.raise_for_status()
    after = response.json()
    assert after["terms"]["technical_full_score"]["value"] == "80" and after["rule_set_id"] != before["rule_set_id"]
    snapshot["rule_set_id"] = after["rule_set_id"]
    frozen = post(f"/projects/{pid}/snapshots", snapshot)
    frozen.raise_for_status()
    output = {"synthetic": True, "project_id": pid, "before": before, "after": after, "blocked_code": blocked.json()["code"],
              "snapshot_id": frozen.json()["snapshot_id"], "automatic_analysis": False, "prediction_invoked": False}
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"project_id": pid, "blocked_code": output["blocked_code"], "selected_value": "80", "new_revision": after["revision"], "snapshot_id": output["snapshot_id"]}, ensure_ascii=False), flush=True)

