"""复用合成项目，验证实际原件上的局部效力、检索提示和新旧输入保留；不调用模型。"""
import argparse
import json
import uuid
from decimal import Decimal
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", required=True)
parser.add_argument("--old-snapshot", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
output_path = args.output
if output_path.exists():
    raise FileExistsError("使用新的验证输出，不能覆盖")
pid = args.project
with httpx.Client(base_url="http://localhost:8080/api/v1", timeout=60) as client:
    response = client.post("/auth/login", json={"username": "admin", "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip()})
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

    before = get(f"/projects/{pid}/rules")
    original_snapshot = get(f"/snapshots/{args.old_snapshot}")
    assert original_snapshot["rule_set_id"] == before["rule_set_id"], "先核对专用合成项目及初始快照"
    target = next(rule for rule in before["rules"] if rule["kind"] == "scoring" and rule["max_points"] and Decimal(rule["max_points"]) == Decimal(60)
                  and any(ref["file_role"] == "tender" for ref in rule["evidence"]))
    old_ref = next(ref for ref in target["evidence"] if ref["file_role"] == "tender")
    new_ref = before["terms"]["technical_full_score"]["evidence"][0]
    available = get(f"/projects/{pid}/rules/source-evidence/{new_ref['version_id']}")
    basis = next(ref for ref in available if "80" in ref["quote"])
    request = {"object_version": str(before["revision"]), "old_version_id": old_ref["version_id"], "new_version_id": new_ref["version_id"],
               "action": "partial_replace", "rule_ids": [target["rule_id"]]}
    preview = post(f"/projects/{pid}/rules/effectivity:preview", request)
    assert preview["affected_rule_ids"] == [target["rule_id"]]
    after = post(f"/projects/{pid}/rules/effectivity:decide", {**request, "scope_basis": [{"evidence_id": basis["evidence_id"], "quote": basis["quote"]}],
                                                         "reason": "合成验证：补遗明确满分80分，仅结束原60分要求，安全文明施工要求继续保留。"})
    ended = [rule for rule in after["rules"] if rule["source_status"] == "superseded"]
    assert len(ended) == 1 and ended[0]["statement"] == target["statement"]
    assert any(rule["source_status"] == "active" and rule["kind"] != "scoring" and any(ref["version_id"] == old_ref["version_id"] for ref in rule["evidence"]) for rule in after["rules"])
    original_page = get(f"/evidence/{old_ref['evidence_id']}")
    assert original_page["quote"] == old_ref["quote"] and any("已被补遗替代" in note for note in original_page["effectivity_notes"])
    search = post(f"/projects/{pid}/search", {"query": "60", "top_k": 10, "mode": "keyword"})
    assert any(hit["evidence"]["effectivity_notes"] for hit in search["hits"])
    old_snapshot = get(f"/snapshots/{args.old_snapshot}")
    assert old_snapshot["rule_set_id"] == before["rule_set_id"]
    files = get(f"/projects/{pid}/files")
    body = next(file["version_id"] for file in files if file["role"] == "body")
    snapshot = post(f"/projects/{pid}/snapshots", {"body_version_id": body, "rule_set_id": after["rule_set_id"], "fact_set_id": None,
                                                 "file_version_ids": [*after["source_version_ids"], body], "reference_dataset_id": None})
    output = {"synthetic": True, "project_id": pid, "before": before, "preview": preview, "after": after, "old_evidence": original_page,
              "old_snapshot": old_snapshot["snapshot_id"], "new_snapshot": snapshot["snapshot_id"], "model_calls": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"project_id": pid, "ended_rules": len(ended), "other_requirements_retained": True,
                      "old_snapshot_preserved": True, "new_snapshot": snapshot["snapshot_id"], "model_calls": 0}, ensure_ascii=False))

