"""部署前后核对合成项目历史快照与报告，部署后尝试应被拒绝的旧输入重放。"""

import argparse
import json
import uuid
from pathlib import Path

import httpx

PROJECTS = ("prj_9f52024b21274c93", "prj_e21c1be8fa1b45ea")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--check-rule-sources", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("使用新的结果路径")
    root = Path(__file__).resolve().parents[2]
    cases, identities = [], []
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

        for pid in PROJECTS:
            snapshots = get(f"projects/{pid}/snapshots")
            report = get(f"projects/{pid}/report")
            jobs_before = sorted(task["task_id"] for task in get(f"projects/{pid}/tasks"))
            stable_snapshots = [
                {
                    key: snapshot[key]
                    for key in (
                        "snapshot_id",
                        "sha256",
                        "contract_version",
                        "file_version_ids",
                        "body_version_id",
                        "rule_set_id",
                        "fact_set_id",
                        "as_of",
                        "word_sources",
                    )
                }
                for snapshot in snapshots
            ]
            stable_report = {key: report[key] for key in ("snapshot_id", "qualification", "prediction", "categories")}
            if args.baseline:
                assert report["basis"]["parse_identity"] == "legacy_unfrozen"
                for snapshot in snapshots:
                    assert snapshot["parse_identity"] == "legacy_unfrozen" and snapshot["parse_ids"] == {}
                    assert "legacy_parse_identity_unfrozen" in snapshot["coverage_gaps"]
                client.headers["Idempotency-Key"] = "legacy-probe-" + uuid.uuid4().hex
                rejected = client.post(f"projects/{pid}/diagnosis", json={"snapshot_id": report["snapshot_id"]})
                assert rejected.status_code == 409 and rejected.json()["code"] == "snapshot_parse_identity_unknown"
                jobs_after = sorted(task["task_id"] for task in get(f"projects/{pid}/tasks"))
                assert jobs_after == jobs_before
                rule_check = {}
                if args.check_rule_sources:
                    rules = get(f"projects/{pid}/rules")
                    assert rules["source_parse_state"] == "legacy_unfrozen"
                    pipeline = get(f"projects/{pid}/pipeline")
                    assert pipeline["stage"] == "waiting_start" and not pipeline["input_verification_required"]
                    client.headers["Idempotency-Key"] = "legacy-rule-probe-" + uuid.uuid4().hex
                    confirmation = client.post(
                        f"projects/{pid}/rules:confirm",
                        json={"object_version": str(rules["revision"]), "rule_ids": [rule["rule_id"] for rule in rules["rules"][:1]]},
                    )
                    assert confirmation.status_code == 409 and confirmation.json()["code"] == "rule_parse_identity_unknown"
                    assert sorted(task["task_id"] for task in get(f"projects/{pid}/tasks")) == jobs_before
                    rule_check = {
                        "source_parse_state": rules["source_parse_state"],
                        "stage": pipeline["stage"],
                        "legacy_confirmation_status": confirmation.status_code,
                    }
                identities.append(
                    {
                        "project_id": pid,
                        "legacy_snapshots": len(snapshots),
                        "report_readable": True,
                        "legacy_replay_status": rejected.status_code,
                        "no_new_jobs": True,
                        **rule_check,
                    }
                )
            cases.append({"project_id": pid, "snapshots": stable_snapshots, "report": stable_report})
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        assert cases == baseline["cases"], "旧快照登记值或历史报告结论发生变化"
    result = {"synthetic_projects": True, "model_calls": 0, "baseline_matched": bool(args.baseline), "cases": cases, "checks": identities}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"projects": len(cases), "baseline_matched": bool(args.baseline), "model_calls": 0}))


if __name__ == "__main__":
    main()
