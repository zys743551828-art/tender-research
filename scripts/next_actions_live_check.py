"""只读核对已部署项目的六级主要动作与报告／总览一致性，不启动模型或修改项目。"""

import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("使用新的结果路径")
    root = Path(__file__).resolve().parents[2]
    cases = []
    with httpx.Client(base_url="http://localhost:8080/api/v1/", timeout=60) as client:
        login = client.post("auth/login", json={"username": "admin", "password": (root / "tender-app/deploy/secrets/admin_initial_password").read_text(encoding="utf-8").strip()})
        login.raise_for_status()
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        response = client.get("projects")
        response.raise_for_status()
        selected = {"prj_95c10b2577124f25", "prj_9f52024b21274c93", "prj_216cd2513bbf4cb2", "prj_e21c1be8fa1b45ea"}
        for project in response.json():
            pid = project["project_id"]
            if pid not in selected:
                continue
            summary = project["summary"]["next_action"]
            assert summary is None or 1 <= summary["priority"] <= 6
            r = client.get(f"projects/{pid}/report")
            if r.status_code == 404:
                assert r.json()["code"] == "report_not_ready"
                actions = [summary] if summary else []
                stage = "no_report"
            else:
                r.raise_for_status()
                report = r.json()
                actions = report["next_actions"]
                stage = report["pipeline"]["stage"]
                assert summary == (actions[0] if actions else None)
                priorities = [a["priority"] for a in actions]
                assert priorities == sorted(priorities)
                if stage != "done" or report["pipeline"]["stale_reasons"]:
                    assert all(a["key"] not in ("handle_disqualification", "handle_critical", "submit") for a in actions)
            cases.append({"project_id": pid, "stage": stage, "actions": [{k: a[k] for k in ("key", "priority", "count")} for a in actions]})
    assert len(cases) == 4, cases
    waiting = next(c for c in cases if c["project_id"] == "prj_95c10b2577124f25")
    assert waiting["actions"][0] == {"key": "start_analysis", "priority": 2, "count": None}
    corrected = next(c for c in cases if c["project_id"] == "prj_9f52024b21274c93")
    assert corrected["actions"][0]["key"] == "record_outcome" and corrected["actions"][0]["priority"] == 6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"synthetic_projects": True, "model_calls": 0, "read_only": True, "cases": cases}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"projects": len(cases), "summary_report_match": True, "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
