"""一次运行全部本地实验与测试并保存记录；不调用外部模型。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path

from . import contracts, paths
from .outcomes import run as evaluate_outcomes
from .retrieval_eval import run as evaluate_rules
from .rule_checks import run as evaluate_checks


class RecordedResult(unittest.TestResult):
    def __init__(self):
        super().__init__(); self.records = []
    def addSuccess(self, test):
        super().addSuccess(test); self.records.append({"test_id":test.id(),"status":"passed"})
    def addFailure(self, test, err):
        super().addFailure(test,err); self.records.append({"test_id":test.id(),"status":"failed","error":self._exc_info_to_string(err,test)})
    def addError(self, test, err):
        super().addError(test,err); self.records.append({"test_id":test.id(),"status":"error","error":self._exc_info_to_string(err,test)})
    def addSkip(self, test, reason):
        super().addSkip(test,reason); self.records.append({"test_id":test.id(),"status":"skipped","reason":reason})


def run(datasets: Path = paths.DATASETS, output: Path = paths.RESULTS) -> int:
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    evaluate_rules(datasets / "公开规则资料/资料清单.json", datasets / "评测集/RULE-001.json", output / "retrieval-rules.json", "retrieval-rules")
    checks = evaluate_checks(datasets, output)
    outcomes = evaluate_outcomes(datasets, output / "outcome-import.json", "outcome-import")
    suite = unittest.defaultTestLoader.discover(str(paths.REPO / "tests"), top_level_dir=str(paths.REPO))
    result = RecordedResult()
    suite.run(result)
    schemas = {name: getattr(contracts, name).model_json_schema() for name in ("Evidence", "OfficialLabel", "ReportedResult", "PairSample", "SystemPrediction", "EvalProtocol", "Finding", "Prediction")}
    (output / "contracts.schema.json").write_text(json.dumps(schemas, ensure_ascii=False, indent=2) + "\n")
    sources = {}
    for path in list(datasets.glob("CASE-*/资料清单.json")) + [datasets / "公开规则资料/资料清单.json"]:
        for row in json.loads(path.read_text())["file_records"]:
            source = paths.original(path.parent, row.get("relative_path", row.get("path")))
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            sources[str(source.relative_to(datasets))] = {"sha256": actual, "matches_manifest": actual == row["sha256"]}
    key_path = paths.REPO / ".secrets/deepseek_api_key"
    key = key_path.read_text().strip() if key_path.exists() else None
    # 只返回是否发现泄漏及路径，永不返回匹配内容。
    leaks = [str(p.relative_to(paths.REPO)) for p in paths.REPO.rglob("*")
             if key and p.is_file() and ".secrets" not in p.parts and ".git" not in p.parts and key.encode() in p.read_bytes()]
    credential_private = not key_path.exists() or not key_path.stat().st_mode & 0o077
    passed = (result.wasSuccessful() and all(c["passed"] for c in checks["checks"] + outcomes["checks"])
              and all(r["matches_manifest"] for r in sources.values()) and not leaks and credential_private)
    report = {"experiment_id": "test-record", "provenance": "synthetic_contract_tests_and_real_source_integrity",
              "started_at": started_at, "finished_at": datetime.now(UTC).isoformat(),
              "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                              "libraries": {n: importlib.metadata.version(n) for n in ("pydantic", "numpy", "pypdf", "pdfplumber", "pypdfium2")}},
              "tests_run": result.testsRun, "failed": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "records": result.records, "source_integrity": sources,
              "credential_leak_files": leaks, "credential_file_permissions_private": credential_private,
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
              "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.json")) if p.name != "test-record.json"},
              "elapsed_seconds": round(time.perf_counter() - start, 3), "local_checks_passed": passed,
              "limits": ["夹具分数不证明真实预测精度", "本地发布回退不代替生产联调或正式发布", "角色参数检查不代替身份认证"]}
    (output / "test-record.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"tests_run": result.testsRun, "local_checks_passed": passed,
                      "original_files_unchanged": all(r["matches_manifest"] for r in sources.values()),
                      "credential_leak_found": bool(leaks)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=Path, default=paths.DATASETS)
    p.add_argument("--output", type=Path, default=paths.RESULTS)
    a = p.parse_args()
    raise SystemExit(run(a.datasets, a.output))
