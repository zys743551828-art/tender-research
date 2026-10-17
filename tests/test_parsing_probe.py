"""探针不能用错误原件、内嵌图片或未跑页面制造命中与覆盖结论。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("parsing_probe", SCRIPTS / "parsing_probe.py")
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
sys.modules["parsing_probe"] = probe
spec_report = importlib.util.spec_from_file_location("parsing_probe_report", SCRIPTS / "parsing_probe_report.py")
assert spec_report and spec_report.loader
report = importlib.util.module_from_spec(spec_report)
spec_report.loader.exec_module(report)


def test_embedded_image_does_not_count_as_text() -> None:
    text = probe.without_embedded_images('收费表<img src="data:image/jpeg;base64,15.12万元">备注')
    assert "15.12万元" not in text and "收费表" in text and "备注" in text


def test_normalize_keeps_wrong_numbers_distinct() -> None:
    assert probe.normalize("１５.１２ 万元\n") == probe.normalize("15.12万元")
    assert probe.normalize("15.12万元") != probe.normalize("15.21万元")


def test_wrong_source_rejected_before_engine_load(tmp_path: Path) -> None:
    (tmp_path / "source.pdf").write_bytes(b"incorrect original")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"samples": [{"id": "one", "source_path": "source.pdf", "source_sha256": "0" * 64}]}))
    result = subprocess.run([
        sys.executable, str(SCRIPTS / "parsing_probe.py"), "--engine", "docling", "--protocol", str(protocol),
        "--datasets", str(tmp_path), "--output", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"),
    ], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode != 0 and "Source hash mismatch" in result.stderr
    assert "No module named" not in result.stderr


@pytest.mark.parametrize("status", ["failed", "parsed"])
def test_report_keeps_missing_pages_and_failed_anchors_visible(tmp_path: Path, status: str) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"scope": "probe", "samples": [
        {"id": "one", "anchors": ["15.12"], "figure_semantics_review": "manual_required"},
        {"id": "two", "anchors": ["2.16"], "figure_semantics_review": "not_applicable"},
    ]}))
    summary = {"engine": "fake", "protocol_sha256": "old-reference", "initialization_seconds": 0,
               "samples": [{"id": "one", "status": status, "seconds": 1}]}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    text = b'15.21<img src="15.12">'
    (tmp_path / "one.md").write_bytes(text)
    result = report.summarize(protocol, tmp_path)
    assert result["expected_pages"] == 2 and result["recorded_pages"] == 1
    assert result["expected_anchors"] == 2 and result["anchor_hits"] == 0
    assert result["reference_changed"] is True
    assert result["samples"][0]["structure_review"] == "not_evaluated"
    if status == "parsed":
        assert result["samples"][0]["artifact_sha256"] == hashlib.sha256(text).hexdigest()
