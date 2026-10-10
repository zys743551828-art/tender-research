"""工程夹具测试；所有分数与发布包均为合成数据，不是业务效果证据。"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from tender_research.contracts import Evidence, OfficialLabel, Prediction
from tender_research.diagnostics import (
    extract_limited_rules,
    pipe_heading_conflicts,
    resolve_interval,
    revision_gate,
)
from tender_research.lifecycle import GateError, Store
from tender_research.outcomes import classify_stage, score_cell
from tender_research.rule_evidence import (
    Document,
    EvidenceIndex,
    InputError,
    check_evidence,
    load_documents,
)


def dt(value):
    return datetime.fromisoformat(value + "T12:00:00+08:00")


def official_label(value=80.0, maximum=100.0):
    return OfficialLabel(target="technical_total", value=value, maximum=maximum, source_file_id="SYNTHETIC-LABEL",
                         source_sha256="a" * 64, source_pdf_page=1, submission_id="SUB1", aggregation="synthetic fixture",
                         confirmed=True, known_at=dt("2025-01-01"), revision="fixture-v1")


def document(text, fid="F1", project_ids=("2026BFFGZ51067",)):
    return Document(fid, Path("synthetic-fixture.pdf"), "synthetic-fixture.pdf", "a" * 64, (), (text,), project_ids)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.docs = [document("技术评审通过后进入报价。")]
        self.hit = EvidenceIndex(self.docs).search("技术评审")[0]

    def test_original_excerpt_valid(self):
        self.assertTrue(check_evidence(self.hit, self.docs)[0])
        Evidence.model_validate(self.hit)

    def test_tampered_quote_rejected(self):
        self.hit["quote"] = "不存在的扣分原因"
        self.assertFalse(check_evidence(self.hit, self.docs)[0])

    def test_tampered_identity_rejected(self):
        self.hit["source_sha256"] = "b" * 64
        self.assertFalse(check_evidence(self.hit, self.docs)[0])

    def test_wrong_page_rejected(self):
        self.hit["pdf_page"] = 2
        self.assertFalse(check_evidence(self.hit, self.docs)[0])

    def test_boolean_position_rejected(self):
        self.hit["pdf_page"] = True
        self.assertFalse(check_evidence(self.hit, self.docs)[0])

    def test_ambiguous_project_isolated(self):
        d = document("技术评审", project_ids=("2026BFFGZ51067", "2026BFJGZ50917"))
        self.assertEqual(EvidenceIndex([d]).search("技术评审", project_id="2026BFFGZ51067"), [])

    def test_unknown_project_no_fallback(self):
        self.assertEqual(EvidenceIndex(self.docs).search("技术评审", project_id="MISSING"), [])

    def test_empty_query_and_file_scope(self):
        index = EvidenceIndex(self.docs)
        self.assertEqual(index.search(""), [])
        self.assertEqual(index.search("技术评审", file_ids=[]), [])

    def test_manifest_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps({"file_records": [{"file_id": "F", "path": "../outside.pdf"}]}))
            with self.assertRaises(InputError):
                load_documents(path)

    def test_source_bytes_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "doc.pdf").write_bytes(b"not the claimed PDF")
            (path / "manifest.json").write_text(json.dumps({"file_records": [{"file_id": "F", "path": "doc.pdf", "sha256": "a" * 64, "bytes": 0}]}))
            with self.assertRaises(InputError):
                load_documents(path / "manifest.json")


class RuleDiagnosticTests(unittest.TestCase):
    def test_unchecked_ai_option_not_selected(self):
        rules = extract_limited_rules([document("□不执行AI“类人”辅助评标")])
        self.assertEqual(rules, [])

    def test_recommendation_not_denial(self):
        rules = extract_limited_rules([document("建议编制要求：施工组织设计的篇幅不超过50页。")])
        self.assertEqual(rules[0]["category"], "recommendation")

    def test_patch_requires_matching_original(self):
        base = extract_limited_rules([document("良好得70分＜F＜80分")])
        patch = extract_limited_rules([document("良好得70分＜F＜85分，现修改为良好得70分＜F＜90分", fid="F2")])
        self.assertEqual(resolve_interval(base + patch, "2026BFFGZ51067")["status"], "needs_review")

    def test_ambiguous_multiple_patches_require_review(self):
        base = extract_limited_rules([document("良好得70分＜F＜80分")])
        p = extract_limited_rules([document("良好得70分＜F＜80分，现修改为良好得70分＜F＜90分", fid="F2")])
        self.assertEqual(resolve_interval(base + p + p, "2026BFFGZ51067")["status"], "needs_review")

    def test_different_material_sections_not_false_conflict(self):
        d = document("一、球墨铸铁管施工\n核对球墨铸铁管。\n二、夹砂管施工\n核对 FRPM 管。")
        self.assertEqual(pipe_heading_conflicts([d]), [])

    def test_contradiction_then_recheck_corrected_fixture(self):
        original = document("一、球墨铸铁管施工\n核对 FRPM 管。\n二、PE管施工\n核对PE管。")
        findings = pipe_heading_conflicts([original])
        self.assertEqual(len(findings), 1)
        self.assertEqual(revision_gate(findings[0], None, False)["status"], "awaiting_engineering_basis")
        reviewed_fixture = document("一、夹砂管施工\n核对 FRPM 管。\n二、PE管施工\n核对PE管。")
        self.assertEqual(pipe_heading_conflicts([reviewed_fixture]), [])


class PredictionContractTests(unittest.TestCase):
    def test_unavailable_prediction_cannot_show_score_or_probability(self):
        with self.assertRaises(ValidationError):
            Prediction(target="technical_total", status="unavailable", bundle_id="fixture", input_snapshot="fixture",
                       estimated_value=80.0, maximum=100.0, validation_status="not_validated", applicability="fixture")


class OutcomeTests(unittest.TestCase):
    def test_slash_unknown_but_explicit_zero_is_zero(self):
        self.assertEqual(score_cell("/",100.0)["status"],"not_disclosed")
        self.assertIsNone(score_cell("/",100.0)["value"])
        self.assertEqual(score_cell("0",100.0)["value"],0.0)

    def test_invalid_numbers_need_review(self):
        for text in ("NaN","Infinity","-1","101","未入围"):
            self.assertEqual(score_cell(text,100.0)["status"],"needs_review")

    def test_wrapped_candidate_title_and_final_title_differ(self):
        self.assertEqual(classify_stage("很长的项目名称\n项目中标候选人公示（评定分离）\n一、相关情况"),"candidate_publication")
        self.assertEqual(classify_stage("某项目中标结果公告\n正文提及中标候选人"),"final_award_publication_requires_identity_review")

    def test_candidate_word_in_body_not_award(self):
        self.assertEqual(classify_stage("某项目中标结果公告\n一、相关情况\n此前中标候选人"),"final_award_publication_requires_identity_review")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/"store"
        self.store = Store(self.root)
        self.f1 = self.store.file_version("PROJECT-FIXTURE", "submitted_body", b"synthetic draft-v1", "2025-01-01T12:00:00+08:00")
        self.snap = self.store.snapshot("PROJECT-FIXTURE", [self.f1], "2025-01-02T12:00:00+08:00")
        exp = self.store.freeze("experiment", {"provenance":"synthetic_test_fixture"})
        self.bundle = self.store.bundle(dict.fromkeys(("parser", "rules", "features", "strategy", "code", "dependencies"), "fixture-v1"), exp)
        self.valid = self.store.freeze("validation", {"bundle":self.bundle,"status":"passed","scope":"synthetic_contract_fixture"})
        self.store.release("fixture", self.bundle, self.valid, "release_owner")

    def tearDown(self):
        self.store.close(); self.tmp.cleanup()

    def test_new_file_preserves_old_snapshot(self):
        f2 = self.store.file_version("PROJECT-FIXTURE", "submitted_body", b"synthetic draft-v2", "2025-02-01T12:00:00+08:00", self.f1)
        snap2 = self.store.snapshot("PROJECT-FIXTURE", [f2], "2025-02-02T12:00:00+08:00")
        self.assertNotEqual(snap2,self.snap)
        self.assertEqual(self.store.get(self.snap)["files"],[self.f1])

    def test_objects_immutable_even_sql_update(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("UPDATE objects SET payload='{}' WHERE id=?", (self.snap,))

    def test_future_file_and_result_input_rejected(self):
        with self.assertRaises(GateError):
            self.store.snapshot("PROJECT-FIXTURE", [self.f1], "2024-01-02T12:00:00+08:00")
        outcome = self.store.file_version("PROJECT-FIXTURE","official_result",b"synthetic score", "2025-01-01T12:00:00+08:00")
        with self.assertRaises(GateError):
            self.store.snapshot("PROJECT-FIXTURE",[outcome],"2025-01-02T12:00:00+08:00")

    def test_cross_project_input_rejected(self):
        with self.assertRaises(GateError):
            self.store.snapshot("OTHER",[self.f1],"2025-01-02T12:00:00+08:00")

    def test_idempotence_and_conflicting_payload(self):
        job = self.store.enqueue("fixture",self.snap,"K",{"task":"fixture"})
        self.assertEqual(job,self.store.enqueue("fixture",self.snap,"K",{"task":"fixture"}))
        with self.assertRaises(GateError):
            self.store.enqueue("fixture",self.snap,"K",{"task":"changed"})

    def test_lease_expiry_recovery_and_late_worker_rejected(self):
        job = self.store.enqueue("fixture",self.snap,"K",{})
        first = self.store.claim(0,10)
        self.assertIsNone(self.store.claim(5,10))
        second = self.store.claim(11,10)
        self.assertEqual(second["attempt"],2)
        with self.assertRaises(GateError):
            self.store.finish(job,first["token"],{"stale":True},12)
        result = self.store.finish(job,second["token"],{"valid":True},12)
        self.assertTrue(self.store.get(result)["result"]["valid"])

    def test_expired_worker_without_reclaim_rejected(self):
        job = self.store.enqueue("fixture",self.snap,"K",{})
        first = self.store.claim(0,10)
        with self.assertRaises(GateError):
            self.store.finish(job,first["token"],{},10)

    def test_concurrent_workers_claim_once(self):
        self.store.enqueue("fixture",self.snap,"K",{})
        barrier = threading.Barrier(2)
        def claim(_):
            other = Store(self.root)
            try:
                barrier.wait(timeout=5)
                return other.claim(1,10)
            finally:
                other.close()
        with ThreadPoolExecutor(2) as pool:
            claims = list(pool.map(claim, range(2)))
        self.assertEqual(sum(c is not None for c in claims),1)

    def test_bundle_switch_and_whole_bundle_rollback(self):
        old = self.store.enqueue("fixture",self.snap,"OLD",{})
        exp = self.store.freeze("experiment",{"provenance":"synthetic_test_fixture","v":2})
        new = self.store.bundle(dict.fromkeys(("parser", "rules", "features", "strategy", "code", "dependencies"), "fixture-v2"),exp)
        valid2 = self.store.freeze("validation",{"bundle":new,"status":"passed","scope":"synthetic_contract_fixture"})
        self.store.release("fixture",new,valid2,"release_owner")
        newer = self.store.enqueue("fixture",self.snap,"NEW",{})
        locked = dict(self.store.db.execute("SELECT id,bundle_id FROM jobs"))
        self.assertEqual(locked[old],self.bundle); self.assertEqual(locked[newer],new)
        self.store.release("fixture",self.bundle,self.valid,"release_owner")
        rollback = self.store.enqueue("fixture",self.snap,"ROLLBACK",{})
        self.assertEqual(dict(self.store.db.execute("SELECT id,bundle_id FROM jobs"))[rollback],self.bundle)

    def test_unvalidated_incompatible_and_production_release_rejected(self):
        failed = self.store.freeze("validation",{"bundle":self.bundle,"status":"failed","scope":"synthetic_contract_fixture"})
        for env, validation, role in [("fixture",failed,"release_owner"),("production",self.valid,"release_owner"),("fixture",self.valid,"viewer")]:
            with self.assertRaises(GateError):
                self.store.release(env,self.bundle,validation,role)
        exp = self.store.freeze("experiment",{"version":3})
        incompatible = self.store.bundle(dict.fromkeys(("parser", "rules", "features", "strategy", "code", "dependencies"), "fixture-v3"),exp,"input-v2")
        val = self.store.freeze("validation",{"bundle":incompatible,"status":"passed","scope":"synthetic_contract_fixture"})
        with self.assertRaises(GateError):
            self.store.release("fixture",incompatible,val,"release_owner")

    def test_latest_component_rejected(self):
        exp = self.store.freeze("experiment",{"fixture":True})
        components = dict.fromkeys(("parser", "rules", "features", "strategy", "code", "dependencies"), "fixture-v1");components["strategy"]="latest"
        with self.assertRaises(GateError):
            self.store.bundle(components,exp)

    def test_late_outcome_correction_preserves_frozen_dataset(self):
        label = official_label().model_dump(mode="json")
        old = self.store.outcome("SUB1",label)
        ds = self.store.freeze("dataset",{"labels":[old],"provenance":"synthetic_test_fixture"})
        label["value"] = 85.0;label["revision"]="fixture-v2"
        new = self.store.outcome("SUB1",label,old)
        self.assertNotEqual(old,new)
        self.assertEqual(self.store.get(ds)["labels"],[old])
        self.assertEqual(self.store.get(old)["label"]["value"],80.0)

    def test_backup_restore_database_and_artifacts(self):
        backup = Path(self.tmp.name)/"backup"
        self.store.backup(backup)
        restored = Store.restore(backup, Path(self.tmp.name)/"restored")
        try:
            restored.verify_restored()
            self.assertEqual(restored.get(self.snap),self.store.get(self.snap))
            self.assertEqual(restored.read_artifact(self.store.get(self.f1)["sha256"]),b"synthetic draft-v1")
        finally:
            restored.close()

    def test_corrupted_artifact_detected(self):
        sha = self.store.get(self.f1)["sha256"]
        (self.root/"artifacts"/sha).write_bytes(b"corrupted")
        with self.assertRaises(GateError):
            self.store.verify_restored()

    def test_corrupted_backup_database_rejected(self):
        backup = Path(self.tmp.name)/"backup"
        self.store.backup(backup)
        with (backup/"records.sqlite").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(GateError):
            Store.restore(backup,Path(self.tmp.name)/"restored")


if __name__ == "__main__":
    unittest.main(verbosity=2)
