"""页证据绑定的准入与上下文接口检查；全为合成夹具，不调用模型。"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tender_research.llm_fulltext import prepare, validate_bound, visual_facts
from tender_research.rule_evidence import Document


class CaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        docs = [Document("FILE-001", Path("fixture-t.pdf"), "fixture-t.pdf", "a"*64, (), ("招标原文",), ()),
                Document("FILE-002", Path("fixture-b.pdf"), "fixture-b.pdf", "b"*64, (), ("正文原文",), (), (1,))]
        self.messages, self.evidence = prepare(docs)
        self.review = {"status": "experimental_candidate", "target": "technical_total", "maximum": 5,
                       "estimated_value": 4.1, "rubric": {"official_subweights_available": False},
                       "rule_evidence_ids": ["FILE-001-P001"],
                       "topic_reviews": [{"topic_id": i, "assessment": "夹具", "evidence_ids": ["FILE-002-P001"], "limitations": "夹具"} for i in range(1, 9)],
                       "findings": [], "reason": "夹具", "limitations": ["合成夹具"]}

    def test_full_source_binding_without_model_quote(self):
        result = validate_bound(self.review, self.evidence)
        self.assertTrue(result["automatic_gate_passed"])
        self.assertEqual(result["bound_sections"][1]["bound_evidence"][0]["quote"], "正文原文")
        self.assertIn("not_implied", result["semantic_review"])

    def test_invented_page_rejected(self):
        self.review["topic_reviews"][0]["evidence_ids"] = ["FILE-002-P999"]
        self.assertFalse(validate_bound(self.review, self.evidence)["automatic_gate_passed"])

    def test_duplicate_source_id_rejected(self):
        self.review["rule_evidence_ids"] *= 2
        self.assertFalse(validate_bound(self.review, self.evidence)["automatic_gate_passed"])

    def test_body_required_for_every_topic(self):
        self.review["topic_reviews"][0]["evidence_ids"] = ["FILE-001-P001"]
        self.assertFalse(validate_bound(self.review, self.evidence)["automatic_gate_passed"])

    def test_missing_topic_cannot_pass_coverage(self):
        self.review["topic_reviews"][0]["topic_id"] = 2
        self.assertFalse(validate_bound(self.review, self.evidence)["automatic_gate_passed"])

    def test_no_official_weights_and_illegal_scale(self):
        for field, value in (("maximum", 100), ("estimated_value", float("nan")), ("target", "overall_total")):
            candidate = copy.deepcopy(self.review); candidate[field] = value
            self.assertFalse(validate_bound(candidate, self.evidence)["automatic_gate_passed"])
        self.review["rubric"]["official_subweights_available"] = True
        self.assertFalse(validate_bound(self.review, self.evidence)["automatic_gate_passed"])

    def test_gap_claim_requires_requirement_source(self):
        self.review["findings"] = [{"id": "F-001", "kind": "响应缺口", "claim": "夹具", "scope_reason": "夹具",
                                    "evidence_ids": ["FILE-002-P001"], "needed_facts": [], "action": "复核"}]
        self.assertFalse(validate_bound(self.review, self.evidence)["automatic_gate_passed"])

    def test_malformed_structure_fails_without_crashing(self):
        self.assertFalse(validate_bound({"topic_reviews": None}, self.evidence)["automatic_gate_passed"])

    def test_prompt_excludes_outcome_and_contains_visual_limit(self):
        text = self.messages[1]["content"]
        self.assertNotIn("4.26", text); self.assertNotIn("4.39", text)
        documents = json.loads(text.split("\n资料：\n", 1)[1])
        self.assertTrue(documents[1]["pages"][0]["image_present"])
        self.assertIn("图表", text)

    def make_visual_fixture(self, root, source_version="b"*64):
        folder=root/'results'; folder.mkdir(parents=True)
        image={"evidence_id":"FILE-002-P001-VISUAL","file_id":"FILE-002","pdf_page":1,"source_sha256":source_version,
               "rendered_png_sha256":"c"*64,"width":100,"height":100,"unexpected_score":4.26}
        probe=folder/'llm-visual-probe.json';probe.write_text(json.dumps({"images":[image]}))
        reference={"has_visible_content":True,"has_network_diagram":True,"has_gantt_chart":True,"has_site_layout":False,"planning_horizon_days":120,"unexpected_cutoff":4.39}
        review={"visual_probes":{"sha256":hashlib.sha256(probe.read_bytes()).hexdigest(),"checks":[{"evidence_id":image['evidence_id'],"core_probe_passed":True,"reference":reference}]}}
        (folder/'review-fulltext.json').write_text(json.dumps(review))

    def test_visual_fact_transfer_uses_whitelist(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.make_visual_fixture(root)
            doc=Document('FILE-002',Path('fixture.pdf'),'fixture.pdf','b'*64,(),('夹具',),())
            text=json.dumps(visual_facts(root/'results',[doc]))
            self.assertNotIn('4.26',text);self.assertNotIn('4.39',text)
            self.assertIn('planning_horizon_days',text)

    def test_visual_fact_cannot_attach_to_different_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.make_visual_fixture(root,source_version='d'*64)
            doc=Document('FILE-002',Path('fixture.pdf'),'fixture.pdf','b'*64,(),('夹具',),())
            with self.assertRaises(ValueError):visual_facts(root/'results',[doc])


if __name__ == "__main__":
    unittest.main()
