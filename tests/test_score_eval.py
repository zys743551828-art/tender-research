"""评分评测与校准测试；全部数据为已知真值的合成夹具，只证明程序行为，不是业务精度证据。"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from tender_research.contracts import EvalProtocol, PairSample, ProjectField, SystemPrediction
from tender_research.score_eval import run
from tender_research.score_model import PF1, apply_bundle, fit_linear_model, model_id
from tender_research.score_text import (
    body_text,
    knn_targets,
    ngram_counts,
    similarity_matrix,
)
from tender_research.scoring import (
    Context,
    Sample,
    assert_disjoint,
    audit_archive,
    build_samples,
    cluster_bootstrap_gain,
    conformal_quantile,
    connected_groups,
    feature_allowed,
    fit_predict,
    grouped_folds,
    label_admission,
    learning_curve,
    load_sources,
    match_items,
    pav,
    prediction_admission,
    probability_metrics,
    ranking_metrics,
    time_holdout,
)
from tests.score_synth import REGIONS, generate, protocol

FAST = {"cv": {"folds": 5, "repeats": 1}, "bootstrap_resamples": 200}
NO_TEXT = ["mean", "median", "scheme_mean", "raw", "linear", "isotonic", "anchor", "ridge", "ridge_anchor", "ridge_isotonic"]
GOLDEN = Path(__file__).resolve().parents[1] / "protocols/scoring-model-1.golden.json"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def record(sid="S1", project="P1", template="T1", body=None, label=None, **extra):
    row = {"sample_id": sid, "project_key": project, "template_group": template, "region": "安徽省合肥市",
           "engineering_type": "市政道路", "scoring_scheme": "综合评估法", "bidder": "企业", "own_company": True,
           "submitted_at": "2025-01-01",
           "tender_files": [{"file_id": "F1", "path": "t.pdf", "sha256": sha(project), "role": "tender"}],
           "body_file": {"file_id": "F3", "path": "b.pdf", "sha256": body or sha(sid), "text_layer": True},
           "label": {"target": "technical_total", "value": 40.0, "maximum": 50.0, "verification_status": "verified",
                     "source_kind": "official_document", "known_at": "2025-02-01"} if label is None else label,
           "eligible": True, "exclusion": None}
    row.update(extra)
    return PairSample.model_validate(row)


def prediction(rec: PairSample, estimate="30", maximum="50", status="succeeded", **extra):
    row = {"schema": "system-prediction/1", "sample_id": rec.sample_id, "status": status,
           "inputs": {"tender_sha256": [f.sha256 for f in rec.tender_files], "body_sha256": rec.body_file.sha256},
           "prediction": {"status": "candidate", "estimate": estimate, "low": None, "high": None, "maximum": maximum},
           "items": [], "body": {"pages": 10, "chars": 1000}, "usage": {"cache_hits": 1}}
    row.update(extra)
    return SystemPrediction.model_validate(row), row


def minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode()
    objects = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
               b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
               b"<</Length %d>>stream\n" % len(stream) + stream + b"\nendstream",
               b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def toy_samples(n_groups=12, per=2, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_groups):
        for k in range(per):
            q = rng.normal()
            y = float(70 + 8 * q)
            out.append(Sample(sample_id=f"S{g}-{k}", project_key=f"P{g}", template_group=f"T{g}", body_sha256=sha(f"{g}{k}"),
                              region="R", engineering_type="E", scoring_scheme="A" if g % 2 else "B", own_company=k == 0,
                              submitted_at=date(2024, 1, 1 + g), known_at=None, label_source="official_document",
                              label_verified=True, y=y, maximum=100.0, body_path=None,
                              system_status="candidate", raw_pct=y - 10 + rng.normal(0, 1), native_low_pct=None,
                              native_high_pct=None, features={"raw_pct": y - 10, "anchor_pct": y + rng.normal(0, 3),
                                                              "coverage_mean": None if k else 0.5},
                              group=g))
    return out


def toy_protocol(**overrides):
    base = json.loads((Path(__file__).resolve().parents[1] / "protocols/评测协议.example.json").read_text(encoding="utf-8"))
    base.update(overrides)
    return EvalProtocol.model_validate(base)


class IntakeHandoffTests(unittest.TestCase):
    def test_project_field_accepts_intake_output(self):
        # 入库工具：没有同场成绩写 null，公示里未公布的单个分数写 null 元素
        self.assertIsNone(ProjectField.model_validate({"field_scores": None, "shortlist_cutoff": None}).field_scores)
        field = ProjectField.model_validate({"field_scores": [52.3, 48.0, None], "shortlist_cutoff": 47.0})
        self.assertEqual(field.field_scores, [52.3, 48.0, None])


class AdmissionTests(unittest.TestCase):
    def test_only_verified_published_technical_totals(self):
        self.assertEqual(label_admission(record()), (True, "eligible"))
        composite = record(label={"target": "composite_total", "value": 80.0, "maximum": 100.0,
                                  "verification_status": "verified", "source_kind": "official_document"})
        self.assertEqual(label_admission(composite)[1], "composite_not_technical")
        pending = record(label={"target": "technical_total", "value": 40.0, "maximum": 50.0,
                                "verification_status": "pending", "source_kind": "company_ledger"})
        self.assertEqual(label_admission(pending), (True, "eligible"))
        self.assertEqual(label_admission(pending, ("official_document",))[1], "label_source_not_admitted")
        oral = record(label={"target": "technical_total", "value": 40.0, "maximum": 50.0,
                             "verification_status": "pending", "source_kind": "user_statement"})
        self.assertEqual(label_admission(oral)[1], "label_source_not_admitted")
        unpublished = record(label={"target": "technical_total", "value": None, "maximum": 50.0,
                                    "verification_status": "verified", "source_kind": "official_document"})
        self.assertEqual(label_admission(unpublished)[1], "missing_label")
        self.assertEqual(label_admission(record(label=None, eligible=False, exclusion="missing_label"))[1], "missing_label")
        self.assertEqual(label_admission(record(eligible=False, exclusion="no_text_layer"))[1], "no_text_layer")
        over = record(label={"target": "technical_total", "value": 51.0, "maximum": 50.0,
                             "verification_status": "verified", "source_kind": "official_document"})
        self.assertEqual(label_admission(over)[1], "score_out_of_range")

    def test_eligible_flag_must_match_exclusion(self):
        with self.assertRaises(ValidationError):
            record(eligible=True, exclusion="duplicate_body")

    def test_prediction_must_bind_same_inputs(self):
        rec = record()
        parsed, _ = prediction(rec)
        self.assertTrue(prediction_admission(rec, parsed)[0])
        self.assertEqual(prediction_admission(rec, None)[1], "prediction_missing")
        other, _ = prediction(record(body=sha("另一份正文")))
        self.assertEqual(prediction_admission(rec, other)[1], "prediction_input_mismatch")

    def test_failed_run_stays_in_evaluation_with_fallback(self):
        rec = record()
        failed = prediction(rec, status="failed", prediction=None)
        samples, excluded = build_samples([rec], {rec.sample_id: failed})
        self.assertEqual(excluded, [])
        self.assertEqual(samples[0].system_status, "failed")
        self.assertIsNone(samples[0].raw_pct)
        self.assertTrue(all(v is None for v in samples[0].features.values()))

    def test_format_b_numbers_are_decimal_strings(self):
        rec = record()
        with self.assertRaises(ValidationError):
            prediction(rec, estimate=30.0)
        with self.assertRaises(ValidationError):
            prediction(rec, estimate="60", maximum="50")


class GroupingTests(unittest.TestCase):
    def test_transitive_project_template_body_groups(self):
        a = record("A", project="P1", template="T1")
        b = record("B", project="P1", template="T2")
        c = record("C", project="P2", template="T2")
        d = record("D", project="P3", template="T3", body=sha("A"))
        e = record("E", project="P4", template="T4")
        groups = connected_groups([a, b, c, d, e])
        self.assertEqual(len({groups[x] for x in "ABCD"}), 1)
        self.assertNotEqual(groups["A"], groups["E"])

    def test_same_template_never_crosses_folds(self):
        samples = toy_samples(n_groups=30)
        for i, s in enumerate(samples):
            s.template_group = f"T{i % 7}"
        records = [record(s.sample_id, project=s.project_key, template=s.template_group, body=s.body_sha256) for s in samples]
        groups = connected_groups(records)
        for s in samples:
            s.group = groups[s.sample_id]
        g = np.array([s.group for s in samples])
        for seed in range(5):
            folds = grouped_folds(g, 5, seed)
            for f in np.unique(folds):
                assert_disjoint(samples, np.flatnonzero(folds != f), np.flatnonzero(folds == f))
                templates_in = {samples[i].template_group for i in np.flatnonzero(folds == f)}
                templates_out = {samples[i].template_group for i in np.flatnonzero(folds != f)}
                self.assertFalse(templates_in & templates_out)

    def test_tampered_split_rejected(self):
        samples = toy_samples()
        with self.assertRaises(ValueError):
            assert_disjoint(samples, np.array([0, 2, 4]), np.array([1, 3]))

    def test_folds_balanced_and_deterministic(self):
        g = np.repeat(np.arange(40), 2)
        a, b = grouped_folds(g, 5, 11), grouped_folds(g, 5, 11)
        self.assertTrue((a == b).all())
        self.assertLessEqual(np.ptp(np.bincount(a)), 2)


class TimeHoldoutTests(unittest.TestCase):
    def test_latest_groups_test_and_crossing_group_quarantined(self):
        samples = toy_samples(n_groups=10, per=1)
        samples[0].submitted_at = date(2024, 1, 1)
        late_member = toy_samples(n_groups=1, per=1)[0]
        late_member.sample_id, late_member.group, late_member.submitted_at = "late", 0, date(2024, 3, 1)
        samples.append(late_member)
        split = time_holdout(samples, 0.2)
        cutoff = date.fromisoformat(split["cutoff"])
        self.assertTrue(all(samples[i].submitted_at >= cutoff for i in split["test"]))
        self.assertTrue(all(samples[i].submitted_at < cutoff for i in split["train"]))
        crossed = {e["sample_id"] for e in split["excluded"] if e["reason"] == "group_crosses_time_boundary"}
        self.assertTrue({"S0-0", "late"} <= crossed or {"S0-0", "late"} <= {samples[i].sample_id for i in split["test"]})

    def test_label_known_after_cutoff_not_used_for_training(self):
        samples = toy_samples(n_groups=10, per=1)
        samples[1].known_at = date(2030, 1, 1)
        split = time_holdout(samples, 0.2)
        self.assertIn({"sample_id": samples[1].sample_id, "reason": "label_not_known_at_cutoff"}, split["excluded"])
        self.assertNotIn(1, split["train"])


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.samples = toy_samples()
        self.ctx = Context(self.samples, np.array([s.y for s in self.samples]), toy_protocol(), np.full((24, 24), np.nan))

    def test_test_labels_never_influence_predictions(self):
        tr, te = np.arange(16), np.arange(16, 24)
        before = {m: fit_predict(m, self.ctx, tr, te, 1)[0] for m in NO_TEXT}
        self.ctx.y[te] = 0.0
        for m in NO_TEXT:
            np.testing.assert_array_equal(before[m], fit_predict(m, self.ctx, tr, te, 1)[0])

    def test_linear_recovers_known_bias(self):
        tr = np.arange(24)
        model = fit_predict("linear", self.ctx, tr, tr, 1)[1]
        slope = model["coefficients"]["raw_pct"] / model["standardize"]["scale"]["raw_pct"]
        at_60 = model["intercept"] + slope * (60 - model["standardize"]["mean"]["raw_pct"])
        self.assertAlmostEqual(slope, 1.0, delta=0.1)
        self.assertAlmostEqual(at_60, 70.0, delta=1.5)

    def test_only_pf1_features_enter_models(self):
        self.assertTrue(feature_allowed("raw_pct", ["*"], []))
        self.assertFalse(feature_allowed("usage.cache_hits", ["*"], []))
        self.assertFalse(feature_allowed("official_value", ["*"], []))
        self.assertFalse(feature_allowed("raw_pct", ["*"], ["raw_*"]))
        self.assertEqual(self.ctx.columns["ridge"], [n for n in PF1 if not n.startswith("anchor_")])
        self.assertIn("anchor_pct", self.ctx.columns["ridge_anchor"])

    def test_pf1_artifact_preferred_and_fallback_derived(self):
        rec = record()
        with_extra = prediction(rec, artifact_extra={"feature_spec": "pf-1", "features": {"raw_pct": 61.0, "anchor_pct": None}})[0]
        samples, _ = build_samples([rec], {rec.sample_id: (with_extra, {})})
        self.assertEqual(samples[0].features, {"raw_pct": 61.0, "anchor_pct": None})
        derived = prediction(rec, items=[{"statement": "a", "max_points": "6", "coverage": 0.5, "grade_index": 0, "grade_count": 3},
                                         {"statement": "b", "max_points": "2", "coverage": 0.9, "grade_index": 2, "grade_count": 3}],
                             technical_full_score="10")[0]
        f = build_samples([rec], {rec.sample_id: (derived, {})})[0][0].features
        self.assertEqual(f["raw_pct"], 60.0)
        self.assertAlmostEqual(f["coverage_mean"], (0.5 * 6 + 0.9 * 2) / 8)
        self.assertEqual(f["coverage_min"], 0.5)
        self.assertAlmostEqual(f["grade_mean"], 6 / 8)
        self.assertAlmostEqual(f["max_ratio"], 0.8)
        self.assertAlmostEqual(f["body_chars_log"], np.log1p(1000))

    def test_median_impute_and_indicator_only_when_missingness_varies(self):
        x = np.array([[1.0, 5.0, np.nan], [np.nan, 5.0, np.nan], [3.0, 5.0, np.nan], [4.0, 5.0, np.nan]])
        model = fit_linear_model(["a", "b", "c"], x, np.array([1.0, 2.0, 3.0, 4.0]), [1.0], (0, 100))[1.0]
        self.assertEqual(model["features"], ["a", "a__missing"])
        self.assertEqual(model["impute"], {"a": 3.0})

    def test_exported_model_formula_matches_internal_prediction(self):
        tr = np.arange(24)
        for method in ("linear", "ridge_anchor", "ridge_isotonic", "isotonic", "anchor"):
            pred, model = fit_predict(method, self.ctx, tr, tr, 3)
            bundle = {**model, "interval": {"0.8": 5.0, "0.9": None}}
            for i in (0, 5, 17):
                self.assertAlmostEqual(apply_bundle(bundle, self.samples[i].features)[0], pred[i], places=9)
            self.assertEqual(apply_bundle(bundle, self.samples[0].features)[3:], (0.0, 100.0))

    def test_learning_curve_nested_fractions(self):
        rows = learning_curve(self.ctx, np.arange(24), ["linear", "mean"], [0.5, 1.0])
        self.assertEqual([r["fraction"] for r in rows], [0.5, 1.0])
        self.assertLess(rows[0]["train_samples_mean"], rows[1]["train_samples_mean"])

    def test_pav_monotone_and_ties(self):
        x = np.array([1, 2, 2, 3, 4, 5.0])
        y = np.array([1, 3, 1, 2, 5, 4.0])
        bx, by = pav(x, y)
        np.testing.assert_allclose(by, [1.0, 2.0, 2.0, 4.5])
        np.testing.assert_allclose(bx, [1.0, 2.0, 3.0, 4.5])

    def test_conformal_quantile_finite_sample(self):
        r = np.arange(1, 10, dtype=float)
        self.assertEqual(conformal_quantile(r, 0.8), 8.0)
        self.assertEqual(conformal_quantile(r, 0.95), float("inf"))

    def test_cluster_bootstrap_deterministic(self):
        g = np.repeat(np.arange(20), 3)
        a, b = np.full(60, 5.0), np.full(60, 3.0)
        x = cluster_bootstrap_gain(a, b, g, 300, 0.95, 4)
        self.assertEqual(x, cluster_bootstrap_gain(a, b, g, 300, 0.95, 4))
        self.assertEqual((x["gain"], x["low"], x["high"]), (2.0, 2.0, 2.0))


class RankingAndProbabilityTests(unittest.TestCase):
    def test_within_project_ranking(self):
        samples = toy_samples(n_groups=6, per=3)
        idx = np.arange(len(samples))
        y = np.array([s.y for s in samples])
        perfect = ranking_metrics(samples, idx, y, y + 1, 5)
        self.assertEqual((perfect["spearman_mean"], perfect["pairwise_accuracy"], perfect["top1_hit_rate"]), (1.0, 1.0, 1.0))
        reversed_ = ranking_metrics(samples, idx, y, -y, 5)
        self.assertEqual((reversed_["pairwise_accuracy"], reversed_["top1_hit_rate"]), (0.0, 0.0))
        constant = ranking_metrics(samples, idx, y, np.zeros(len(y)), 5)
        self.assertEqual(constant["pairwise_accuracy"], 0.5)
        self.assertEqual(ranking_metrics(samples[:8], idx[:8], y[:8], y[:8], 5)["status"], "not_evaluable")

    def test_probability_metrics(self):
        prob = np.array([0.9] * 15 + [0.1] * 15)
        outcome = np.array([1] * 15 + [0] * 15)
        m = probability_metrics(prob, outcome, 20)
        self.assertEqual(m["auc"], 1.0)
        self.assertLess(m["brier"], m["base_rate_brier"])
        self.assertEqual(probability_metrics(prob[:10], outcome[:10], 20)["status"], "not_evaluable")

    def test_item_matching_one_to_one(self):
        system = [{"statement": "施工方案与技术措施"}, {"statement": "质量管理体系与措施"}, {"statement": "无关条款"}]
        labels = [{"item": "施工方案与技术措施（评审）", "value": 4.0, "maximum": 5.0, "evaluator": "1"},
                  {"item": "施工方案与技术措施（评审）", "value": 3.0, "maximum": 5.0, "evaluator": "2"},
                  {"item": "质量管理体系及措施", "value": 2.0, "maximum": 5.0, "evaluator": "1"}]
        matches, n = match_items(system, labels, 0.5)
        self.assertEqual(n, 2)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][1]["values"], [80.0, 60.0])


class TextTests(unittest.TestCase):
    def test_hashing_deterministic_and_similarity(self):
        a = ngram_counts("施工组织设计 质量保证措施", (2, 3), 12)
        b = ngram_counts("施工组织设计质量保证措施", (2, 3), 12)
        np.testing.assert_array_equal(a[0], b[0])
        sim = similarity_matrix([a, b, ngram_counts("完全不同的内容文字", (2, 3), 12), None], 12)
        self.assertAlmostEqual(sim[0, 1], 1.0)
        self.assertLess(sim[0, 2], 0.5)
        self.assertTrue(np.isnan(sim[3]).all())

    def test_knn_uses_only_pool_and_excludes_same_group(self):
        sim = np.array([[1, .9, .8, .1], [.9, 1, .2, .3], [.8, .2, 1, .4], [.1, .3, .4, 1]], dtype=float)
        y = np.array([10.0, 99.0, 30.0, 50.0])
        groups = np.array([0, 0, 1, 2])
        out = knn_targets(sim, y, np.array([0]), np.array([1, 2, 3]), groups, 1)
        self.assertEqual(out[0], 30.0)

    def test_body_text_verified_and_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "body.pdf"
            pdf.write_bytes(minimal_pdf("construction method statement"))
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            self.assertEqual(body_text(pdf, "0" * 64, Path(tmp) / "cache")[1], "body_sha256_mismatch")
            text, state = body_text(pdf, digest, Path(tmp) / "cache")
            self.assertEqual(state, "extracted")
            self.assertIn("construction", text)
            self.assertEqual(body_text(pdf, digest, Path(tmp) / "cache"), (text, "cached"))


class ArchiveAuditTests(unittest.TestCase):
    def test_admissible_and_verified_official_results_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "CASE-001_X"
            case.mkdir()
            body = "b" * 64
            manifest = {"case_id": "CASE-001", "tender_id": "ZB1", "file_records": [{"file_id": "FILE-002", "kind": "技术标", "sha256": body}],
                        "assessment": {"max_scores": {"technical": 60}, "official_results": [
                            {"result_id": "R1", "target": "technical_total", "value": 52.3, "maximum": 60,
                             "verification_status": "verified", "source_kind": "official_document", "submission_file_id": "FILE-002", "submission_sha256": body},
                            {"result_id": "R2", "target": "technical_total", "value": 50.0, "maximum": 60,
                             "verification_status": "pending", "source_kind": "company_ledger", "submission_file_id": "FILE-002",
                             "submission_sha256": body},
                            {"result_id": "R3", "target": "technical_total", "value": 50.0, "maximum": 60,
                             "verification_status": "pending", "source_kind": "user_statement", "submission_file_id": "FILE-002", "submission_sha256": body}]}}
            (case / "资料清单.json").write_text(json.dumps(manifest, ensure_ascii=False))
            audit = audit_archive(Path(tmp), Path(tmp))
            self.assertEqual((audit["eligible_real_score_pairs"], audit["verified_official_pairs"]), (2, 1))
            self.assertEqual(audit["status"], "independent_official_scoring_not_evaluable")
            manifest["assessment"]["official_results"][0]["submission_sha256"] = "c" * 64
            (case / "资料清单.json").write_text(json.dumps(manifest, ensure_ascii=False))
            with self.assertRaises(ValueError):
                audit_archive(Path(tmp), Path(tmp))


class SyntheticSignalEvalTests(unittest.TestCase):
    """官方百分制 = f(系统原始估计) + 噪声，系统估计带已知偏差 -12：校准必须优于基线，区间覆盖接近名义。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.results = root / "results"
        cls.data = generate(root / "data", cache=cls.results / "score-eval-text-cache")
        cls.report = run(cls.data["pairs"], cls.data["predictions"], protocol(root, target_mae=6.0), "signal",
                         results=cls.results)
        cls.out = cls.results / "score-eval-signal"
        cls.features = {p.stem: json.loads(p.read_text(encoding="utf-8")).get("artifact_extra", {}).get("features", {})
                        for p in cls.data["predictions"].glob("*.json")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_scale_and_split_counts(self):
        a = self.report["admission"]
        self.assertGreaterEqual(a["records"], 250)
        self.assertGreaterEqual(a["independent_projects"], 100)
        self.assertIn("composite_not_technical", a["excluded_by_reason"])
        self.assertIn("label_source_not_admitted", a["excluded_by_reason"])
        self.assertEqual(set(a["label_mix"]), {"official_document/verified", "company_ledger/pending"})

    def test_verified_official_subset_reported(self):
        vs = self.report["verified_official_subset"]
        self.assertEqual(vs["status"], "evaluated")
        self.assertLess(vs["n"], self.report["admission"]["admitted"])
        self.assertEqual(vs["selected_method"], self.report["verdict"]["selected_method"])
        self.assertIn("已核验官方标签子集", (self.out / "report.md").read_text(encoding="utf-8"))
        bundle = json.loads((self.out / "calibration_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(bundle["evaluation"]["label_sources"], ["official_document", "company_ledger"])

    def test_calibration_beats_baselines_and_verdict_passes(self):
        m, v = self.report["metrics"], self.report["verdict"]
        best_baseline = min(m[b]["mae"] for b in ("mean", "median", "scheme_mean"))
        self.assertLess(m["linear"]["mae"], best_baseline - 2)
        self.assertLess(m["raw"]["bias"], -9)
        self.assertLess(abs(m["linear"]["bias"]), 1)
        self.assertEqual(v["status"], "passed")
        self.assertNotIn(v["selected_method"], ("mean", "median", "scheme_mean"))
        self.assertGreater(v["gain"]["low"], 0)

    def test_conformal_coverage_near_nominal(self):
        for method in ("linear", "ridge", "isotonic"):
            for level, x in self.report["intervals"][method].items():
                self.assertLess(abs(x["coverage"] - float(level)), 0.06, (method, level))

    def test_within_project_ranking_and_extras(self):
        r = self.report["ranking"]["linear"]
        self.assertEqual(r["status"], "evaluated")
        self.assertGreater(r["pairwise_accuracy"], 0.75)
        self.assertEqual(self.report["shortlist"]["linear"]["status"], "evaluated")
        self.assertGreater(self.report["shortlist"]["linear"]["auc"], 0.75)
        self.assertEqual(self.report["relative_target"]["status"], "evaluated")
        items = self.report["items"]
        self.assertEqual(items["status"], "evaluated")
        self.assertGreater(items["match_rate"], 0.8)
        self.assertLess(items["bias"], -5)

    def test_learning_curve_and_text_arms_not_selected(self):
        curve = self.report["learning_curve"]
        self.assertEqual([c["fraction"] for c in curve], [0.25, 0.5, 0.75, 1.0])
        selected = self.report["verdict"]["selected_method"]
        self.assertGreater(curve[0]["mae"][selected], curve[-1]["mae"][selected] - 0.5)
        self.assertNotIn(selected, ("knn", "ridge_knn"))
        self.assertIn("anchor", self.report["metrics"])
        self.assertLess(self.report["metrics"]["anchor"]["mae"], self.report["metrics"]["mean"]["mae"])

    def test_scoring_model_bundle_contract(self):
        self.assertTrue((self.out / "report.md").read_text(encoding="utf-8").startswith("# 技术分评测"))
        bundle = json.loads((self.out / "calibration_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual((bundle["schema"], bundle["feature_spec"], bundle["evaluation"]["verdict"]), ("scoring-model/1", "pf-1", "passed"))
        self.assertEqual(bundle["model_id"], model_id(bundle))
        self.assertLess(bundle["interval"]["0.8"], bundle["interval"]["0.9"])
        self.assertTrue({c.removesuffix("__missing") for c in bundle["features"]} <= set(PF1))
        self.assertEqual(bundle["scope"]["regions"], sorted(set(REGIONS)))  # 训练数据里出现过的完整地区
        self.assertEqual(bundle["evaluation"]["n"], self.report["admission"]["admitted"])
        self.assertEqual(len(self.report["per_sample"]), self.report["admission"]["admitted"])
        refit = self.report["per_sample"][:3]
        for row in refit:
            pct, lo80, hi80, lo90, hi90 = apply_bundle(bundle, self.features[row["sample_id"]])
            self.assertTrue(0 <= lo90 <= lo80 <= pct <= hi80 <= hi90 <= 100)
        with self.assertRaises(FileExistsError):
            run(self.data["pairs"], self.data["predictions"], protocol(Path(self.tmp.name)), "signal", results=self.results)


class SyntheticVerdictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_signal_not_passed_and_text_methods_dropped(self):
        data = generate(self.root / "data", n_projects=80, signal=False, with_text=False)
        report = run(data["pairs"], data["predictions"], protocol(self.root, target_mae=6.0, **FAST), "nosignal",
                     results=self.root / "results")
        self.assertEqual(report["verdict"]["status"], "not_passed")
        self.assertNotIn("knn", report["methods_run"])
        self.assertTrue(report["notes"])

    def test_tiny_set_not_evaluable_without_accuracy(self):
        data = generate(self.root / "data", n_projects=6)
        report = run(data["pairs"], data["predictions"], protocol(self.root, **FAST), "tiny", results=self.root / "results")
        self.assertEqual(report["verdict"]["status"], "not_evaluable")
        self.assertNotIn("metrics", report)
        md = (self.root / "results/score-eval-tiny/report.md").read_text(encoding="utf-8")
        self.assertIn("不可评价", md)
        self.assertNotIn("方法对比", md)

    def test_deterministic_with_fixed_seed(self):
        data = generate(self.root / "data", n_projects=40, cache=self.root / "results/score-eval-text-cache")
        path = protocol(self.root, methods=["mean", "scheme_mean", "linear", "ridge", "knn"], **FAST)
        a = run(data["pairs"], data["predictions"], path, "a", results=self.root / "results")
        b = run(data["pairs"], data["predictions"], path, "b", results=self.root / "results")
        def strip(r):
            return {k: v for k, v in r.items() if k not in ("experiment_id", "generated_at", "elapsed_seconds", "calibration_bundle")}
        self.assertEqual(json.dumps(strip(a), sort_keys=True), json.dumps(strip(b), sort_keys=True))
        bundles = [json.loads((self.root / f"results/score-eval-{n}/calibration_bundle.json").read_text(encoding="utf-8"))
                   for n in ("a", "b")]
        self.assertEqual(*[{k: v for k, v in x.items() if k not in ("created_at", "model_id")} for x in bundles])

    def test_holdout_runs_once_per_protocol(self):
        data = generate(self.root / "data", n_projects=60, with_text=False)
        path = protocol(self.root, methods=["mean", "median", "linear"], **FAST)
        results = self.root / "results"
        report = run(data["pairs"], data["predictions"], path, "final", mode="holdout", results=results)
        self.assertIn(report["verdict"]["status"], ("passed", "not_passed"))
        cutoff = report["holdout"]["cutoff"]
        pairs = {s["sample_id"]: s for s in json.loads(data["pairs"].read_text(encoding="utf-8"))["samples"]}
        self.assertTrue(all(pairs[r["sample_id"]]["submitted_at"] >= cutoff for r in report["per_sample"]))
        with self.assertRaises(RuntimeError):
            run(data["pairs"], data["predictions"], path, "final-again", mode="holdout", results=results)
        run(data["pairs"], data["predictions"], path, "final-again", mode="holdout", results=results, force=True)
        registry = json.loads((results / "score-eval-holdout-registry.json").read_text(encoding="utf-8"))
        self.assertEqual([e["forced"] for e in next(iter(registry.values()))], [False, True])

    def test_merged_sources_dedup_by_body(self):
        first = generate(self.root / "a", n_projects=20, seed=1)
        second = generate(self.root / "b", n_projects=20, seed=2, prefix="EXPORT")
        merged = load_sources([first["pairs"], second["pairs"], first["pairs"]],
                              [first["predictions"], second["predictions"]])
        n_first = len(json.loads(first["pairs"].read_text(encoding="utf-8"))["samples"])
        n_second = len(json.loads(second["pairs"].read_text(encoding="utf-8"))["samples"])
        self.assertEqual(len(merged["records"]), n_first + n_second)
        self.assertEqual(sum(e["reason"] == "duplicate_source" for e in merged["invalid"]), n_first)
        report = run([first["pairs"], second["pairs"]], [first["predictions"], second["predictions"]],
                     protocol(self.root, methods=["mean", "linear"], **FAST), "merged", results=self.root / "results")
        self.assertEqual(len(report["data"]["pairs_sha256"]), 2)

    def test_not_evaluable_bundle_has_no_model(self):
        data = generate(self.root / "data", n_projects=6)
        run(data["pairs"], data["predictions"], protocol(self.root, **FAST), "tiny2", results=self.root / "results")
        bundle = json.loads((self.root / "results/score-eval-tiny2/calibration_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual((bundle["method"], bundle["evaluation"]["verdict"]), (None, "not_evaluable"))
        self.assertEqual(bundle["model_id"], model_id(bundle))

    def test_protocol_requires_baseline_and_system_method(self):
        with self.assertRaises(ValidationError):
            toy_protocol(methods=["mean", "median"])
        with self.assertRaises(ValidationError):
            toy_protocol(selection="mean")
        with self.assertRaises(ValidationError):
            toy_protocol(selection="knn")


class ScoringModelContractTests(unittest.TestCase):
    """protocols/scoring-model-1.golden.json：研究侧与产品后端共用的参考向量。"""

    def test_golden_vectors(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        bundle = golden["bundle"]
        self.assertEqual(bundle["model_id"], model_id(bundle))
        for vector in golden["vectors"]:
            got = apply_bundle(bundle, vector["features"])
            for key, value in zip(("pct", "lo80", "hi80", "lo90", "hi90"), got):
                self.assertAlmostEqual(value, vector["expected"][key], places=9)

    def test_golden_first_vector_by_hand(self):
        bundle = json.loads(GOLDEN.read_text(encoding="utf-8"))["bundle"]
        linear = 72 + 5 * (64 - 60) / 8 - 0.5 * (0 - 0.1) / 0.3 + 3 * (75 - 70) / 10 + 2 * (0.7 - 0.6) / 1
        expected = 74 + (linear - 75) * (90 - 74) / (85 - 75)
        self.assertAlmostEqual(apply_bundle(bundle, {"raw_pct": 64.0, "anchor_pct": 75.0, "coverage_mean": 0.7})[0], expected)

    def test_model_id_changes_with_parameters(self):
        bundle = json.loads(GOLDEN.read_text(encoding="utf-8"))["bundle"]
        changed = {**bundle, "intercept": 72.5}
        self.assertNotEqual(model_id(changed), bundle["model_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
