"""提供方错误与凭据边界、结果口径、引用及请求隔离的回归。全部为夹具，不调用 API。"""
import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from tender_research.contracts import OfficialLabel, ReportedResult
from tender_research.deepseek_client import DeepSeekClient, NoRedirect, ProviderError, get_key
from tender_research.llm_common import case_messages, check_citation, freeze_input, validate_case, validate_rules
from tender_research.llm_rules import bind_evidence
from tender_research.llm_rules import run as run_rules_trial
from tender_research.paths import Workspace
from tender_research.rule_evidence import Document


class DeepSeekBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / '.secrets').mkdir()
        self.path = self.root / '.secrets/deepseek_api_key'
        self.fixture_key = 'sk-unit-test-credential-with-no-real-account'
        self.path.write_text(self.fixture_key)
        self.path.chmod(0o600)
        self.env = patch.dict(os.environ, {'DEEPSEEK_API_KEY': ''})
        self.env.start()
        self.client = DeepSeekClient(self.root)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def response(self, content='{}', finish='stop'):
        return {'choices': [{'message': {'content': content}, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 12, 'completion_tokens': 3}, 'model': 'deepseek-flash'}

    def test_private_credential_read(self):
        self.assertEqual(get_key(self.root), self.fixture_key)

    def test_world_readable_credential_rejected(self):
        self.path.chmod(0o644)
        with self.assertRaises(ProviderError) as caught:
            get_key(self.root)
        self.assertEqual(caught.exception.category, 'credential_permissions_too_open')

    def test_external_endpoint_rejected_without_network(self):
        with patch.object(self.client.opener, 'open') as opened, self.assertRaises(ProviderError):
            self.client.request('https://outside.test/chat/completions')
        opened.assert_not_called()

    def test_http_error_body_never_enters_exception(self):
        error = urllib.error.HTTPError('https://api.deepseek.com', 401, 'fixture', {}, io.BytesIO(self.fixture_key.encode()))
        with patch.object(self.client.opener, 'open', side_effect=error), self.assertRaises(ProviderError) as caught:
            self.client.request('/models')
        self.assertEqual(caught.exception.http_status, 401)
        self.assertNotIn(self.fixture_key, str(caught.exception))

    def test_redirect_refused(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, 'fixture', {}, 'https://outside.test'))

    def test_budget_rejected_before_network(self):
        with patch.object(self.client, 'request') as request, self.assertRaises(ProviderError):
            self.client.complete([], 'deepseek-flash', max_tokens=32769)
        request.assert_not_called()

    def test_model_content_credential_rejected(self):
        with patch.object(self.client, 'request', return_value=self.response(json.dumps({'key': self.fixture_key}))), self.assertRaises(ProviderError):
            self.client.complete([], 'deepseek-flash')

    def test_truncated_json_not_accepted(self):
        with patch.object(self.client, 'request', return_value=self.response('{}', 'length')), self.assertRaises(ProviderError) as caught:
            self.client.complete([], 'deepseek-flash')
        self.assertEqual(caught.exception.category, 'incomplete_generation')

    def test_truncation_keeps_only_numeric_usage_and_safe_reason(self):
        value = self.response(self.fixture_key, finish='length')
        value['usage']['debug'] = self.fixture_key
        value['usage']['prompt_cache_miss_tokens'] = 12
        with patch.object(self.client, 'request', return_value=value), self.assertRaises(ProviderError) as caught:
            self.client.complete([], 'deepseek-flash')
        self.assertEqual(caught.exception.telemetry['finish_reason'], 'length')
        self.assertEqual(caught.exception.telemetry['usage']['prompt_cache_miss_tokens'], 12)
        self.assertNotIn(self.fixture_key, json.dumps(caught.exception.telemetry))

    def test_larger_thinking_budget_can_reach_provider(self):
        with patch.object(self.client, 'request', return_value=self.response()) as request:
            self.client.complete([], 'deepseek-flash', max_tokens=32768, thinking=True)
        self.assertEqual(request.call_args.args[1]['max_tokens'], 32768)

    def test_json_array_not_accepted(self):
        with patch.object(self.client, 'request', return_value=self.response('[]')), self.assertRaises(ProviderError):
            self.client.complete([], 'deepseek-flash')

    def test_reasoning_and_request_headers_not_archived(self):
        response = self.response('{"answer":"fixture"}')
        response['choices'][0]['message']['reasoning_content'] = 'private fixture reasoning'
        with patch.object(self.client, 'request', return_value=response) as request:
            result = self.client.complete([], 'deepseek-flash')
        self.assertNotIn(self.fixture_key, json.dumps(result))
        self.assertNotIn('reasoning_content', json.dumps(result))
        self.assertEqual(request.call_args.args[1]['thinking'], {'type': 'disabled'})

    def test_thinking_does_not_send_ineffective_temperature(self):
        with patch.object(self.client, 'request', return_value=self.response()) as request:
            self.client.complete([], 'deepseek-flash', thinking=True)
        self.assertNotIn('temperature', request.call_args.args[1])
        self.assertEqual(request.call_args.args[1]['thinking'], {'type': 'enabled'})
        self.assertEqual(request.call_args.args[1]['reasoning_effort'], 'high')


class ScoreAndEvidenceTests(unittest.TestCase):
    def reported(self):
        return {'result_id': 'fixture-result', 'source_kind': 'user_statement', 'received_date': '2026-10-07',
                'source_description': 'synthetic fixture', 'verification_status': 'official_source_pending',
                'target': 'technical_total', 'target_interpretation': 'fixture', 'value': 4.26, 'maximum': 5.,
                'reported_shortlist_cutoff': 4.39, 'cutoff_scope_status': 'pending', 'submission_file_id': 'fixture-body',
                'submission_sha256': 'a' * 64, 'official_training_eligible': False}

    def test_user_report_cannot_be_official_label(self):
        value = ReportedResult.model_validate(self.reported())
        with self.assertRaises(ValueError):
            OfficialLabel.model_validate(value.model_dump())

    def test_user_report_cannot_self_approve_training(self):
        value = self.reported(); value['official_training_eligible'] = True
        with self.assertRaises(ValueError):
            ReportedResult.model_validate(value)

    def test_out_of_scale_cutoff_rejected(self):
        value = self.reported(); value['reported_shortlist_cutoff'] = 5.1
        with self.assertRaises(ValueError):
            ReportedResult.model_validate(value)

    def test_case_input_excludes_metadata(self):
        doc = Document('FILE-002', Path('fixture.pdf'), 'fixture.pdf', 'a' * 64, (), ('施工正文测试',), ())
        prompt = json.dumps(case_messages([doc]), ensure_ascii=False)
        self.assertNotIn('4.26', prompt); self.assertNotIn('4.39', prompt)
        self.assertIn('施工正文测试', prompt)

    def test_score_in_original_text_triggers_leakage_audit(self):
        doc = Document('FILE-002', Path('fixture.pdf'), 'fixture.pdf', 'a' * 64, (), ('最终成绩4.26',), ())
        with self.assertRaises(ValueError):
            case_messages([doc])

    def test_wrong_project_page_rejected(self):
        c = {'file_id': 'OTHER', 'pdf_page': 1, 'quote': '不执行辅助评标'}
        self.assertFalse(check_citation(c, {('PUB-008', 79): '不执行辅助评标'})['valid'])

    def test_discontinuous_quote_rejected(self):
        c = {'file_id': 'PUB', 'pdf_page': 1, 'quote': '必须复核结果'}
        self.assertFalse(check_citation(c, {('PUB', 1): '必须先由委员会复核全部结果'})['valid'])

    def test_invalid_evidence_id_rejected(self):
        output = {'task_id': 'RULE-001', 'status': 'supported', 'answer': 'fixture', 'evidence_ids': ['E99'], 'limitations': ''}
        self.assertFalse(bind_evidence(output, 'RULE-001', {'E1': {'quote': '原文'}})['contract_passed'])

    def test_duplicate_evidence_id_rejected(self):
        output = {'task_id': 'RULE-001', 'status': 'supported', 'answer': 'fixture', 'evidence_ids': ['E1', 'E1'], 'limitations': ''}
        self.assertFalse(bind_evidence(output, 'RULE-001', {'E1': {'quote': '原文'}})['contract_passed'])

    def test_evidence_binding_retains_exact_source(self):
        source = {'quote': '逐字原文', 'source_sha256': 'a' * 64, 'pdf_page': 3}
        output = {'task_id': 'RULE-001', 'status': 'supported', 'answer': 'fixture', 'evidence_ids': ['E1'], 'limitations': ''}
        bound = bind_evidence(output, 'RULE-001', {'E1': source})
        self.assertTrue(bound['contract_passed']); self.assertEqual(bound['bound_evidence'], [source])
        self.assertIn('pending', bound['semantic_review'])

    def test_duplicate_answer_ids_rejected(self):
        a = {'task_id': 'A', 'status': 'supported', 'answer': 'fixture', 'evidence': [{'file_id': 'PUB', 'pdf_page': 1, 'quote': '原始文本内容'}], 'limitations': ''}
        self.assertFalse(validate_rules({'answers': [a, a]}, ['A', 'B'], {('PUB', 1): '原始文本内容'})['contract_passed'])

    def test_candidate_with_fake_official_weights_rejected(self):
        self.assertIn('unsupported_official_subweights', validate_case({'rubric': {'official_subweights_available': True}}, {})['contract_errors'])

    def test_frozen_input_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'input.json'
            freeze_input(path, {'fixture': 1})
            with self.assertRaises(ValueError):
                freeze_input(path, {'fixture': 2})

    def test_public_rules_trial_never_sends_private_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ws = Workspace(datasets=root / 'datasets', results=root / 'results', secrets=root)
            for folder in (ws.results, ws.datasets / '评测集', ws.datasets / '公开规则资料', ws.datasets / 'CASE-001_广德_污水管网改造及新建'):
                folder.mkdir(parents=True, exist_ok=True)
            ref = {'tasks': [{'task_id': 'RULE-FIXTURE', 'query': '公开问题'}]}
            ref_path = ws.datasets / '评测集/RULE-001.json'; ref_path.write_text(json.dumps(ref))
            (ws.datasets / '公开规则资料/资料清单.json').write_text(json.dumps({'file_records': [{'file_id': 'PUB-FIXTURE'}]}))
            import hashlib
            public = Document('PUB-FIXTURE', Path('public.pdf'), 'public.pdf', 'a' * 64, (), ('公开证据测试文本',), ())
            hit = {'file_id': public.file_id, 'source_sha256': public.sha256, 'source_path': public.source_path, 'pdf_page': 1,
                   'start_char': 0, 'end_char': len(public.pages[0]), 'quote': public.pages[0]}
            retrieval = {'reference_sha256': hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                         'candidates': [{'candidate_id': 'pdfplumber-render-guard-900-180',
                                         'tasks': [{'task_id': 'RULE-FIXTURE', 'hits': [hit]}]}]}
            (ws.results / 'retrieval-rules.json').write_text(json.dumps(retrieval))
            captured, loaded = [], []
            def complete(messages, model, **kwargs):
                captured.append(json.dumps(messages, ensure_ascii=False))
                return {'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 1},
                        'requested_model': model, 'output': {'task_id': 'RULE-FIXTURE', 'status': 'supported', 'answer': '夹具',
                                                             'evidence_ids': ['E1'], 'limitations': ''}}
            def load(path):
                loaded.append(str(path))
                return [public]
            with patch('tender_research.llm_rules.create_provider') as client, patch('tender_research.llm_rules.load_documents', side_effect=load), contextlib.redirect_stdout(io.StringIO()):
                client.return_value.provider_id = 'deepseek'
                client.return_value.complete.side_effect = complete
                run_rules_trial(ws, 'deepseek-flash')
            self.assertEqual(len(captured), 1)
            self.assertEqual(len(loaded), 1)
            self.assertNotIn('CASE-001', loaded[0])
            self.assertNotIn('FILE-002', captured[0])


if __name__ == '__main__':
    unittest.main()
