"""Checks for comparison-critical handling, with no Azure or model calls."""
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import azure_rag_eval as A
import report_azure_compare as R
from ragharness import protocol


class ComparisonTests(unittest.TestCase):
    def test_round6_requires_the_actual_local_test_jobs(self):
        with self.assertRaisesRegex(ValueError, '--reference-eval'):
            A.build_jobs('r6t_hotpot')

    def test_shared_request_limiter_spaces_concurrent_requests(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / '.cache/tmp'
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_tmp) as folder, patch.object(A, 'ROOT', Path(folder)):
            client = A.Azure('test-key', 512, 'https://example.test/chat/completions', .03)
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(lambda _: client.wait_for_request_slot(), range(5)))
            self.assertGreaterEqual(time.monotonic() - start, .105)

    def test_nested_azure_filter_is_separate_from_model_failure(self):
        body = b'{"choices":[{"finish_reason":"content_filter"}],"usage":{"prompt_tokens":620}}'
        error = HTTPError('https://example.test', 400, 'Bad Request', {}, BytesIO(body))
        with patch('azure_rag_eval.urllib.request.urlopen', side_effect=error):
            with self.assertRaises(A.ProviderFiltered) as caught:
                A.Azure('test-key', 512, 'https://example.test/chat/completions').request([])
        self.assertEqual(caught.exception.usage['prompt_tokens'], 620)

    def test_native_normalization_does_not_fabricate_citations(self):
        plain = 'Paris <citations>c00012</citations>'
        parsed, _ = protocol.parse_turn_lenient(A.canonical_text(plain))
        self.assertEqual(parsed.kind, 'answer')
        self.assertEqual(parsed.answer, 'Paris')
        self.assertEqual(parsed.citations, ['c00012'])
        unsupported = 'Paris'
        parsed, _ = protocol.parse_turn_lenient(A.canonical_text(unsupported))
        self.assertEqual(parsed.kind, 'invalid')

    def test_explicit_refusal_is_abstention(self):
        parsed, _ = protocol.parse_turn_lenient(A.canonical_text("I cannot answer from the available evidence."))
        self.assertEqual(parsed.kind, 'abstain')

    def test_pairwise_common_sample_drops_missing_and_evidence_exclusions(self):
        def row(qid, excluded=False):
            return {'part': 'p', 'qid': str(qid), 'variant': 'open', 'excluded_v2': excluded}
        data = {'main': [row(1), row(2), row(3), row(4)],
                'local': [row(1), row(2), row(3), row(4)],
                'azure_native': [row(1), row(2, True), row(3)]}
        _, keys, counts = R.common_data(data)
        self.assertEqual(keys, [('p', '1', 'open'), ('p', '3', 'open')])
        self.assertEqual(counts['provider_unavailable_or_missing'], 1)
        self.assertEqual(counts['excluded_on_at_least_one_actor'], 1)

    def test_fail_closed_scores_invalid_as_refusal(self):
        rows = [{'final':'invalid', 'answerable':False}, {'final':'invalid', 'answerable':True}]
        actual = R.S.fail_closed(rows)
        self.assertEqual([x['outcome_v2'] for x in actual], ['correct_abstain', 'false_abstain'])
        self.assertEqual(rows[0]['final'], 'invalid')


if __name__ == '__main__':
    unittest.main()
