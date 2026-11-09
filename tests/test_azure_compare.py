"""Checks for comparison-critical handling, with no Azure or model calls."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import azure_rag_eval as A
import report_azure_compare as R
from ragharness import protocol


class ComparisonTests(unittest.TestCase):
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
