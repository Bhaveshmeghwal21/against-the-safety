"""Checks for DeepSeek's metering calculations, without model calls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_deepseek_roi as R


class CostTests(unittest.TestCase):
    def test_cached_input_is_priced_once_and_blocked_usage_is_kept(self):
        rows = [
            {'score': {}, 'api_calls': [{'usage': {'prompt_tokens': 1000, 'completion_tokens': 200,
                'prompt_tokens_details': {'cached_tokens': 400, 'cache_write_tokens': 100}}}]},
            {'score': {'provider_blocked': True}, 'api_calls': [{'usage': {'prompt_tokens': 500}}]},
        ]
        with patch.object(R.R, 'read', return_value=rows):
            result = R.compute(Path('unused'), ('suite',))
        expected = (1100*.19 + 400*.028 + 200*.51)/1_000_000
        self.assertAlmostEqual(result['recorded_estimate_usd'], expected)
        self.assertEqual(result['provider_blocked'], 1)
        self.assertEqual(result['reported_cache_write_tokens'], 100)
        self.assertAlmostEqual(result['estimated_usd_per_10000'], expected * 5000)


if __name__ == '__main__':
    unittest.main()
