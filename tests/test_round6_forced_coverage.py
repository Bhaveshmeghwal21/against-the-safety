"""Actor-specific coverage must not count the other actor's contexts as missing."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.build_recover_cases import forced_coverage


def test_coverage_counts_only_the_requested_actor():
    contexts = {'a': {'system': 'main'}, 'b': {'system': 'main'}, 'c': {'system': 'candF1'}}
    result = forced_coverage(contexts, [{'id': 'a'}, {'id': 'b'}], 'main')
    assert (result['input_contexts'], result['contexts'], result['forced'], result['missing_forced']) == (3, 2, 2, 0)
    assert forced_coverage(contexts, [{'id': 'a'}])['missing_forced'] == 1


def test_empty_or_wrong_actor_requires_an_unambiguous_scope():
    contexts = {'a': {'system': 'main'}, 'b': {'system': 'candF1'}}
    with pytest.raises(ValueError, match='Supply --system'):
        forced_coverage(contexts, [])
    with pytest.raises(ValueError, match='requested system'):
        forced_coverage(contexts, [{'id': 'a'}], 'candF1')
    assert forced_coverage(contexts, [], 'main')['missing_forced'] == 1
