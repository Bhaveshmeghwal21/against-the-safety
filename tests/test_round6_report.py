"""Offline wrapper checks; no model output or network access is needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.report_round6 import checked


def row(answerable=True):
    return {'final': 'answer', 'answerable': answerable, 'outcome_v2': 'supported_correct',
            'outcome_v3': 'supported_correct', 'good_v2': True, 'good_v3': True,
            'strong_v2': True, 'strong_v3': True}


def case(missing=False):
    return {'passages': 'evidence', 'deterministic': {'cites_unshown': missing,
                                                     'answer_part_absent_from_cited': False}}


def test_threshold_is_inclusive_and_rejections_become_refusals():
    assert checked(row(), case(), .8, .8)['final'] == 'answer'
    rejected = checked(row(False), case(), .79, .8)
    assert rejected['final'] == 'abstain' and rejected['outcome_v3'] == 'correct_abstain'


def test_deterministic_rejection_never_needs_a_probability():
    assert checked(row(), case(missing=True), None, .5)['outcome_v3'] == 'false_abstain'
    with pytest.raises(ValueError, match='probability'):
        checked(row(), case(), None, .5)
