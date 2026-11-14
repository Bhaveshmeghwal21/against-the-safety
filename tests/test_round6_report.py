"""Offline wrapper checks; no model output or network access is needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.report_round6 import checked, committed_selection


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


def test_grid_amendment_does_not_unlock_sealed_test(monkeypatch):
    from types import SimpleNamespace
    from scripts import report_round6
    monkeypatch.setattr(report_round6.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=b'## Amendment 2: threshold-grid revision\n'))
    with pytest.raises(ValueError, match='Amendment 3'):
        committed_selection(report_round6.ROOT / 'reports/round6_selection.json')


def test_selection_amendment_must_match_the_committed_report(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace
    from scripts import report_round6

    report = {'chosen_checker': 'V2m', 'checker_gates': {'V2m': {'passes': True}}}
    saved = json.dumps(report) + '\n'
    selection = tmp_path / 'reports/round6_selection.json'
    selection.parent.mkdir()
    selection.write_text(saved, encoding='utf-8')
    fingerprint = hashlib.sha256(saved.encode('utf-8')).hexdigest()
    prereg = f'## Amendment 3: development selection\n\nSelection report SHA256: `{fingerprint}`\n'
    def git_show(cmd, **kwargs):
        value = prereg if 'preregistration.md' in cmd[-1] else saved
        return SimpleNamespace(returncode=0, stdout=value.encode('utf-8'))
    monkeypatch.setattr(report_round6, 'ROOT', tmp_path)
    monkeypatch.setattr(report_round6.subprocess, 'run', git_show)
    assert committed_selection(selection) == report
    prereg = prereg.replace(fingerprint, '0' * 64)
    with pytest.raises(ValueError, match='does not match'):
        committed_selection(selection)
