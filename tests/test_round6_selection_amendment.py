"""Only a correctly applied development selection can be recorded."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.select_round6 import binary_diagnostics, checker_gate, recovery_gate
from scripts.write_round6_selection_amendment import render_amendment


def result():
    rows = [{'answer': 'yes', 'answerable': True, 'outcome': 'supported_correct', 'p': .8}] * 10
    rows += [{'answer': 'no', 'answerable': False, 'outcome': 'unsupported_wrong', 'p': .1}] * 20
    gates = {name: checker_gate(rows) for name in ('V', 'V2m', 'V2b')}
    binary = {name: binary_diagnostics(rows) for name in gates}
    recovery = recovery_gate([{'answerable': False, 'outcome': None, 'p': None}] * 20)
    return {'data': 'r4dev and HotpotQA dev only', 'checker_gates': gates,
            'binary_answer_diagnostics': binary, 'chosen_checker': 'V2m',
            'second_look_gate': recovery, 'second_look_enabled': False}


def test_selection_amendment_records_fixed_tie_and_diagnostics():
    text = render_amendment(result())
    assert '## Amendment 3: development selection' in text
    assert 'V2m' in text and '0.20' in text and 'yes/no' in text.lower()
    assert 'second look' in text.lower() and 'off' in text.lower()


def test_inconsistent_or_wrong_grid_cannot_be_recorded():
    data = result()
    data['chosen_checker'] = 'V2b'
    with pytest.raises(ValueError, match='tie'):
        render_amendment(data)
    data = result()
    data['checker_gates']['V2m']['thresholds'].pop(0)
    with pytest.raises(ValueError, match='grid'):
        render_amendment(data)


def test_failed_safety_gate_reports_counts_at_highest_threshold():
    rows = [{'answer': 'yes', 'answerable': True, 'outcome': 'supported_correct', 'p': .8}] * 10
    rows += [{'answer': 'no', 'answerable': False, 'outcome': 'unsupported_wrong', 'p': .99}] * 10
    gates = {name: checker_gate(rows) for name in ('V', 'V2m', 'V2b')}
    data = {'data': 'r4dev and HotpotQA dev only', 'checker_gates': gates,
            'binary_answer_diagnostics': {name: binary_diagnostics(rows) for name in gates},
            'chosen_checker': None, 'second_look_gate': None, 'second_look_enabled': False}
    text = render_amendment(data)
    assert 'no threshold passed' in text and 'hidden answers passed 10/10' in text
    assert 'Selected checker: none; retain V' in text


def test_inline_future_heading_does_not_count_as_an_existing_amendment(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import write_round6_selection_amendment as writer

    prior = '# Registration\n\n## Amendment 2: grid\n\nFuture: `## Amendment 3: development selection`.\n'
    prereg = tmp_path / 'prereg.md'
    selection = tmp_path / 'selection.json'
    prereg.write_text(prior, encoding='utf-8')
    selection.write_text(json.dumps(result()), encoding='utf-8')
    monkeypatch.setattr(writer, 'PREREG', prereg)
    monkeypatch.setattr(writer, 'SELECTION', selection)
    monkeypatch.setattr(writer.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout=prior.encode('utf-8')))
    writer.main()
    assert '\n## Amendment 3: development selection\n' in prereg.read_text(encoding='utf-8')
