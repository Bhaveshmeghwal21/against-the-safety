"""The selection gates use the first safety threshold, inclusive boundaries and no empty-set wins."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.select_round6 import THRESHOLDS, binary_diagnostics, checker_gate, recovery_gate


def test_checker_grid_starts_at_twenty_percent_and_keeps_gate_boundaries():
    assert THRESHOLDS == [i / 100 for i in range(20, 100, 5)]
    correct = [{'answerable': True, 'outcome': 'supported_correct', 'p': .20}] * 9
    correct += [{'answerable': True, 'outcome': 'supported_correct', 'p': .10}]
    hidden = [{'answerable': False, 'outcome': 'unsupported_wrong', 'p': .80}] * 3
    hidden += [{'answerable': False, 'outcome': 'unsupported_wrong', 'p': .10}] * 17
    result = checker_gate(correct + hidden)
    assert result['chosen']['threshold'] == .20
    assert (result['chosen']['correct_kept'], result['chosen']['hidden_passed']) == (9, 3)
    assert result['passes']  # exactly 90% retained and 15% hidden passed


def test_yes_and_no_diagnostics_are_separate_and_do_not_change_gate():
    rows = [
        {'answer': 'Yes.', 'answerable': True, 'outcome': 'supported_correct', 'p': .60},
        {'answer': 'yes', 'answerable': False, 'outcome': 'unsupported_wrong', 'p': .30},
        {'answer': 'no', 'answerable': True, 'outcome': 'wrong_answer', 'p': .80},
        {'answer': 'no', 'answerable': False, 'outcome': 'unsupported_wrong', 'p': .10},
        {'answer': 'yes, perhaps', 'answerable': True, 'outcome': 'wrong_answer', 'p': .90},
    ]
    before = checker_gate(rows)
    diagnostic = binary_diagnostics(rows)
    yes = next(r for r in diagnostic['yes'] if r['threshold'] == .50)
    no = next(r for r in diagnostic['no'] if r['threshold'] == .50)
    assert (yes['correct_kept'], yes['correct_total'], yes['hidden_passed'], yes['hidden_total']) == (1, 1, 0, 1)
    assert (no['wrong_passed'], no['wrong_total'], no['hidden_passed'], no['hidden_total']) == (1, 1, 0, 1)
    assert checker_gate(rows) == before


def test_checker_uses_first_safe_threshold_then_tests_retention():
    correct = [{'answerable': True, 'outcome': 'supported_correct', 'p': .80}] * 9
    correct += [{'answerable': True, 'outcome': 'supported_correct', 'p': .10}]
    hidden = [{'answerable': False, 'outcome': 'unsupported_wrong', 'p': .70}] * 10
    result = checker_gate(correct + hidden)
    assert result['chosen']['threshold'] == .75
    assert result['chosen']['correct_kept'] == 9 and result['passes']


def test_no_positive_or_hidden_data_cannot_pass():
    assert not checker_gate([])['passes']
    assert not recovery_gate([])['passes']


def test_recovery_needs_a_real_gain_and_keeps_equal_threshold():
    hidden = [{'answerable': False, 'outcome': None, 'p': None}] * 20
    assert not recovery_gate(hidden)['passes']
    good = [{'answerable': True, 'outcome': 'supported_correct', 'p': .60}] * 3
    result = recovery_gate(hidden + good)
    assert result['passes'] and result['chosen']['threshold'] == .60
    assert result['chosen']['correct_added'] == 3


def test_development_selection_uses_only_complete_named_runs(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import select_round6 as selector

    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    cases, contexts, verdicts = [], [], []
    for run in sorted(selector.ALLOWED_RUNS):
        transcripts = []
        for qid, answerable, kind in [('right', True, 'answer'), ('hidden', False, 'answer'), ('refusal', False, 'abstain')]:
            outcome = 'supported_correct' if answerable else 'unsupported_wrong' if kind == 'answer' else 'correct_abstain'
            row = {'qid': qid, 'variant': 'open', 'final': {'kind': kind},
                   'provided_evidence': [{'chunk_id': 'c00001'}] if kind == 'abstain' else [], 'steps': [], 'mock_score': {
                'final': kind, 'answerable': answerable, 'outcome_v3': outcome, 'excluded_v3': False}}
            transcripts.append(row)
            key = {'run': run, 'qid': qid, 'variant': 'open'}
            if kind == 'answer':
                cases.append({**key, 'answer': 'yes' if answerable else 'no', 'passages': 'shown',
                              'deterministic': {'cites_unshown': False, 'answer_part_absent_from_cited': False}})
                verdicts.append({**key, 'p_pos': .9 if answerable else .1})
            else:
                contexts.append({**key, 'id': run, 'system': run.split('_')[0], 'transcript': row})
        write(tmp_path / 'eval' / run / 'lenient/transcripts.jsonl', transcripts)
    write(tmp_path / 'cases/answers.jsonl', cases)
    write(tmp_path / 'cases/recover_contexts.jsonl', contexts)
    write(tmp_path / 'cases/forced_all.jsonl', [])
    for actor in ('main', 'candF1'):
        write(tmp_path / f'forced/forced_{actor}.jsonl', [{'id': c['id']} for c in contexts if c['system'] == actor])
    for checker in ('V', 'V2m', 'V2b'):
        write(tmp_path / f'verify/{checker}/verdicts_A.jsonl', verdicts)
        write(tmp_path / f'verify_forced/{checker}/verdicts_A.jsonl', [])
    monkeypatch.setattr(selector, 'score_transcript', lambda run, transcript: transcript['mock_score'])
    monkeypatch.setattr(sys, 'argv', ['select_round6.py', '--run', str(tmp_path), '--out', str(tmp_path / 'result')])
    selector.main()
    result = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))
    assert result['chosen_checker'] == 'V2m'  # fixed tie break
    assert not result['second_look_enabled']
    for checker in ('V', 'V2m', 'V2b'):
        binary = result['binary_answer_diagnostics'][checker]
        assert binary['yes'][0]['correct_total'] == 4
        assert binary['no'][0]['hidden_total'] == 4
    text = (tmp_path / 'result.md').read_text(encoding='utf-8')
    assert all(f'### {checker}: yes/no answers' in text for checker in ('V', 'V2m', 'V2b'))
