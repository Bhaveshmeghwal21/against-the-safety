"""Independently replay the development selection; never reads Round 6 test data."""
import json
import math
from collections import Counter
from pathlib import Path

import rescore_round5 as RS
from ragharness import checker, rescore
from ragharness.retrieval import Hit

ROOT = Path(__file__).resolve().parents[1]
RUNS = {'main_bench_r4dev', 'main_dev', 'candF1_bench_r4dev', 'candF1_dev'}
GRID = [i / 100 for i in range(20, 100, 5)]


def key(row):
    return row['run'], row['qid'], row['variant']


def rows_by_key(rows):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate case or verdict identifiers')
    return result


def counts(rows, threshold):
    correct = [r for r in rows if r['outcome'] == 'supported_correct']
    hidden = [r for r in rows if not r['answerable']]
    wrong = [r for r in rows if r['answerable'] and r['outcome'] != 'supported_correct']
    return {'threshold': threshold, 'correct_kept': sum(r['p'] >= threshold for r in correct),
            'correct_total': len(correct), 'hidden_passed': sum(r['p'] >= threshold for r in hidden),
            'hidden_total': len(hidden), 'wrong_passed': sum(r['p'] >= threshold for r in wrong),
            'wrong_total': len(wrong)}


def main():
    root = ROOT / 'runs/kaggle/ats-r6-system-dev/out'
    eval_root = root / 'eval'
    if not eval_root.is_dir():
        raise FileNotFoundError(eval_root)
    names = {p.name for p in eval_root.iterdir() if p.is_dir()}
    if names != RUNS:
        raise ValueError('Evaluation directory contains unexpected or missing runs')
    transcripts = {}
    for run in sorted(RUNS):
        _, world = RS.kind_of(run)
        corp, questions = RS.world(world)
        for t in RS.read(eval_root / run / 'lenient/transcripts.jsonl'):
            if t['final']['kind'] != 'answer':
                continue
            k = (run, t['qid'], t['variant'])
            if k in transcripts:
                raise ValueError('Duplicate development transcript')
            score = rescore.score(t, questions[t['qid']], corp)
            ids = [h['chunk_id'] for h in t.get('provided_evidence', [])]
            ids += [h['chunk_id'] for step in t['steps'] for h in step['retrieved']]
            hits = []
            for cid in dict.fromkeys(ids):
                chunk = corp.chunk(cid)
                hits.append(Hit(cid, chunk['title'], chunk['text'], 0.0))
            flags = checker.deterministic_flags(t['final']['answer'], t['final']['citations'], hits, t.get('max_chars', 800))
            transcripts[k] = (t, score, flags)
    cases = rows_by_key(RS.read(root / 'cases/answers.jsonl'))
    if set(cases) != set(transcripts):
        raise ValueError('Answered development cases do not match transcripts')
    eligible = {}
    for k, case in cases.items():
        t, score, flags = transcripts[k]
        if case['answer'] != t['final']['answer'] or case['citations'] != t['final']['citations'] or case['question'] != t['question']:
            raise ValueError('A checker case does not match its transcript')
        for field in ('cites_unshown', 'answer_part_absent_from_cited', 'absent_parts'):
            if case['deterministic'][field] != flags[field]:
                raise ValueError('A deterministic checker flag does not reproduce')
        if case['passages'] != flags['passages']:
            raise ValueError('A cited-passage rendering does not reproduce')
        if not score['excluded_v3'] and flags['passages'] and not flags['cites_unshown'] and not flags['answer_part_absent_from_cited']:
            eligible[k] = {'answerable': score['answerable'], 'outcome': score['outcome_v3'], 'answer': case['answer']}
    report = json.loads((ROOT / 'reports/round6_selection.json').read_text(encoding='utf-8'))
    checked = {}
    for name in ('V', 'V2m', 'V2b'):
        verdicts = rows_by_key(RS.read(root / f'verify/{name}/verdicts_A.jsonl'))
        rows = []
        for k, item in eligible.items():
            if k not in verdicts:
                raise ValueError(f'Missing {name} development verdict')
            p = verdicts[k]['p_pos']
            if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f'Invalid {name} development verdict')
            rows.append({**item, 'p': p})
        expected = [counts(rows, t) for t in GRID]
        if expected != report['checker_gates'][name]['thresholds']:
            raise ValueError(f'{name} threshold counts do not reproduce')
        for answer in ('yes', 'no'):
            subset = [r for r in rows if r['answer'].strip().casefold().rstrip('.!?').strip() == answer]
            if [counts(subset, t) for t in GRID] != report['binary_answer_diagnostics'][name][answer]:
                raise ValueError(f'{name} {answer} diagnostics do not reproduce')
        checked[name] = {'eligible_checker_cases': len(rows), 'outcomes': dict(Counter(r['outcome'] for r in rows)),
                         'verdicts_in_file': len(verdicts), 'thresholds_checked': len(GRID)}
    result = {'scope': 'r4dev and HotpotQA dev only; no Round 6 test data or outputs',
              'answered_transcripts': len(transcripts), 'eligible_checker_cases': len(eligible),
              'deterministic_flags_and_passages_match': True, 'checker_thresholds_and_binary_diagnostics_match': True,
              'checkers': checked}
    path = ROOT / 'reports/round6_selection_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
