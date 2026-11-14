"""Apply the preregistered gates to completed round-six development outputs only."""
import argparse
import json
import math
from pathlib import Path

import rescore_round5 as RS
from ragharness import rescore

ALLOWED_RUNS = {'main_bench_r4dev', 'main_dev', 'candF1_bench_r4dev', 'candF1_dev'}
THRESHOLDS = [i / 100 for i in range(20, 100, 5)]
RECOVERY_THRESHOLDS = [i / 100 for i in range(60, 100, 5)]


def valid(case):
    flags = case['deterministic']
    return bool(case['passages']) and not flags['cites_unshown'] and not flags['answer_part_absent_from_cited']


def threshold_counts(cases, tau):
    correct = [r for r in cases if r['outcome'] == 'supported_correct']
    hidden = [r for r in cases if not r['answerable']]
    wrong = [r for r in cases if r['answerable'] and r['outcome'] != 'supported_correct']
    return {'threshold': tau, 'correct_kept': sum(r['p'] >= tau for r in correct), 'correct_total': len(correct),
            'hidden_passed': sum(r['p'] >= tau for r in hidden), 'hidden_total': len(hidden),
            'wrong_passed': sum(r['p'] >= tau for r in wrong), 'wrong_total': len(wrong)}


def checker_gate(cases):
    """Rows have outcome, answerable, and p; deterministic checks already passed."""
    table = [threshold_counts(cases, tau) for tau in THRESHOLDS]
    # Pick the first threshold satisfying the hidden gate, then apply the retention gate there.
    chosen = next((r for r in table if r['hidden_total'] and r['hidden_passed'] / r['hidden_total'] <= .15), None)
    adopted = bool(chosen and chosen['correct_total'] and chosen['correct_kept'] / chosen['correct_total'] >= .9)
    return {'thresholds': table, 'chosen': chosen, 'passes': adopted}


def binary_diagnostics(cases):
    """Exact yes/no answer rows, reported separately and never used for selection."""
    def answer_class(case):
        return str(case.get('answer', '')).strip().casefold().rstrip('.!?').strip()
    return {word: [threshold_counts([case for case in cases if answer_class(case) == word], tau) for tau in THRESHOLDS]
            for word in ('yes', 'no')}


def recovery_gate(rows):
    """Every eligible refusal has a row, even when forcing or checking failed."""
    hidden = [r for r in rows if not r['answerable']]
    table = []
    for tau in RECOVERY_THRESHOLDS:
        accepted = [r for r in rows if r['p'] is not None and r['p'] >= tau]
        added = sum(r['answerable'] and r['outcome'] == 'supported_correct' for r in accepted)
        risky = sum(not r['answerable'] for r in accepted)
        table.append({'threshold': tau, 'correct_added': added, 'hidden_added': risky, 'hidden_total': len(hidden)})
    chosen = next((r for r in table if r['hidden_total'] and r['hidden_added'] / r['hidden_total'] <= .05), None)
    adopted = bool(chosen and chosen['correct_added'] > 0 and chosen['correct_added'] >= 3 * chosen['hidden_added'])
    return {'thresholds': table, 'chosen': chosen, 'passes': adopted}


def verdicts(path):
    values = {}
    for row in RS.read(path):
        key = (row['run'], row['qid'], row['variant'])
        p = row['p_pos']
        if key in values or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f'Invalid or duplicate verifier result: {key}')
        values[key] = p
    return values


def score_transcript(run, transcript):
    _, world = RS.kind_of(run)
    corp, qmap = RS.world(world)
    return rescore.score(transcript, qmap[transcript['qid']], corp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='completed r6-system development out directory')
    ap.add_argument('--out', default='reports/round6_selection')
    args = ap.parse_args()
    root = Path(args.run)
    runs = {}
    for folder in (root / 'eval').rglob('lenient'):
        if (folder / 'transcripts.jsonl').exists():
            run = folder.parent.name
            if run not in ALLOWED_RUNS or run in runs:
                raise ValueError(f'Unexpected or duplicate development run: {run}')
            runs[run] = RS.read(folder / 'transcripts.jsonl')
    if set(runs) != ALLOWED_RUNS:
        raise FileNotFoundError(f"{root / 'eval'} (missing runs: {sorted(ALLOWED_RUNS - set(runs))})")
    scored = {}
    for run, transcripts in runs.items():
        for transcript in transcripts:
            key = (run, transcript['qid'], transcript['variant'])
            if key in scored:
                raise ValueError(f'Duplicate development transcript: {key}')
            scored[key] = score_transcript(run, transcript)
    cases = RS.read(root / 'cases/answers.jsonl')
    case_keys = [(case['run'], case['qid'], case['variant']) for case in cases]
    expected_answers = {key for key, row in scored.items() if row['final'] == 'answer'}
    if len(case_keys) != len(set(case_keys)) or set(case_keys) != expected_answers:
        raise ValueError('Development answer cases are duplicated or incomplete')
    contexts = RS.read(root / 'cases/recover_contexts.jsonl')
    forced = RS.read(root / 'cases/forced_all.jsonl')
    forced_by_id = {r['id']: r for r in forced}
    if len(forced_by_id) != len(forced) or len({r['id'] for r in contexts}) != len(contexts):
        raise ValueError('Duplicate recovery IDs')
    if not set(forced_by_id) <= {r['id'] for r in contexts}:
        raise ValueError('Forced cases do not belong to development contexts')
    expected_contexts = set()
    for run, transcripts in runs.items():
        for transcript in transcripts:
            shown = transcript.get('provided_evidence', []) or any(s.get('retrieved', []) for s in transcript.get('steps', [])[:-1])
            if transcript['final']['kind'] in ('abstain', 'invalid') and shown:
                expected_contexts.add((run, transcript['qid'], transcript['variant']))
    context_keys = [(c['run'], c['qid'], c['variant']) for c in contexts]
    if len(context_keys) != len(set(context_keys)) or set(context_keys) != expected_contexts:
        raise ValueError('Recovery contexts are duplicated or incomplete')
    # Invalid forced text is allowed, but every eligible context must have actually
    # been attempted; otherwise treating a missing output as a rejection biases safety.
    for actor in ('main', 'candF1'):
        expected_ids = {c['id'] for c in contexts if c['system'] == actor}
        if not expected_ids:
            continue
        raw = RS.read(root / f'forced/forced_{actor}.jsonl')
        ids = [item['id'] for item in raw]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f'Forced responses are duplicated or incomplete for {actor}')
    gates, binary, recovery = {}, {}, {}
    for checker in ('V', 'V2m', 'V2b'):
        predictions = verdicts(root / f'verify/{checker}/verdicts_A.jsonl')
        rows = []
        for case in cases:
            key = (case['run'], case['qid'], case['variant'])
            if key not in scored:
                raise ValueError(f'Answer case without a development transcript: {key}')
            truth = scored[key]
            if truth['excluded_v3'] or not valid(case):
                continue
            if key not in predictions:
                raise ValueError(f'Missing verifier prediction: {key}')
            rows.append({'answerable': truth['answerable'], 'outcome': truth['outcome_v3'],
                         'answer': case['answer'], 'p': predictions[key]})
        gates[checker] = checker_gate(rows)
        binary[checker] = binary_diagnostics(rows)
    passing = [c for c in ('V2m', 'V2b') if gates[c]['passes']]
    chosen = max(passing, key=lambda c: gates[c]['chosen']['correct_kept']) if passing else None
    if chosen:
        predictions = verdicts(root / f'verify_forced/{chosen}/verdicts_A.jsonl')
        rows = []
        for ctx in contexts:
            run, t = ctx['run'], ctx['transcript']
            if run not in ALLOWED_RUNS:
                raise ValueError(f'Non-development recovery context: {run}')
            truth = score_transcript(run, t)
            if truth['excluded_v3']:
                continue
            candidate = forced_by_id.get(ctx['id'])
            outcome, p = None, None
            if candidate and valid(candidate):
                key = (candidate['run'], candidate['qid'], candidate['variant'])
                if key != (run + '_forced', ctx['qid'], ctx['variant']):
                    raise ValueError(f'Forced answer identity mismatch: {key}')
                if key not in predictions:
                    raise ValueError(f'Missing forced verifier prediction: {key}')
                p = predictions[key]
                replacement = {**t, 'final': {'kind': 'answer', 'answer': candidate['answer'], 'citations': candidate['citations']}}
                outcome = score_transcript(run, replacement)['outcome_v3']
            rows.append({'answerable': truth['answerable'], 'outcome': outcome, 'p': p})
        recovery[chosen] = recovery_gate(rows)
    report = {'data': 'r4dev and HotpotQA dev only', 'checker_gates': gates, 'binary_answer_diagnostics': binary, 'chosen_checker': chosen,
              'second_look_gate': recovery.get(chosen), 'second_look_enabled': bool(chosen and recovery[chosen]['passes'])}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Round 6 development selection', '', f'Chosen checker: {chosen or "none"}.', '']
    for checker, gate in gates.items():
        lines += [f'## {checker}', '', '| Threshold | Correct kept | Hidden passed | Wrong passed |', '| ---: | ---: | ---: | ---: |']
        for row in gate['thresholds']:
            lines.append(f"| {row['threshold']:.2f} | {row['correct_kept']}/{row['correct_total']} | "
                         f"{row['hidden_passed']}/{row['hidden_total']} | {row['wrong_passed']}/{row['wrong_total']} |")
        lines.append('')
        lines += [f'### {checker}: yes/no answers', '',
                  'Diagnostic only, on eligible answers that passed deterministic checks; not a selection gate. '
                  'Exact yes/no answers allow terminal punctuation.', '',
                  '| Threshold | Yes: correct kept | Yes: hidden passed | Yes: wrong passed | '
                  'No: correct kept | No: hidden passed | No: wrong passed |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for yes, no in zip(binary[checker]['yes'], binary[checker]['no']):
            lines.append(f"| {yes['threshold']:.2f} | {yes['correct_kept']}/{yes['correct_total']} | "
                         f"{yes['hidden_passed']}/{yes['hidden_total']} | {yes['wrong_passed']}/{yes['wrong_total']} | "
                         f"{no['correct_kept']}/{no['correct_total']} | {no['hidden_passed']}/{no['hidden_total']} | "
                         f"{no['wrong_passed']}/{no['wrong_total']} |")
        lines.append('')
    if chosen:
        lines += ['## Second look', '', '| Threshold | Correct added | Hidden added / hidden refusals |', '| ---: | ---: | ---: |']
        for row in recovery[chosen]['thresholds']:
            lines.append(f"| {row['threshold']:.2f} | {row['correct_added']} | {row['hidden_added']}/{row['hidden_total']} |")
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'chosen_checker': chosen, 'second_look_enabled': report['second_look_enabled']}))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
