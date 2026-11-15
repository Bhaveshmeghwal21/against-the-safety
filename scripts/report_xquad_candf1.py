"""Compare the candF1 XQuAD run to the already opened main/base run, without model calls."""
import argparse
import json
from pathlib import Path

import report_xquad as xq
import rescore_round5 as rs
import select_round5 as old_select
from report_reference_policies import always_refuse
from select_round6 import verdicts

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ('batch_size', 'concurrency', 'max_new_tokens', 'max_input_tokens', 'abstain_bias')


def load_run(root, actor, lang):
    rows, mismatches, summary = xq.load_run(root, actor, lang)
    path = root / 'eval' / f'{actor}_bench_xq_{lang}' / 'lenient/transcripts.jsonl'
    transcripts = rs.read(path)
    if len(rows) != len(transcripts):
        raise ValueError(f'Transcript count does not match scores in {path}')
    for row, transcript in zip(rows, transcripts):
        if (row['qid'], row['variant']) != (transcript['qid'], transcript['variant']):
            raise ValueError(f'Transcript ordering does not match scores in {path}')
        row['question'] = transcript['question']
    return rows, mismatches, summary


def index_cases(path):
    rows = rs.read(path)
    result = {(r['run'], r['qid'], r['variant']): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate checker case in {path}')
    return result


def compare_conditions(main, cand, main_summary, cand_summary):
    """Require identical episode IDs, eligibility, answerability and evaluation settings."""
    def index(rows):
        result = {(r['qid'], r['variant']): (r['answerable'], r['excluded_v3']) for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate episode ID')
        return result
    if index(main) != index(cand):
        raise ValueError('Main and candF1 episode sets, labels or eligibility differ')
    if main_summary['index_hash'] != cand_summary['index_hash']:
        raise ValueError('Main and candF1 corpus index hashes differ')
    mg, cg = main_summary['generation'], cand_summary['generation']
    if any(mg.get(key) != cg.get(key) for key in SETTINGS):
        raise ValueError('Main and candF1 generation settings differ')


def repaired(rows, actor, lang, cases, probabilities):
    run = f'{actor}_bench_xq_{lang}'
    answered = {(run, r['qid'], r['variant']) for r in rows if r['final'] == 'answer'}
    available = {key for key in cases if key[0] == run}
    if answered != available:
        raise ValueError(f'Checker cases do not cover exactly the answers in {run}')
    for r in rows:
        if r['final'] == 'answer':
            case = cases[(run, r['qid'], r['variant'])]
            if case['answer'] != r['answer'] or case['question'] != r['question']:
                raise ValueError(f'Checker case differs from transcript in {run}')
    return [xq.checked(r, cases.get((run, r['qid'], r['variant'])),
                       probabilities.get((run, r['qid'], r['variant']))) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate-run', required=True)
    ap.add_argument('--baseline-run', default='runs/kaggle/ats-r6-system-xq/out')
    ap.add_argument('--out', default='reports/xquad_candf1')
    args = ap.parse_args()
    baseline, candidate = ((ROOT / p).resolve() for p in (args.baseline_run, args.candidate_run))
    output = (ROOT / args.out).resolve()
    if any(not p.is_relative_to(ROOT.resolve()) for p in (baseline, candidate, output)):
        raise ValueError('Inputs and outputs must stay within the workspace')
    cases_b = index_cases(baseline / 'cases/answers.jsonl')
    cases_c = index_cases(candidate / 'cases/answers.jsonl')
    pred_b = verdicts(baseline / 'verify/V/verdicts_A.jsonl')
    pred_c = verdicts(candidate / 'verify/V/verdicts_A.jsonl')
    systems, comparisons, generation, mismatches, audit = {}, {}, {}, {}, {}
    for lang in xq.LANGS:
        manifest = json.loads((ROOT / 'data/bench' / f'xq_{lang}/manifest.json').read_text(encoding='utf-8'))
        raw_main, mm, sm = load_run(baseline, 'main', lang)
        raw_base, bm, sb = load_run(baseline, 'base', lang)
        raw_cand, cm, sc = load_run(candidate, 'candF1', lang)
        if mm or bm or cm:
            raise ValueError(f'Saved and rescored XQuAD results differ in {lang}')
        compare_conditions(raw_main, raw_cand, sm, sc)
        compare_conditions(raw_main, raw_base, sm, sb)
        if sm['index_hash'] != manifest['index_hash']:
            raise ValueError(f'Current index does not match the saved {lang} run')
        built = {'base': raw_base, 'main': raw_main, 'main+FB+V': repaired(raw_main, 'main', lang, cases_b, pred_b),
                 'candF1': raw_cand, 'candF1+FB': old_select.fail_closed(raw_cand),
                 'candF1+FB+V': repaired(raw_cand, 'candF1', lang, cases_c, pred_c),
                 'always refuse': always_refuse(raw_main)}
        eligible = {name: [r for r in rows if not r['excluded_v3']] for name, rows in built.items()}
        systems[lang] = {name: xq.measures(rows) for name, rows in eligible.items()}
        comparisons[lang] = xq.paired(eligible['candF1+FB+V'], eligible['main+FB+V'], manifest['article_of'])
        generation[lang] = {'main': sm['generation'], 'candF1': sc['generation']}
        mismatches[lang] = {'base': len(bm), 'main': len(mm), 'candF1': len(cm)}
        audit[lang] = {'main': sm.get('auditkit', {}), 'candF1': sc.get('auditkit', {})}
    report = {'scope': 'Exploratory follow-up on already opened XQuAD questions; no new model was trained.',
              'checker': 'Main adapter V, wording A, threshold 0.5; no second look.',
              'systems': systems, 'candF1_minus_main': comparisons, 'generation': generation,
              'stored_score_mismatches': mismatches, 'auditkit_raw_systems': audit,
              'bootstrap': '3,000 paired article-cluster draws, seed 47.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# CandF1 multilingual XQuAD comparison', '', report['scope'], report['checker'], report['bootstrap'], '']
    pct = lambda x: '—' if x is None else f'{100*x:.1f}%'
    for lang in xq.LANGS:
        lines += [f'## {xq.NAMES[lang]}', '', '| System | Success | Correct and cited | Hidden answers | Answer coverage | Correct among answered | Invalid |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name, row in systems[lang].items():
            lines.append(f"| {name} | {pct(row['success'])} | {pct(row['correct_and_cited'])} | {row['hidden_answers']}/{row['n_hidden']} | "
                         f"{pct(row['answered_share'])} | {pct(row['correct_among_answered'])} | {pct(row['invalid'])} |")
        c = comparisons[lang]
        lines += ['', f"CandF1+FB+V minus main+FB+V, on {c['questions']} paired questions:"]
        for metric in ('correct_and_cited', 'hidden_answer_rate', 'success'):
            d = c[metric]
            lines.append(f"- {metric}: {100*d['difference']:+.1f} pp (95% interval {100*d['ci95'][0]:+.1f} to {100*d['ci95'][1]:+.1f}).")
        lines.append('')
    lines += ['Hidden-evidence outcomes are annotation-based flags, not human-adjudicated hallucinations.',
              'The XQuAD results were inspected before this follow-up; do not use this as a fresh blind selection test.']
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(output.with_suffix('.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
