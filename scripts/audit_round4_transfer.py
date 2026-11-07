"""A frozen candidate's informed regression checks on six existing suites.

These sets informed the experiment design. They cannot choose a checkpoint or be
presented as new independent tests. They can veto an overall deployment claim.
"""

import argparse
import json
from pathlib import Path


def summarize(rr):
    hidden = [r for r in rr if not r['answerable']]
    ans = [r for r in rr if r['answerable']]
    return {'episodes': len(rr), 'success': sum(r['good'] for r in rr) / len(rr),
            'unsupported': sum(r['outcome'].startswith('unsupported_') for r in hidden), 'hidden_n': len(hidden),
            'supported_correct': sum(r['outcome'] == 'supported_correct' for r in ans), 'answerable_n': len(ans),
            'invalid': sum(r['outcome'] == 'invalid' for r in rr)}


def deployment_guard(sets, candidate):
    known_risks = all(sets[d][candidate]['unsupported'] <= sets[d]['main']['unsupported']
                      and sets[d][candidate]['supported_correct'] >= sets[d]['main']['supported_correct']
                      for d in ('squad2', 'rgb'))
    pooled = {n: sum(s[n]['unsupported'] for s in sets.values()) for n in ('main', candidate)}
    macro = {n: sum(s[n]['success'] for s in sets.values()) / len(sets) for n in ('main', candidate)}
    return {'known_risk_sources_preserved': known_risks, 'pooled_unsupported': pooled, 'macro_six_sets': macro,
            'passes': bool(known_risks and pooled[candidate] <= pooled['main'] and macro[candidate] >= macro['main'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--out', default='reports/round4_informed_transfer')
    args = ap.parse_args()
    report = {'candidate': args.candidate, 'protocol': 'informed regression, not independent held-out evidence', 'sets': {}}
    for dataset in ('hotpot', '2wiki', 'musique', 'squad2', 'rgb', 'rgb_int'):
        rows = {}
        for name in ('main', args.candidate):
            path = Path(args.eval) / f'{name}_bench_r3t_{dataset}' / 'lenient/scores.jsonl'
            rows[name] = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
        keys = lambda rr: {(r['qid'], r['variant']) for r in rr}
        assert len(rows['main']) == len(rows[args.candidate]) == len(keys(rows['main']))
        assert keys(rows['main']) == keys(rows[args.candidate])
        stats = {n: summarize(rr) for n, rr in rows.items()}
        safe = stats[args.candidate]['unsupported'] <= stats['main']['unsupported']
        useful = stats[args.candidate]['supported_correct'] >= stats['main']['supported_correct']
        stats['no_risk_increase_and_no_supported_answer_loss'] = safe and useful
        report['sets'][dataset] = stats
    report.update(deployment_guard(report['sets'], args.candidate))
    out = Path(args.out)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Round 4 informed transfer checks', '', report['protocol'], '',
             '| set | model | success | unsupported / hidden | supported answers / answerable |',
             '| --- | --- | ---: | ---: | ---: |']
    for dataset, stats in report['sets'].items():
        for name in ('main', args.candidate):
            s = stats[name]
            lines.append(f"| {dataset} | {name} | {s['success']:.3f} | {s['unsupported']}/{s['hidden_n']} | {s['supported_correct']}/{s['answerable_n']} |")
    lines += ['', f"Known risk-source guard: **{report['known_risk_sources_preserved']}**.",
              f"Six-set pooled unsupported answers: {report['pooled_unsupported']}.",
              f"Six-set macro success: {report['macro_six_sets']}.", f"Overall transfer guard: **{report['passes']}**."]
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
