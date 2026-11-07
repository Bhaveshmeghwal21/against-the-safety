"""Frozen final-test comparison, including parent-question grouped uncertainty."""

import argparse
import json
import random
from pathlib import Path


def read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def describe(rr):
    rate = lambda xs, predicate: sum(predicate(x) for x in xs) / len(xs) if xs else None
    ans, hidden = [r for r in rr if r['answerable']], [r for r in rr if not r['answerable']]
    strong = lambda r: r['evidence_supported_good']
    components = {k: rate([r for r in rr if r['qtype'] == k], strong) for k in ('single', 'compound')}
    return {'macro': sum(components.values()) / len(components), 'components': components,
            'legacy_success': rate(rr, lambda r: r['good']), 'supported_success': rate(ans, strong),
            'unsupported': sum(r['outcome'].startswith('unsupported_') for r in hidden), 'hidden_n': len(hidden),
            'unsupported_rate': rate(hidden, lambda r: r['outcome'].startswith('unsupported_')),
            'false_abstain': rate(ans, lambda r: r['outcome'] == 'false_abstain'),
            'invalid': rate(rr, lambda r: r['outcome'] == 'invalid'),
            'exact_match_when_answered': rate([r for r in ans if r['final'] == 'answer'], lambda r: bool(r['em'])),
            'n': len(rr)}


def paired_interval(a, b, statistic, reps=3000):
    rng = random.Random(47)
    def group(rows):
        out = {}
        for r in rows:
            out.setdefault(r['qtype'], {}).setdefault(r['qid'], []).append(r)
        return out
    aa, bb = group(a), group(b)
    assert {k: set(v) for k, v in aa.items()} == {k: set(v) for k, v in bb.items()}
    values = []
    for _ in range(reps):
        ra, rb = [], []
        for kind in sorted(aa):
            ids = sorted(aa[kind])
            for qid in rng.choices(ids, k=len(ids)):
                ra.extend(aa[kind][qid])
                rb.extend(bb[kind][qid])
        values.append(statistic(rb) - statistic(ra))
    values.sort()
    return [values[int(.025 * reps)], values[int(.975 * reps)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='append', required=True)
    ap.add_argument('--selection', default='reports/round4_selection.json')
    ap.add_argument('--out', default='reports/round4_test')
    args = ap.parse_args()
    selected = json.loads(Path(args.selection).read_text(encoding='utf-8'))['selected']
    candidate = selected or 'candE1'
    paths = {}
    for root in args.eval:
        for p in Path(root).rglob('scores.jsonl'):
            if p.parent.name == 'lenient' and p.parent.parent.name.endswith('_bench_r4test'):
                paths[p.parent.parent.name.removesuffix('_bench_r4test')] = p
    rows = {name: read(paths[name]) for name in ('base', 'main', 'candA', candidate)}
    expected = {(r['qid'], r['variant']) for r in rows['main']}
    for name, rr in rows.items():
        assert len(rr) == len(expected)
        assert {(r['qid'], r['variant']) for r in rr} == expected, name
    descriptions = {name: describe(rr) for name, rr in rows.items()}
    macro_ci = paired_interval(rows['main'], rows[candidate], lambda rr: describe(rr)['macro'])
    safety_ci = paired_interval(rows['main'], rows[candidate], lambda rr: describe(rr)['unsupported_rate'])
    base, cand = descriptions['main'], descriptions[candidate]
    qualifies = bool(selected and macro_ci[0] > 0 and cand['unsupported'] <= base['unsupported'])
    report = {'candidate': candidate, 'preselected': selected is not None, 'systems': descriptions,
              'macro_difference': cand['macro'] - base['macro'], 'macro_difference_ci95': macro_ci,
              'unsupported_difference': cand['unsupported_rate'] - base['unsupported_rate'],
              'unsupported_difference_ci95': safety_ci, 'recommended_default': candidate if qualifies else 'main',
              'note': 'A diagnostic checkpoint cannot be promoted from this test when development selection failed.'}
    out = Path(args.out)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Round 4 frozen final test', '',
             '`base` is the original Tiny Aya Global without our LoRA, not the separate Tiny Aya Base model.', '',
             '| system | macro evidence success | single | compound | unsupported / hidden | false refusals | invalid |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, r in descriptions.items():
        lines.append(f"| {name} | {r['macro']:.3f} | {r['components']['single']:.3f} | {r['components']['compound']:.3f} | "
                     f"{r['unsupported']}/{r['hidden_n']} | {r['false_abstain']:.3f} | {r['invalid']:.3f} |")
    lines += ['', f"Candidate versus main: {report['macro_difference']:+.3f}, 95% interval {macro_ci}.",
              f"Unsupported-answer difference: {report['unsupported_difference']:+.3f}, 95% interval {safety_ci}.",
              f"Primary-test recommendation, provisional until transfer checks and answer inspection: **{report['recommended_default']}**.", '', report['note']]
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
