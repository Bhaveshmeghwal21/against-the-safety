"""Round-4 predeclared safety-constrained selection, with complete episode alignment checks."""

import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def stats(new, old):
    hidden = [r for r in new if not r['answerable']]
    answerable = [r for r in new if r['answerable']]
    unsupported = lambda rr: sum(r['outcome'].startswith('unsupported_') for r in rr)
    good = lambda rr: sum(r['evidence_supported_good'] for r in rr) / len(rr) if rr else None
    components = {kind: [r for r in new if r['qtype'] == kind] for kind in ('single', 'compound')}
    old = [r for r in old if r['variant'] in ('open', 'support_masked')]
    old_hidden = [r for r in old if not r['answerable']]
    values = [good(components[k]) for k in components] + [sum(r['good'] for r in old) / len(old)]
    return {'macro': sum(values) / 3, 'components': dict(zip(('single', 'compound', 'hotpot'), values)),
            'supported_correct': good(answerable), 'legacy_good': sum(r['good'] for r in new) / len(new),
            'unsupported': unsupported(hidden), 'hidden_n': len(hidden),
            'unsupported_rate': unsupported(hidden) / len(hidden),
            'unsupported_by_family': {family: unsupported([r for r in hidden if ('partial' in r['variant']) == partial])
                                      for family, partial in [('partial', True), ('fully_missing', False)]},
            'invalid': sum(r['outcome'] == 'invalid' for r in new) / len(new),
            'hotpot_unsupported': unsupported(old_hidden), 'n': len(new)}


def key(row):
    return row['qid'], row['variant']


def choose(records):
    base = records['main']
    eligible = []
    for name in ('candE1', 'candE2', 'candF1', 'candF2'):
        if name not in records:
            raise ValueError(f'missing predeclared checkpoint {name}')
        r = records[name]
        gates = {'no_more_unsupported': r['unsupported'] <= base['unsupported'],
                 'no_worse_family': all(r['unsupported_by_family'][k] <= base['unsupported_by_family'][k] for k in base['unsupported_by_family']),
                 'at_most_five_percent': r['unsupported_rate'] <= .05,
                 'invalid': r['invalid'] <= base['invalid'] + .01,
                 'answerable': r['supported_correct'] >= base['supported_correct'],
                 'hotpot_success': r['components']['hotpot'] >= base['components']['hotpot'] - .03,
                 'hotpot_unsupported': r['hotpot_unsupported'] <= base['hotpot_unsupported']}
        r['gates'] = gates
        r['eligible'] = all(gates.values())
        if r['eligible']:
            eligible.append(name)
    if not eligible:
        return None
    zero_error = [n for n in eligible if records[n]['unsupported'] == 0 and records[n]['hotpot_unsupported'] == 0
                  and records[n]['macro'] >= base['macro'] + .03]
    if zero_error:
        eligible = zero_error
    best = max(records[n]['macro'] for n in eligible)
    tied = [n for n in eligible if records[n]['macro'] >= best - .005]
    winner = min(tied, key=lambda n: (records[n]['unsupported'], int(n[-1]), n.startswith('candF')))
    return winner if records[winner]['macro'] >= base['macro'] + .03 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='append', required=True, help='evaluation root, can repeat')
    ap.add_argument('--out', default='reports/round4_selection')
    args = ap.parse_args()
    paths = {}
    for root in args.eval:
        for path in Path(root).rglob('scores.jsonl'):
            if path.parent.name != 'lenient':
                continue
            run = path.parent.parent.name
            paths[run] = path
    records, reference_keys = {}, {}
    for name in ('main', 'candA', 'candE1', 'candE2', 'candF1', 'candF2'):
        new = read(paths[f'{name}_bench_r4dev'])
        old = read(paths[f'{name}_dev'])
        for split, rr in [('new', new), ('old', old)]:
            keys = [key(r) for r in rr]
            assert len(keys) == len(set(keys)), f'duplicate scored episode for {name}/{split}'
            if split in reference_keys:
                assert set(keys) == reference_keys[split], f'episode mismatch for {name}/{split}'
            else:
                reference_keys[split] = set(keys)
        records[name] = stats(new, old)
    winner = choose(records)
    result = {'records': records, 'selected': winner, 'headline': 'main',
              'selected_zero_unsupported_on_development': bool(winner and records[winner]['unsupported'] == 0
                                                               and records[winner]['hotpot_unsupported'] == 0),
              'note': 'Development selection only; release recommendation requires the frozen final holdout.'}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Round 4 development selection', '',
             'Primary success requires the original correctness check and all required evidence aspects cited.', '',
             '| system | macro | single | compound | Hotpot | unsupported / hidden | invalid | eligible |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name, r in records.items():
        lines.append(f"| {name} | {r['macro']:.3f} | {r['components']['single']:.3f} | {r['components']['compound']:.3f} | "
                     f"{r['components']['hotpot']:.3f} | {r['unsupported']}/{r['hidden_n']} | {r['invalid']:.3f} | {r.get('eligible', 'reference')} |")
    lines += ['', f"Selected for frozen holdout: **{winner or 'none; diagnostic candE1 only'}**.", '', result['note']]
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
