"""Pre-registered inference-time wrappers, scored on the round-4 final test (no new model calls).

Written 6 November 2026, 23:25 IST, before the bench_r4test traces were downloaded or seen.
Rule text: research/2026-11-06-round4-wrapper-preregistration.md. Nothing here may be tuned
after the test traces are read: the systems, the agreement threshold and the normalisation are fixed.

FB(x)   fail closed: if adapter x ends on an invalid or looping turn, return an abstention.
G(p, c) agreement gate: run adapters p and c independently on the same episode; return p's answer
        only if both answered and their normalised answers have token F1 >= 0.5; otherwise abstain.
"""

import argparse
import collections
import json
import re
import string
from pathlib import Path

from report_round4 import describe, paired_interval, read

THRESHOLD = 0.5


def norm(text):
    text = text or ''
    m = re.search(r'<answer>(.*?)</answer>', text, re.S)
    text = m.group(1) if m else text
    text = re.sub(r'\[?chunk_id=\w+\]?|\[c\d+\]|\bc\d{5}\b', ' ', text).lower()
    text = ''.join(' ' if ch in string.punctuation else ch for ch in text)
    return [t for t in text.split() if t not in ('a', 'an', 'the', 'and')]


def token_f1(a, b):
    common = sum((collections.Counter(a) & collections.Counter(b)).values()) if a and b else 0
    if not common:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def abstain(r):
    return {**r, 'final': 'abstain', 'outcome': 'false_abstain' if r['answerable'] else 'correct_abstain',
            'good': not r['answerable'], 'evidence_supported_good': not r['answerable']}


def load(folder):
    scores = read(folder / 'scores.jsonl')
    answers = {(t['metadata']['qid'], t['metadata']['variant']): t['final_answer'] for t in read(folder / 'trajectories.jsonl')}
    keys = [(r['qid'], r['variant']) for r in scores]
    assert len(keys) == len(set(keys)) == len(answers) and set(keys) == set(answers), folder
    return scores, answers


def fail_closed(system):
    scores, _ = system
    return [r if r['final'] in ('answer', 'abstain') else abstain(r) for r in scores]


def gate(primary, checker):
    (ps, pa), (cs, ca) = primary, checker
    cs = {(r['qid'], r['variant']): r for r in cs}
    out = []
    for r in ps:
        k = (r['qid'], r['variant'])
        agree = r['final'] == 'answer' and cs[k]['final'] == 'answer' and token_f1(norm(pa[k]), norm(ca[k])) >= THRESHOLD
        out.append(r if agree else abstain(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True, help='folder holding main_bench_r4test/ and candA_bench_r4test/')
    ap.add_argument('--out', default='reports/round4_wrappers')
    args = ap.parse_args()
    root = Path(args.eval)
    main_, cand = load(root / 'main_bench_r4test/lenient'), load(root / 'candA_bench_r4test/lenient')
    systems = {'main': main_[0], 'candA': cand[0], 'FB(main)': fail_closed(main_), 'FB(candA)': fail_closed(cand),
               'G(main, candA)': gate(main_, cand)}
    d = {k: describe(v) for k, v in systems.items()}
    base = systems['main']
    macro = lambda rr: describe(rr)['macro']
    unsafe = lambda rr: describe(rr)['unsupported_rate']
    supported = lambda rr: describe(rr)['supported_success']
    ci = {k: {'macro': paired_interval(base, systems[k], macro), 'unsupported_rate': paired_interval(base, systems[k], unsafe),
              'supported_success': paired_interval(base, systems[k], supported)} for k in ('FB(main)', 'G(main, candA)')}
    fb, g = d['FB(main)'], d['G(main, candA)']
    claims = {
        'FB(main) raises the score without more unsupported answers':
            ci['FB(main)']['macro'][0] > 0 and fb['unsupported'] <= d['main']['unsupported'],
        'G(main, candA) lowers the unsupported-answer rate': ci['G(main, candA)']['unsupported_rate'][1] < 0,
        'G(main, candA) does not lower the score': ci['G(main, candA)']['macro'][0] >= 0,
    }
    report = {'systems': d, 'ci95_vs_main': ci, 'claims': claims, 'threshold': THRESHOLD,
              'note': 'Pre-registered before the test traces were read. Inference-time wrappers, not new adapters. '
                      'They do not change the headline automatically.'}
    out = Path(args.out)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Round 4 pre-registered wrappers on the final test', '',
             '| system | macro evidence success | single | compound | supported success | unsupported / hidden | false refusals | invalid |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k, r in d.items():
        lines.append(f"| {k} | {r['macro']:.3f} | {r['components']['single']:.3f} | {r['components']['compound']:.3f} | "
                     f"{r['supported_success']:.3f} | {r['unsupported']}/{r['hidden_n']} | {r['false_abstain']:.3f} | {r['invalid']:.3f} |")
    lines += ['', 'Differences against main, 95% intervals grouped by question:', '']
    for k, v in ci.items():
        lines.append(f"- {k}: macro {d[k]['macro'] - d['main']['macro']:+.3f} {v['macro']}; unsupported rate "
                     f"{d[k]['unsupported_rate'] - d['main']['unsupported_rate']:+.3f} {v['unsupported_rate']}; supported success "
                     f"{d[k]['supported_success'] - d['main']['supported_success']:+.3f} {v['supported_success']}")
    lines += ['', 'Pre-registered claims:', ''] + [f'- {k}: **{v}**' for k, v in claims.items()] + ['', report['note']]
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
