"""Prepare actual actor-visible traces for human review; never select a checkpoint.

The deterministic review includes every candidate answer on hidden-evidence cases,
every answerable answer with a citation/coverage failure, twenty stratified strong
successes, and five paired wins/losses. Scores alone cannot prove entailment.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load(root, name, split):
    folder = root / f'{name}_{split}' / 'lenient'
    scores, transcripts, logs = [read(folder / f'{kind}.jsonl') for kind in ('scores', 'transcripts', 'eventlogs')]
    assert len(scores) == len(transcripts) == len(logs)
    records = {}
    for score, transcript, log in zip(scores, transcripts, logs):
        key = score['qid'], score['variant']
        assert key == (transcript['qid'], transcript['variant']) and key not in records
        records[key] = {'score': score, 'transcript': transcript, 'events': log['events']}
    return records


def select_review(main, candidate):
    assert set(main) == set(candidate)
    selected = {}

    def add(keys, reason):
        for key in keys:
            selected.setdefault(key, []).append(reason)

    def sample(keys, count):
        keys = sorted(keys)
        return random.Random(47).sample(keys, min(count, len(keys)))

    add([k for k, r in candidate.items() if r['score']['outcome'].startswith('unsupported_')], 'hidden-evidence answer')
    add([k for k, r in candidate.items() if r['score']['answerable'] and r['score']['final'] == 'answer'
         and (not r['score']['cited_all_shown'] or not r['score']['all_aspects_cited'])], 'citation/coverage failure')
    for kind in ('single', 'compound'):
        add(sample([k for k, r in candidate.items() if r['score']['answerable']
                    and r['score']['qtype'] == kind and r['score']['evidence_supported_good']], 10),
            f'seeded {kind} supported-success sample')
    for good, label in ((True, 'paired win'), (False, 'paired loss')):
        add(sample([k for k in candidate if candidate[k]['score']['evidence_supported_good'] == good
                    and main[k]['score']['evidence_supported_good'] != good], 5), label)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', type=Path, required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--split', default='bench_r4test')
    ap.add_argument('--out', type=Path, default=Path('runs/round4_review'))
    args = ap.parse_args()
    systems = {n: load(args.eval, n, args.split) for n in ('main', args.candidate)}
    chosen = select_review(systems['main'], systems[args.candidate])
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for number, (key, reasons) in enumerate(sorted(chosen.items()), 1):
        path = args.out / f'case_{number:03d}.json'
        path.write_text(json.dumps({'qid': key[0], 'variant': key[1], 'reasons': reasons,
                                    'systems': {n: rr[key] for n, rr in systems.items()}},
                                   ensure_ascii=False, indent=2), encoding='utf-8')
        index.append({'file': path.name, 'qid': key[0], 'variant': key[1], 'reasons': reasons})
    report = {'candidate': args.candidate, 'split': args.split, 'cases': index,
              'reason_counts': dict(Counter(reason for reasons in chosen.values() for reason in reasons)),
              'note': 'Unreviewed trace packet, not an automatic semantic-accuracy claim or checkpoint selector.'}
    (args.out / 'index.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'cases': len(index), 'reason_counts': report['reason_counts'], 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
