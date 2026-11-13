"""Build checker inputs for the forced continuations, recording invalid output counts."""
import argparse
import json
from collections import Counter
from pathlib import Path

import rescore_round5 as RS
from ragharness import checker, protocol
from ragharness.retrieval import Hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contexts', required=True)
    ap.add_argument('--forced', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    contexts = {r['id']: r for r in RS.read(args.contexts)}
    forced = RS.read(args.forced)
    if len({r['id'] for r in forced}) != len(forced):
        raise ValueError('Duplicate forced-answer IDs')
    cases, counts = [], Counter()
    for item in forced:
        if item['id'] not in contexts:
            raise ValueError('Forced answer does not have a matching context')
        ctx = contexts[item['id']]
        turn, _ = protocol.parse_turn_lenient(item['text'])
        if turn.kind != 'answer':
            counts['not_an_answer'] += 1
            continue
        if not turn.citations:
            counts['no_citations'] += 1
            continue
        corp, _ = RS.world(ctx['set'])
        transcript = ctx['transcript']
        ids = [x['chunk_id'] for x in transcript.get('provided_evidence', [])]
        ids += [h['chunk_id'] for s in transcript['steps'][:-1] for h in s['retrieved']]
        hits = []
        for cid in dict.fromkeys(ids):
            chunk = corp.chunk(cid)
            hits.append(Hit(cid, chunk['title'], chunk['text'], 0.0))
        flags = checker.deterministic_flags(turn.answer or '', turn.citations, hits, transcript.get('max_chars', 800))
        cases.append({'run': ctx['run'] + '_forced', 'world': ctx['set'], 'qid': ctx['qid'], 'variant': ctx['variant'],
                      'question': transcript['question'], 'answer': turn.answer, 'citations': turn.citations,
                      'passages': flags.pop('passages'), 'deterministic': flags, 'id': item['id']})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in cases), encoding='utf-8')
    summary = {'contexts': len(contexts), 'forced': len(forced), 'cases': len(cases),
               'missing_forced': len(contexts) - len(forced), 'rejected': dict(counts)}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
