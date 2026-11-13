"""Reconstruct the precise prompt preceding each recoverable refusal."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import rescore_round5 as RS
from ragharness import protocol
from ragharness.retrieval import Hit, render_observation
from ragharness.tools import CorpusSearchTool


def contexts(root, systems):
    if not Path(root).exists():
        raise FileNotFoundError(root)
    system_prompt = protocol.system_prompt(CorpusSearchTool(None))
    found = []
    for folder in sorted(Path(root).rglob('lenient')):
        path = folder / 'transcripts.jsonl'
        if not path.exists():
            continue
        run = folder.parent.name
        system, world = RS.kind_of(run)
        if system not in systems:
            continue
        corp, _ = RS.world(world)
        for row in RS.read(path):
            if row['final']['kind'] not in ('abstain', 'invalid'):
                continue
            steps = row['steps'][:-1]
            ids = [h['chunk_id'] for h in row.get('provided_evidence', [])]
            ids += [h['chunk_id'] for s in steps for h in s['retrieved']]
            if not ids:
                continue
            prefill = []
            for item in row.get('provided_evidence', []):
                chunk = corp.chunk(item['chunk_id'])
                prefill.append(Hit(item['chunk_id'], chunk['title'], chunk['text'], 0.0))
            user = row['question']
            if prefill:
                user += '\n\nEvidence already provided:\n' + render_observation(prefill, max_chars=row.get('max_chars', 800))
            prompt = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user}]
            for step in steps:
                if step['kind'] != 'search' or step['observation'] is None:
                    raise ValueError(f'Cannot reconstruct prior turn for {run}/{row["qid"]}')
                prompt += [{'role': 'assistant', 'content': step['assistant']},
                           {'role': 'user', 'content': protocol.format_tool_result(step['observation'])}]
            ident = hashlib.sha256(f'{run}\0{row["qid"]}\0{row["variant"]}'.encode()).hexdigest()[:24]
            found.append({'id': ident, 'system': system, 'set': world, 'run': run, 'qid': row['qid'],
                          'variant': row['variant'], 'depth': len(steps), 'prompt': prompt, 'transcript': row})
    if len({r['id'] for r in found}) != len(found):
        raise ValueError('Duplicate recovery episode identifiers')
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True)
    ap.add_argument('--system', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    found = contexts(args.eval, set(args.system.split(',')))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in found), encoding='utf-8')
    print(json.dumps({'contexts': len(found), 'by_run': dict(Counter(r['run'] for r in found))}))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
