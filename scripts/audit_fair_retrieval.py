"""Replay saved Round 5 searches under the current hosted-evaluation job specs.

This checks recorded queries only; it cannot prove retrieval equivalence for new queries.
It never reads Round 6 test data or calls a model.
"""
import json
from collections import Counter
from pathlib import Path

import azure_rag_eval as azure
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = {}
    for name in ('r5t_squad', 'r5t_hotpot', 'r5t_rgb'):
        corp, backend, jobs, _ = azure.build_jobs(name)
        by_id = {(j.q.qid, j.variant): j for j in jobs}
        path = azure.R5 / f'main_bench_{name}/lenient/transcripts.jsonl'
        rows = azure.read_jsonl(path)
        counts = Counter()
        for row in rows:
            job = by_id[(row['qid'], row['variant'])]
            if row['question'] != job.q.question:
                raise ValueError('Reference question text differs from the current question')
            tool = CorpusSearchTool(backend, top_k=3)
            prefill = job.spec.prefill or []
            actual = [(h.chunk_id, h.title) for h in prefill]
            expected = [(h['chunk_id'], h['title']) for h in row['provided_evidence']]
            if actual != expected:
                counts['different_prefill_episodes'] += 1
            tool.start_episode(job.spec.masked, job.spec.first_search_masked,
                               frozenset(h.chunk_id for h in prefill))
            changed = False
            for step in row['steps']:
                if step['kind'] != 'search':
                    continue
                counts['searches'] += 1
                if tool.execute(query=step['query']).output != step['observation']:
                    counts['different_searches'] += 1
                    changed = True
            counts['episodes'] += 1
            counts['episodes_with_different_search'] += int(changed)
        out[name] = {k: counts[k] for k in ('episodes', 'searches', 'different_searches',
                    'episodes_with_different_search', 'different_prefill_episodes')}
    report = {'suites': out, 'scope': 'Replayed recorded main-model search queries and checked prefill IDs and titles. '
              'This cannot establish equivalence on new queries or prove identical masks.'}
    (ROOT / 'reports/fair_retrieval_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if any(v['different_searches'] or v['different_prefill_episodes'] for v in out.values()):
        raise SystemExit('The current harness differs on recorded reference searches or prefilled evidence.')


if __name__ == '__main__':
    main()
