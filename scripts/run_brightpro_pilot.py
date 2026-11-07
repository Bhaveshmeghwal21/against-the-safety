"""Measure official-label aspect recall without using references or labels as actor inputs."""

import argparse
import json
import time
from pathlib import Path

from ragharness import builder, corpus as C, data, episodes, evaluation, generate, modelref, protocol
from ragharness.retrieval import open_verified_backend
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    folder = ROOT / 'data/brightpro/robotics'
    tasks = json.loads((folder / 'tasks.json').read_text(encoding='utf-8'))
    doc_ids = json.loads((folder / 'doc_ids.json').read_text())
    corp = C.load_corpus(folder / 'corpus.json')
    backend = open_verified_backend(folder / 'corpus.sqlite', corp)
    jobs = [evaluation.Job(data.Question('brightpro-robotics-' + t['id'], t['query'], '', 'brightpro', 'transfer', [], {}),
                           'open', builder.Spec('open'), []) for t in tasks]
    # The acting model receives only the query and ordinary search observations.
    model_path, revision, _ = modelref.resolve()
    model, tok = generate.load_model(model_path, revision, adapter=args.adapter)
    gen = generate.HFGenerator(model, tok)
    tool = CorpusSearchTool(backend, top_k=3)
    start = time.time()
    results = evaluation.run_jobs_batched(jobs, corp, lambda: CorpusSearchTool(backend, top_k=3),
                                          protocol.system_prompt(tool), gen, 3, 12, progress=5, lenient=True)
    output = []
    for task, (_, ep, cost) in zip(tasks, results):
        retrieved = {doc_ids[h.chunk_id] for h in ep.retrieved_chunks()}
        shown = {h.chunk_id for h in ep.retrieved_chunks()}
        citations = {doc_ids[c] for c in (ep.final.turn.citations or []) if c in shown and c in doc_ids}
        coverage = lambda ids: sum(a['normalized_weight'] for a in task['aspects'] if set(a['supporting_docs']) & ids)
        by_search, ids = {}, set()
        for step in ep.steps:
            if step.turn.kind == 'search':
                ids.update(doc_ids[h.chunk_id] for h in step.hits)
                by_search[str(len(by_search) + 1)] = coverage(ids)
        output.append({'id': task['id'], 'weighted_aspect_recall': coverage(retrieved),
                       'cited_aspect_recall': coverage(citations), 'by_search': by_search,
                       'final_kind': ep.final.turn.kind, 'answer': ep.final.turn.answer,
                       'searches': ep.n_searches, 'retrieved_docids': sorted(retrieved),
                       'steps': [{'assistant': s.text, 'observation': s.observation} for s in ep.steps], **cost})
    report = {'n': len(output), 'weighted_aspect_recall': sum(r['weighted_aspect_recall'] for r in output) / len(output),
              'cited_aspect_recall': sum(r['cited_aspect_recall'] for r in output) / len(output),
              'invalid': sum(r['final_kind'] == 'invalid' for r in output), 'truncated_inputs': gen.truncated_inputs,
              'wall_seconds': time.time() - start, 'adapter': args.adapter,
              'note': '10-query custom robotics transfer diagnostic, not the official 175-query benchmark; no answer-quality judge'}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scores.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in output), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
