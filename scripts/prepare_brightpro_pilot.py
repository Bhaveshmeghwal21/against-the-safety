"""Evaluation only: fixed official robotics query subset and its full retrieval corpus."""

import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from ragharness import corpus as C

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'dbdc22babbef310210e267b99249e7cec86d5edf'
IDS = ['4', '7', '8', '17', '18', '27', '29', '31', '33', '34']


def main():
    out = ROOT / 'data/brightpro/robotics'
    out.mkdir(parents=True, exist_ok=True)
    def load(kind):
        path = hf_hub_download('yale-nlp/Bright-Pro', f'{kind}/robotics.parquet', repo_type='dataset',
                               revision=REVISION, cache_dir=str(ROOT / '.cache/huggingface/hub'), token=False)
        return pq.read_table(path).to_pylist()
    docs = sorted(load('documents'), key=lambda d: str(d['id']))
    random.Random(47).shuffle(docs)
    chunks, mapping = [], {}
    digest = hashlib.sha256()
    for i, doc in enumerate(docs):
        cid = f'c{i + 1:05d}'
        title, text = 'Document ' + cid, doc['content']
        chunks.append({'chunk_id': cid, 'doc_id': str(doc['id']), 'title': title, 'text': text, 'chunk_index': 0})
        mapping[cid] = str(doc['id'])
        digest.update(f'{cid}\t{title}\t{text}\n'.encode())
    assert len(chunks) == 63920 and all(len(c['chunk_id']) == 6 for c in chunks)
    corp = C.Corpus(chunks, {}, digest.hexdigest())
    C.save_corpus(out / 'corpus.json', corp)
    C.build_index(corp, out / 'corpus.sqlite')
    examples = {str(r['id']): r for r in load('examples')}
    all_aspects = load('aspects')
    tasks = []
    for qid in IDS:
        aspects = [a for a in all_aspects if a['id'].startswith(f'robotics-{qid}-a')]
        assert aspects
        weight = sum(a['weight'] for a in aspects)
        tasks.append({'id': qid, 'query': examples[qid]['query'],
                      'aspects': [{**a, 'normalized_weight': a['weight'] / weight} for a in aspects]})
    (out / 'tasks.json').write_text(json.dumps(tasks, ensure_ascii=False), encoding='utf-8')
    (out / 'doc_ids.json').write_text(json.dumps(mapping), encoding='utf-8')
    manifest = {'dataset': 'yale-nlp/Bright-Pro', 'revision': REVISION, 'domain': 'robotics', 'query_ids': IDS,
                'n_documents': len(chunks), 'index_hash': corp.index_hash, 'license': 'MIT (dataset card)',
                'training_use': False, 'protocol': 'custom evaluation-only: lexical search, top3, three searches, 800 characters',
                'source_sample': 'https://github.com/yale-nlp/Bright-Pro/blob/main/agentic_sample_ids.json'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
