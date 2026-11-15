"""Stage the exact saved candF1 answer cases for an authorized verifier-only retry."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/kaggle/ats-r6-system-candf1-xq/out'
OUT = ROOT / 'runs/kaggle_bundle_candf1_checker_retry'


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    cases_path = RUN / 'cases/answers.jsonl'
    cases = read(cases_path)
    by_id = {(r['run'], r['qid'], r['variant']): r for r in cases}
    if len(by_id) != len(cases) or len(cases) != 104:
        raise ValueError('Unexpected number of unique saved checker cases')
    expected = {}
    for lang in ('hi', 'en', 'es'):
        run = f'candF1_bench_xq_{lang}'
        transcripts = read(RUN / 'eval' / run / 'lenient/transcripts.jsonl')
        for t in transcripts:
            if t['final']['kind'] == 'answer':
                key = (run, t['qid'], t['variant'])
                if key in expected:
                    raise ValueError('Duplicate answer transcript')
                expected[key] = t
    if set(expected) != set(by_id):
        raise ValueError('Checker cases do not match the complete saved answer transcripts')
    for key, case in by_id.items():
        t = expected[key]
        if (case['question'] != t['question'] or case['answer'] != t['final']['answer']
                or case['citations'] != t['final']['citations'] or not case['passages']):
            raise ValueError('A saved checker case differs from its answer transcript')
    verifier = ROOT / 'scripts/verify_round5.py'
    model = ROOT / 'runs/kaggle/ats-train-both/out/main/adapter/adapter_model.safetensors'
    driver = ROOT / 'kaggle/candf1_verifier_retry.py'
    payload = {'cases/answers.jsonl': cases_path.read_bytes(), 'retry/verify_round5.py': verifier.read_bytes(),
               'retry/driver.py': driver.read_bytes()}
    sha = lambda b: hashlib.sha256(b).hexdigest()
    manifest = {'cases': len(cases), 'cases_sha256': sha(payload['cases/answers.jsonl']),
                'verifier_sha256': sha(payload['retry/verify_round5.py']), 'driver_sha256': sha(payload['retry/driver.py']),
                'main_adapter_sha256': sha(model.read_bytes()),
                'source_bundle_id': 'd306a67f543870712632', 'batch_size': 1, 'last_logits_only': True,
                'wording': 'A', 'threshold': 0.5, 'scope': 'Saved candF1 XQuAD cases only; no model answer generation.'}
    payload['CHECKER_RETRY_MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with zipfile.ZipFile(OUT / 'against-the-safety.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(payload.items()):
            z.writestr(name, data)
            digest.update(name.encode() + b'\0' + hashlib.sha256(data).digest())
        z.writestr('BUNDLE_ID', digest.hexdigest()[:20])
    (OUT / 'dataset-metadata.json').write_text(json.dumps({'title': 'ats-candf1-checker-retry',
        'id': 'REPLACE_WITH_YOUR_KAGGLE_USERNAME/ats-candf1-checker-retry', 'licenses': [{'name': 'CC-BY-SA-4.0'}]}, indent=2), encoding='utf-8')
    print('Prepared private verifier-only bundle:', len(cases), 'unchanged answer cases; ID', digest.hexdigest()[:20])


if __name__ == '__main__':
    main()
