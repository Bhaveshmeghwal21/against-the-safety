"""GPU-side verifier-only retry for the saved candF1 XQuAD answer cases."""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


retry_root = Path(__file__).resolve().parent.parent
manifest = json.loads((retry_root / 'CHECKER_RETRY_MANIFEST.json').read_text(encoding='utf-8'))
cases = retry_root / 'cases/answers.jsonl'
verifier = retry_root / 'retry/verify_round5.py'
if (sha(cases) != manifest['cases_sha256'] or sha(verifier) != manifest['verifier_sha256']
        or sha(Path(__file__).resolve()) != manifest['driver_sha256']):
    raise SystemExit('Retry data or verifier code does not match its manifest')
source_marker = next(Path('/kaggle/input').rglob('XQUAD_CANDF1_ONLY'))
if (source_marker.parent / 'BUNDLE_ID').read_text().strip() != manifest['source_bundle_id']:
    raise SystemExit('The original candidate input bundle has changed')
bootstrap = source_marker.parent / 'kaggle/bootstrap.py'
spec = importlib.util.spec_from_file_location('original_bootstrap', bootstrap)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.OUT.mkdir(parents=True, exist_ok=True)
b.unpack()
adapter = b.REPO / 'initial_adapters/main/adapter'
if sha(adapter / 'adapter_model.safetensors') != manifest['main_adapter_sha256']:
    raise SystemExit('The main verifier adapter is not the registered adapter')
rows = [json.loads(line) for line in cases.read_text(encoding='utf-8').splitlines() if line.strip()]
keys = {(r['run'], r['qid'], r['variant']) for r in rows}
allowed = {f'candF1_bench_xq_{lang}' for lang in ('hi', 'en', 'es')}
if len(rows) != manifest['cases'] or len(keys) != len(rows) or any(r['run'] not in allowed for r in rows):
    raise SystemExit('The saved checker cases differ from the registered scope')
py = b.python_for_env()
b.install(py)
os.chdir(b.REPO)
os.environ.update(PYTHONUTF8='1', HF_HOME=str(b.TMP / 'hf'), TOKENIZERS_PARALLELISM='false',
                  TINY_AYA_PATH=b.find_model(), CUDA_VISIBLE_DEVICES='0')
out = b.OUT / 'verify/V'
subprocess.run([py, str(verifier), '--cases', str(cases), '--adapter', str(adapter), '--tag', 'V',
                '--out', str(out), '--wordings', 'A', '--batch-size', '1', '--last-logits-only'], check=True)
verdicts = [json.loads(line) for line in (out / 'verdicts_A.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
verdict_keys = {(r['run'], r['qid'], r['variant']) for r in verdicts}
if len(verdicts) != len(rows) or verdict_keys != keys or any(not math.isfinite(r['p_pos']) or not 0 <= r['p_pos'] <= 1 for r in verdicts):
    raise SystemExit('Incomplete, duplicate or invalid verifier verdicts')
(b.OUT / 'retry_integrity.json').write_text(json.dumps({'cases': len(rows), 'verdicts': len(verdicts),
    'cases_sha256': sha(cases), 'verifier_sha256': sha(verifier), 'main_adapter_sha256': manifest['main_adapter_sha256'],
    'batch_size': 1, 'last_logits_only': True}, indent=2) + '\n', encoding='utf-8')
print('Verifier-only retry completed:', len(verdicts), 'verdicts.', flush=True)
