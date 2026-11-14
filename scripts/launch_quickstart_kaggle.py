"""Run the checked-in quickstart notebook on the approved private Kaggle GPU.

Uses the existing development-only dataset; does not publish a dataset version.
Credentials are loaded only by kaggle_launch.creds().
"""
import base64
import json
import os
from pathlib import Path

import kaggle_launch as launch

ROOT = Path(__file__).resolve().parents[1]
launch.BUNDLE = ROOT / 'runs/kaggle_bundle_round6_dev_only'
os.environ['ATS_KAGGLE_CREDENTIALS_ENV'] = 'kaggleLegacyToken#2'


def main():
    if launch.served_bundle_id() != launch.bundle_id():
        raise SystemExit('The private development bundle served by Kaggle does not match the local bundle.')
    notebook = (ROOT / 'notebooks/quickstart.ipynb').read_bytes()
    encoded = base64.b64encode(notebook).decode('ascii')
    remote = '''import base64, importlib.util, os, pathlib, subprocess, sys\n
p = next(pathlib.Path('/kaggle/input').rglob('bootstrap.py'))
spec = importlib.util.spec_from_file_location('bootstrap', p)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.OUT.mkdir(parents=True, exist_ok=True)
b.unpack()
py = b.python_for_env()
b.install(py)
os.chdir(b.REPO)
os.environ.update(PYTHONUTF8='1', HF_HOME=str(b.TMP / 'hf'), TOKENIZERS_PARALLELISM='false', TINY_AYA_PATH=b.find_model(), CUDA_VISIBLE_DEVICES='0')
subprocess.run([py, '-m', 'pip', 'install', '-q', 'nbclient', 'nbformat', 'ipykernel'], check=True)
subprocess.run([py, '-m', 'ipykernel', 'install', '--prefix', str(pathlib.Path(py).parent.parent), '--name', 'python3'], check=True)
path = b.REPO / 'notebooks' / 'quickstart.ipynb'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(base64.b64decode(NOTEBOOK))
runner = """import nbformat, sys\nfrom nbclient import NotebookClient\np=sys.argv[1]; out=sys.argv[2]\nn=nbformat.read(p, as_version=4)\ntry:\n NotebookClient(n, timeout=1800, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()\nfinally:\n nbformat.write(n, out)\n"""
subprocess.run([py, '-c', runner, str(path), str(b.OUT / 'quickstart-executed.ipynb')], check=True)
print('Quickstart completed; executed notebook saved.', flush=True)
'''
    remote = 'NOTEBOOK = ' + repr(encoded) + '\n' + remote
    folder = ROOT / 'runs/kaggle/push-ats-quickstart-check'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'run.py').write_text(remote, encoding='utf-8')
    meta = {'id': f'{launch.username()}/ats-quickstart-check', 'title': 'ats-quickstart-check',
            'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': [f'{launch.username()}/{launch.SLUG}'], 'competition_sources': [],
            'kernel_sources': [], 'model_sources': [launch.MODEL_SOURCE]}
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    launch.kaggle('kernels', 'push', '-p', str(folder), '--accelerator', 'NvidiaTeslaT4', '--timeout', '3600')
    print('Private quickstart GPU kernel submitted.')


if __name__ == '__main__':
    main()
