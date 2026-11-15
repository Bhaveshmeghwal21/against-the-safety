"""Stage or launch the prepared candF1 verifier-only retry.

Running without --launch only writes the private kernel files locally. Upload the
prepared retry dataset and obtain the user's explicit approval before --launch.
"""
import argparse
import json
import os
from pathlib import Path

import kaggle_launch as launch

ROOT = Path(__file__).resolve().parents[1]
RETRY = ROOT / 'runs/kaggle_bundle_candf1_checker_retry'
SOURCE = ROOT / 'runs/kaggle_bundle_candf1_xquad'
RETRY_SLUG = 'ats-candf1-checker-retry'
SOURCE_SLUG = 'ats-candf1-xquad-bundle'
KERNEL_SLUG = 'ats-candf1-xquad-v-retry'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true', help='push and run the prepared private kernel after explicit approval')
    args = parser.parse_args()
    os.environ['ATS_KAGGLE_CREDENTIALS_ENV'] = 'kaggleLegacyToken#2'
    launch.BUNDLE = RETRY
    expected = launch.bundle_id()
    owner = launch.username()
    folder = ROOT / 'runs/kaggle/push-ats-candf1-xquad-v-retry'
    folder.mkdir(parents=True, exist_ok=True)
    code = ('import pathlib, runpy\n'
            "root = next(pathlib.Path('/kaggle/input').rglob('CHECKER_RETRY_MANIFEST.json')).parent\n"
            f"if (root / 'BUNDLE_ID').read_text().strip() != {expected!r}:\n"
            "    raise SystemExit('Stale retry dataset mount')\n"
            "runpy.run_path(str(root / 'retry/driver.py'), run_name='__main__')\n")
    (folder / 'run.py').write_text(code, encoding='utf-8')
    meta = {'id': f'{owner}/{KERNEL_SLUG}', 'title': KERNEL_SLUG, 'code_file': 'run.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [f'{owner}/{SOURCE_SLUG}', f'{owner}/{RETRY_SLUG}'],
            'competition_sources': [], 'kernel_sources': [], 'model_sources': [launch.MODEL_SOURCE]}
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if not args.launch:
        print('Prepared the private verifier-only kernel locally; nothing was uploaded or launched.')
        return
    for slug, bundle in ((SOURCE_SLUG, SOURCE), (RETRY_SLUG, RETRY)):
        launch.SLUG, launch.BUNDLE = slug, bundle
        if launch.served_bundle_id() != launch.bundle_id():
            raise SystemExit('The private dataset on Kaggle is missing or stale: ' + slug)
    print(launch.kaggle('kernels', 'push', '-p', str(folder), '--accelerator', 'NvidiaTeslaT4', '--timeout', '3600'))


if __name__ == '__main__':
    main()
