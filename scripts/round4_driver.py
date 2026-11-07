"""Finish the explicitly authorized, bounded round-4 run; no test-driven checkpoint search.

Resume by rerunning: completed Kaggle stages are read, not launched again. A failed
stage stops for inspection. The selected checkpoint is determined only by the frozen
development selector. Credentials are supplied via the existing account helper.
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import kaggle_launch as launch

ROOT = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30))
STATE = ROOT / 'runs/round4_state.json'


def update(**values):
    current = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    if values.get('phase') and values['phase'] != 'needs inspection':
        current.pop('error', None)
    current.update(values, updated_at=datetime.now(IST).isoformat())
    STATE.write_text(json.dumps(current, indent=2), encoding='utf-8')
    print(json.dumps({k: current[k] for k in values} | {'updated_at': current['updated_at']}), flush=True)


def status(slug):
    return launch.kaggle('kernels', 'status', f'{launch.username()}/{slug}', check=False)


def wait(slug, max_hours=5):
    started = time.monotonic()
    previous = None
    while time.monotonic() - started < max_hours * 3600:
        value = status(slug)
        if value != previous:
            print(value, flush=True)
            update(active_kernel=slug, kernel_status=value.splitlines()[-1])
            previous = value
        if 'KernelWorkerStatus.COMPLETE' in value:
            return
        if any(s in value for s in ('KernelWorkerStatus.ERROR', 'KernelWorkerStatus.CANCEL')):
            raise RuntimeError(f'{slug} failed; inspect its log before any retry')
        time.sleep(30)
    raise RuntimeError(f'{slug} did not complete inside the monitoring ceiling')


def ensure(stage, tag, sources, extra, timeout):
    slug = f'ats-{stage}-{tag}'
    value = status(slug)
    if 'KernelWorkerStatus.COMPLETE' not in value and 'KernelWorkerStatus.RUNNING' not in value and 'KernelWorkerStatus.QUEUED' not in value:
        if 'KernelWorkerStatus.ERROR' in value or 'KernelWorkerStatus.CANCEL' in value:
            raise RuntimeError(f'existing {slug} failed; automatic relaunch is disabled')
        # Kaggle 2.2.4 rewrites a missing private slug into a generic permission
        # message with no HTTP code. Verify absence in this account's own list
        # before creation; do not treat every permission error as a missing job.
        listing = launch.kaggle('kernels', 'list', '--mine', '--page-size', '100', '--csv')
        lines = listing.splitlines()
        first = next((i for i, line in enumerate(lines) if line.startswith('ref,')), None)
        if first is None:
            raise RuntimeError(f'cannot verify own notebook list before creating {slug}')
        refs = {r['ref'] for r in csv.DictReader(io.StringIO('\n'.join(lines[first:]))) if r.get('ref')}
        if f'{launch.username()}/{slug}' in refs or len(refs) >= 100:
            raise RuntimeError(f'cannot safely interpret status for existing or unlisted {slug}: {value}')
        launch.run_stage(stage, sources, extra, tag, timeout)
    wait(slug)
    return slug


def download(slug, pattern='.*(json|jsonl|log|md)$'):
    dest = ROOT / 'runs/account2_round4' / slug
    dest.mkdir(parents=True, exist_ok=True)
    output = launch.kaggle('kernels', 'output', f'{launch.username()}/{slug}', '-p', str(dest), '--file-pattern', pattern)
    print(output[-1800:], flush=True)
    return dest


def local(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--credentials-env', default='kaggleLegacyToken#2')
    ap.add_argument('--bundle-dir', default='runs/kaggle_bundle_account2')
    args = ap.parse_args()
    os.environ['ATS_KAGGLE_CREDENTIALS_ENV'] = args.credentials_env
    launch.BUNDLE = (ROOT / args.bundle_dir).resolve()
    train_slug = 'ats-train-evidence-r4'
    update(phase='training', active_kernel=train_slug, training_bundle='e9cb19cf02b90cc450cf')
    wait(train_slug)
    training = download(train_slug)
    for name in ('candE', 'candF'):
        report = json.loads((training / 'out' / name / 'train_report.json').read_text())
        if report['rows_trained'] != 721 or report['rows_selected'] != 721 or report['optimizer_steps'] < 2:
            raise RuntimeError(f'incomplete training report for {name}')
    update(phase='development evaluation', training_reports=str(training))
    names = ('main', 'candA', 'candE1', 'candE2', 'candF1', 'candF2')
    jobs = ','.join(f'{n}:{split}:{weight}' for n in names for split, weight in (('bench_r4dev', 35), ('dev', 17)))
    dev_slug = ensure('eval-jobs', 'r4dev', train_slug, '--jobs ' + jobs + ' --parse lenient --extra-questions 0', 14400)
    dev = download(dev_slug)
    local('scripts/select_round4.py', '--eval', str(dev / 'out/eval'))
    selection = json.loads((ROOT / 'reports/round4_selection.json').read_text(encoding='utf-8'))
    candidate = selection['selected'] or 'candE1'
    update(phase='frozen final test', selected=selection['selected'], evaluated_candidate=candidate)
    jobs = ','.join(f'{n}:bench_r4test:{15 if n == "base" else 45}' for n in ('main', 'candA', candidate, 'base'))
    test_slug = ensure('eval-jobs', 'r4test', train_slug, '--jobs ' + jobs + ' --parse lenient --extra-questions 0', 10800)
    final = download(test_slug)
    local('scripts/report_round4.py', '--eval', str(final / 'out/eval'))
    result = json.loads((ROOT / 'reports/round4_test.json').read_text(encoding='utf-8'))
    if result['recommended_default'] != 'main':
        update(phase='informed safety transfer checks')
        sizes = {'hotpot': 88, '2wiki': 86, 'musique': 67, 'squad2': 166, 'rgb': 266, 'rgb_int': 66}
        jobs = ','.join(f'{n}:bench_r3t_{dataset}:{weight}' for n in ('main', candidate) for dataset, weight in sizes.items())
        informed_slug = ensure('eval-jobs', 'r4transfer', train_slug, '--jobs ' + jobs + ' --parse lenient --extra-questions 0', 10800)
        informed = download(informed_slug)
        local('scripts/audit_round4_transfer.py', '--eval', str(informed / 'out/eval'), '--candidate', candidate)
        check = json.loads((ROOT / 'reports/round4_informed_transfer.json').read_text(encoding='utf-8'))
        result['informed_transfer_passes'] = check['passes']
        if not check['passes']:
            result['recommended_default'] = 'main'
            result['transfer_veto'] = 'The candidate did not preserve the known-risk source checks and the six-set safety/utility guard.'
        (ROOT / 'reports/round4_test.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        with (ROOT / 'reports/round4_test.md').open('a', encoding='utf-8') as stream:
            stream.write(f"\nInformed six-set transfer guard, including source-specific SQuAD/RGB checks: {check['passes']}. Final numerical recommendation: **{result['recommended_default']}**.\n")
    update(phase='Bright-Pro transfer diagnostic')
    bright_slug = ensure('brightpro', 'r4', train_slug, '--systems main,' + candidate, 3600)
    bright = download(bright_slug)
    # Save the frozen comparison checkpoint locally, including its weights and template.
    base_name, epoch = candidate[:-1], candidate[-1]
    folder = 'adapter_ep1' if epoch == '1' else 'adapter'
    checkpoint = download(train_slug, f'out/{base_name}/{folder}/.*')
    update(phase='complete', final_report=str(ROOT / 'reports/round4_test.md'),
           brightpro_results=str(bright / 'out/brightpro'), checkpoint=str(checkpoint / 'out' / base_name / folder))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        update(phase='needs inspection', error=f'{type(exc).__name__}: {exc}')
        raise
