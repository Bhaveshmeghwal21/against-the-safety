"""A development-only bundle must not contain or evaluate the sealed test."""
from pathlib import Path
import importlib.util

from scripts.bundle_for_kaggle import permitted_path

spec = importlib.util.spec_from_file_location('round6_bootstrap', Path(__file__).resolve().parents[1] / 'kaggle/bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
development_jobs_only = bootstrap.development_jobs_only


def test_development_bundle_excludes_sealed_inputs():
    assert permitted_path(Path('data/bench/r4dev/questions.jsonl'), True)
    assert not permitted_path(Path('data/bench/r6t_squad/questions.jsonl'), True)
    assert not permitted_path(Path('data/bench/r6t_hotpot/corpus.sqlite'), True)
    assert not permitted_path(Path('reports/round6_test_manifest.json'), True)
    assert not permitted_path(Path('reports/round6_allocation.json'), True)
    assert permitted_path(Path('data/bench/r6t_squad/questions.jsonl'), False)


def test_development_job_allowlist_rejects_test_and_duplicates():
    valid = 'main:bench_r4dev,main:dev,candF1:bench_r4dev,candF1:dev'
    assert development_jobs_only(valid)
    assert not development_jobs_only(valid + ',main:bench_r6t_squad')
    assert not development_jobs_only('main:dev,main:dev,candF1:bench_r4dev,candF1:dev')
