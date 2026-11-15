"""Safety and comparison checks for the exploratory candF1 XQuAD run."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import report_xquad_candf1 as report

spec = importlib.util.spec_from_file_location('candf1_bootstrap', ROOT / 'kaggle/bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_job_allowlist_has_only_the_three_requested_candidate_suites():
    allowed = 'candF1:bench_xq_hi:40,candF1:bench_xq_en:30,candF1:bench_xq_es:20'
    assert bootstrap.xquad_candf1_jobs_only(allowed)
    assert not bootstrap.xquad_candf1_jobs_only(allowed + ',candF1:bench_r6t_squad')
    assert not bootstrap.xquad_candf1_jobs_only(allowed.replace('candF1:bench_xq_es:20', 'main:bench_xq_es:20'))
    assert not bootstrap.xquad_candf1_jobs_only(allowed.replace('candF1:bench_xq_es:20', 'candF1:bench_xq_en:20'))
    assert not bootstrap.xquad_candf1_jobs_only(allowed.replace(':40', ':nan'))


def test_comparison_rejects_different_episodes_indexes_and_generation():
    rows = [{'qid': 'q1', 'variant': 'open', 'answerable': True, 'excluded_v3': False}]
    summary = {'index_hash': 'abc', 'generation': {'batch_size': 12, 'concurrency': 24, 'max_new_tokens': 160,
               'max_input_tokens': 8000, 'abstain_bias': 0.0}}
    report.compare_conditions(rows, rows, summary, summary)
    with pytest.raises(ValueError, match='episode sets'):
        report.compare_conditions(rows, [{**rows[0], 'qid': 'q2'}], summary, summary)
    with pytest.raises(ValueError, match='index hashes'):
        report.compare_conditions(rows, rows, summary, {**summary, 'index_hash': 'changed'})
    with pytest.raises(ValueError, match='generation settings'):
        report.compare_conditions(rows, rows, summary, {**summary, 'generation': {**summary['generation'], 'max_input_tokens': 4096}})


def test_checker_cases_must_match_every_answer():
    row = {'qid': 'q', 'variant': 'open', 'final': 'answer', 'answer': 'one', 'question': 'Q',
           'answerable': True, 'outcome_v3': 'supported_correct'}
    key = ('candF1_bench_xq_hi', 'q', 'open')
    case = {'answer': 'one', 'question': 'Q', 'passages': 'one',
            'deterministic': {'cites_unshown': False, 'answer_part_absent_from_cited': False}}
    assert report.repaired([row], 'candF1', 'hi', {key: case}, {key: .9}) == [row]
    assert report.repaired([row], 'candF1', 'hi', {key: case}, {key: .1})[0]['final'] == 'abstain'
    with pytest.raises(ValueError, match='cover exactly'):
        report.repaired([row], 'candF1', 'hi', {}, {})
    with pytest.raises(ValueError, match='differs from transcript'):
        report.repaired([row], 'candF1', 'hi', {key: {**case, 'answer': 'wrong'}}, {key: .9})
    with pytest.raises(ValueError, match='missing or invalid'):
        report.repaired([row], 'candF1', 'hi', {key: case}, {})
