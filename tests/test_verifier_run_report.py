"""A malformed training result must not silently pass the development gate."""
from scripts.check_verifier_run import assess, decision


def report(auc, name='V2m'):
    return {'stand_in': False, 'name': name, 'peak_gpu_memory_gb': 7.0, 'train_rows': 2, 'val_rows': 2,
            'args': {'name': name, 'lr': 3e-5 if name == 'V2m' else 1e-4,
                     'init_adapter': 'main' if name == 'V2m' else None, 'epochs': 2,
                     'grad_accum': 8, 'max_length': 2560, 'seed': 61}, 'epochs': [
        {'epoch': i, 'train_loss': .4, 'auroc': auc, 'supported_kept_at_0_5': .9, 'unsupported_passed_at_0_5': .1}
        for i in (1, 2)]}


def test_nonfinite_metrics_and_standins_need_review():
    assert not assess(report(float('nan')))['valid']
    assert not assess({**report(.9), 'stand_in': True})['valid']


def test_registered_floor_applies_to_both_checkers():
    assert decision({'V2m': report(.85), 'V2b': report(.84, 'V2b')})['status'] == 'stop_both_below_registered_floor'
    assert decision({'V2m': report(.86), 'V2b': report(.84, 'V2b')})['status'] == 'eligible_for_development_gate'
    assert decision({'V2m': report(.95), 'V2b': report(float('nan'), 'V2b')})['status'] == 'manual_review_invalid_or_incomplete_report'
    assert decision({'V2m': report(.95), 'V2b': report(.95)})['status'] == 'manual_review_invalid_or_incomplete_report'
