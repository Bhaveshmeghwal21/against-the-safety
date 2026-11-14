"""Validate completed checker-training reports before the development run."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assess(report, expected_name=None):
    errors = []
    if report.get('stand_in') is not False:
        errors.append('not a real GPU training report')
    memory = report.get('peak_gpu_memory_gb')
    if not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory <= 0:
        errors.append('missing or invalid GPU memory measurement')
    if expected_name:
        args = report.get('args', {})
        if report.get('name') != expected_name or args.get('name') != expected_name:
            errors.append('checker identity does not match')
        expected_lr = 3e-5 if expected_name == 'V2m' else 1e-4
        if args.get('lr') != expected_lr or bool(args.get('init_adapter')) != (expected_name == 'V2m'):
            errors.append('initial adapter or learning rate differs from preregistration')
        for key, value in (('epochs', 2), ('grad_accum', 8), ('max_length', 2560), ('seed', 61)):
            if args.get(key) != value:
                errors.append(f'{key} differs from preregistration')
    epochs = report.get('epochs', [])
    if len(epochs) != 2:
        errors.append('expected two completed epochs')
    if not report.get('train_rows') or not report.get('val_rows'):
        errors.append('training or validation is empty')
    for index, epoch in enumerate(epochs, 1):
        if epoch.get('epoch') != index:
            errors.append(f'epoch {index} is missing or out of order')
        loss = epoch.get('train_loss')
        if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
            errors.append(f'epoch {index} loss is invalid')
        for metric in ('auroc', 'supported_kept_at_0_5', 'unsupported_passed_at_0_5'):
            value = epoch.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f'epoch {index} {metric} is invalid')
    return {'valid': not errors, 'errors': errors,
            'final_auroc': epochs[-1].get('auroc') if epochs and not errors else None,
            'train_rows': report.get('train_rows'), 'val_rows': report.get('val_rows'),
            'train_dropped_for_length': report.get('train_dropped_for_length'),
            'val_dropped_for_length': report.get('val_dropped_for_length')}


def decision(reports):
    results = {name: assess(reports[name], name) for name in ('V2m', 'V2b')}
    if not all(result['valid'] for result in results.values()):
        status = 'manual_review_invalid_or_incomplete_report'
    elif all(result['final_auroc'] < .86 for result in results.values()):
        status = 'stop_both_below_registered_floor'
    else:
        status = 'eligible_for_development_gate'
    return {'status': status, 'checkers': results}


def assess_adapter(path):
    """Check the exact files needed for the next bundle and reject non-finite weights."""
    from safetensors import safe_open
    import torch

    required = ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return {'valid': False, 'errors': ['missing files: ' + ', '.join(missing)]}
    errors = []
    config = json.loads((path / 'adapter_config.json').read_text(encoding='utf-8'))
    for field, expected in (('peft_type', 'LORA'), ('r', 16), ('lora_alpha', 32), ('lora_dropout', .05)):
        if config.get(field) != expected:
            errors.append(f'adapter {field} does not match preregistration')
    try:
        with safe_open(str(path / 'adapter_model.safetensors'), framework='pt', device='cpu') as weights:
            keys = list(weights.keys())
            if not keys or not all('lora_' in key for key in keys):
                errors.append('unexpected or empty adapter weights')
            for key in keys:
                if not bool(torch.isfinite(weights.get_tensor(key)).all()):
                    errors.append('non-finite adapter weights')
                    break
    except Exception as exc:
        keys = []
        errors.append('adapter weights cannot be read: ' + type(exc).__name__)
    return {'valid': not errors, 'errors': errors, 'weight_tensors': len(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='directory containing V2m and V2b output directories')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.run)
    reports = {name: json.loads((root / name / 'train_report.json').read_text(encoding='utf-8')) for name in ('V2m', 'V2b')}
    result = decision(reports)
    adapters = {name: assess_adapter(root / name / 'adapter') for name in ('V2m', 'V2b')}
    result['adapters'] = adapters
    if not all(adapter['valid'] for adapter in adapters.values()):
        result['status'] = 'manual_review_invalid_or_incomplete_adapter'
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError('Output must stay within the workspace')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
