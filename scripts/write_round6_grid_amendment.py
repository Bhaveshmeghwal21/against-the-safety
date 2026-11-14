"""Record the validation-informed grid change before any development output."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = ('reports/round6_selection.json', 'reports/round6_selection.md',
             'runs/kaggle/push-ats-r6-system-dev', 'runs/kaggle/ats-r6-system-dev', 'runs/round6/dev')
    found = [path for path in paths if (ROOT / path).exists()]
    if found:
        raise SystemExit('Development artifacts already exist: ' + ', '.join(found))
    target = ROOT / 'research/2026-11-14-round6-preregistration.md'
    text = target.read_text(encoding='utf-8')
    if '## Amendment 2:' in text:
        raise SystemExit('Amendment 2 already exists; do not silently rewrite it')
    reports = {name: json.loads((ROOT / f'reports/round6_verifier_train_{name}.json').read_text(encoding='utf-8'))
               for name in ('V2m', 'V2b')}
    for name, report in reports.items():
        if report.get('name') != name or report.get('stand_in') is not False or len(report.get('epochs', [])) != 2:
            raise ValueError(f'Incomplete or mismatched training report: {name}')
    values = {name: report['epochs'][-1] for name, report in reports.items()}
    data = json.loads((ROOT / 'reports/round6_verifier_data.json').read_text(encoding='utf-8'))
    def pct(value):
        return f'{value:.0%}'
    text += ('\n## Amendment 2: threshold-grid revision\n\n'
             'Recorded before any Round 6 development-model output existed, after inspecting the final held-out '
             'training-validation results at threshold 0.50. '
             f"V2m kept {pct(values['V2m']['supported_kept_at_0_5'])} of positive cases and passed "
             f"{pct(values['V2m']['unsupported_passed_at_0_5'])} of negative cases; "
             f"V2b kept {pct(values['V2b']['supported_kept_at_0_5'])} and passed "
             f"{pct(values['V2b']['unsupported_passed_at_0_5'])}, respectively. "
             f"These are internal validation examples, not the development gate; the validation set contains "
             f"{data['splits']['val']['yes_no']} yes/no examples. The grid change is therefore informed by validation "
             'and is not a blind preregistration relative to that validation.\n\n'
             'Replace only the checker-gate threshold grid with `{0.20, 0.25, ..., 0.95}`, inclusive, in increments of 0.05. '
             'For each checker, choose the smallest threshold at which at most 15% of hidden-evidence answers pass; '
             'the checker is eligible only if at least 90% of correct and cited answers are kept at that threshold. '
             'The actor pool, deterministic checks, checker tie break, second-look threshold grid and second-look gate '
             'are unchanged.\n\n'
             'The selection report will also show exact yes and no answers separately for every checker and every '
             'checker threshold, among eligible answers that passed deterministic checks. Terminal punctuation is '
             'ignored when identifying yes/no. These counts are diagnostic only and do not affect either gate.\n\n'
             'The eventual committed development decision will be recorded separately as `## Amendment 3: '
             'development selection` along with the exact selection report, before any sealed-test output is opened.\n')
    target.write_text(text, encoding='utf-8')
    print('Recorded validation-informed grid revision and reserved Amendment 3 for development selection.')


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
