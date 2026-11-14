"""Record the fixed development decision before any sealed-test output is opened."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from select_round6 import RECOVERY_THRESHOLDS, THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / 'research/2026-11-14-round6-preregistration.md'
SELECTION = ROOT / 'reports/round6_selection.json'


def render_amendment(report, selection_sha=None):
    if report.get('data') != 'r4dev and HotpotQA dev only':
        raise ValueError('Selection data is not the registered development data')
    gates = report['checker_gates']
    binary = report['binary_answer_diagnostics']
    for checker in ('V', 'V2m', 'V2b'):
        gate = gates[checker]
        rows = gate['thresholds']
        if [row['threshold'] for row in rows] != THRESHOLDS:
            raise ValueError(f'{checker} checker threshold grid is incorrect')
        first = next((row for row in rows if row['hidden_total'] and row['hidden_passed'] / row['hidden_total'] <= .15), None)
        passes = bool(first and first['correct_total'] and first['correct_kept'] / first['correct_total'] >= .90)
        if gate['chosen'] != first or gate['passes'] is not passes:
            raise ValueError(f'{checker} checker gate is inconsistent')
        for answer in ('yes', 'no'):
            if [row['threshold'] for row in binary[checker][answer]] != THRESHOLDS:
                raise ValueError(f'{checker} {answer} diagnostic grid is incorrect')
    passing = [name for name in ('V2m', 'V2b') if gates[name]['passes']]
    expected = max(passing, key=lambda name: gates[name]['chosen']['correct_kept']) if passing else None
    chosen = report['chosen_checker']
    if chosen != expected:
        raise ValueError('Chosen checker does not follow the registered retention rule and tie break')
    recovery = report.get('second_look_gate')
    if chosen is None:
        if recovery is not None or report['second_look_enabled']:
            raise ValueError('A second look cannot be selected without an eligible checker')
    else:
        if recovery is None or [row['threshold'] for row in recovery['thresholds']] != RECOVERY_THRESHOLDS:
            raise ValueError('Second-look threshold grid is incorrect')
        first = next((row for row in recovery['thresholds'] if row['hidden_total'] and row['hidden_added'] / row['hidden_total'] <= .05), None)
        passes = bool(first and first['correct_added'] > 0 and first['correct_added'] >= 3 * first['hidden_added'])
        if recovery['chosen'] != first or recovery['passes'] is not passes or report['second_look_enabled'] is not passes:
            raise ValueError('Second-look gate is inconsistent')
    lines = ['## Amendment 3: development selection', '',
             'Recorded from the fixed development selection before opening any Round 6 sealed-test output.', '']
    if selection_sha:
        lines += [f'Selection report SHA256: `{selection_sha}`', '']
    for checker in ('V', 'V2m', 'V2b'):
        gate = gates[checker]
        row = gate['chosen']
        if row is None:
            high = gate['thresholds'][-1]
            lines.append(f"- {checker}: no threshold passed the hidden-evidence limit; at {high['threshold']:.2f}, "
                         f"correct kept {high['correct_kept']}/{high['correct_total']} and "
                         f"hidden answers passed {high['hidden_passed']}/{high['hidden_total']}.")
        else:
            lines.append(f"- {checker}: threshold {row['threshold']:.2f}; correct kept {row['correct_kept']}/{row['correct_total']}; "
                         f"hidden answers passed {row['hidden_passed']}/{row['hidden_total']}; "
                         f"checker gate {'passed' if gate['passes'] else 'failed'}.")
    lines += ['', f"Selected checker: {chosen or 'none; retain V'}."]
    if chosen:
        row = recovery['chosen']
        if report['second_look_enabled']:
            lines.append(f"Second look: on at {row['threshold']:.2f}, adding {row['correct_added']} correct answers and "
                         f"{row['hidden_added']}/{row['hidden_total']} hidden-evidence answers.")
        else:
            lines.append('Second look: off; its development gate did not pass.')
    else:
        lines.append('Second look: off; no trained checker passed the development gate.')
    lines += ['', 'Yes/no answers are listed separately for each checker and threshold in the selection report; '
              'they are diagnostics only and did not change either gate.']
    return '\n'.join(lines) + '\n'


def main():
    committed = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'show',
                                'HEAD:research/2026-11-14-round6-preregistration.md'],
                               cwd=ROOT, capture_output=True, check=False)
    if committed.returncode:
        raise ValueError('The preregistration is not committed')
    original = committed.stdout.decode('utf-8').replace('\r\n', '\n')
    if original != PREREG.read_text(encoding='utf-8').replace('\r\n', '\n'):
        raise ValueError('The preregistration differs from the committed version')
    if not re.search(r'(?m)^## Amendment 2:', original) or re.search(r'(?m)^## Amendment 3:', original):
        raise ValueError('Amendment 2 must be committed and Amendment 3 must not already exist')
    saved = SELECTION.read_text(encoding='utf-8').replace('\r\n', '\n')
    report = json.loads(saved)
    sha = hashlib.sha256(saved.encode('utf-8')).hexdigest()
    PREREG.write_text(original.rstrip() + '\n\n' + render_amendment(report, sha), encoding='utf-8')
    print('Recorded Amendment 3 from the fixed development selection; commit it with the selection report.')


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
    except (ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
