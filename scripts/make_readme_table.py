"""Print the corrected, source-backed submission table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'reports' / name).read_text(encoding='utf-8'))


def render():
    reference = load('reference_policies.json')['sealed_r5test']
    local = load('round5_corrected.json')['systems']
    azure = load('azure_round5_comparison_corrected.json')['systems']
    deepseek = load('deepseek_comparison_corrected.json')['systems']
    entries = [
        ('Refuses every question', 'always_refuse', None),
        ('Base Tiny Aya', 'base', local['base']['macro']),
        ('Main, no repairs', 'main', local['main']['macro']),
        ('Main with both repairs', 'main+FB+V', local['main+FB+V']['macro']),
        ('candF1 with both repairs', 'candF1+FB+V', local['candF1+FB+V']['macro']),
        ('GPT-6 Luna with fail-closed', 'azure_native+FB', azure['azure_native+FB']['macro']),
        ('DeepSeek V4 Flash with fail-closed', 'azure_deepseek_native+FB', deepseek['azure_deepseek_native+FB']['macro']),
    ]
    lines = ['| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answered |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    pct = lambda value: '—' if value is None else f'{value * 100:.1f}%'
    for title, key, macro in entries:
        row = reference[key]
        lines.append(f"| {title} | {pct(row['macro'] if macro is None else macro)} | {pct(row['supported_correct'])} | "
                     f"{row['unsupported']}/{row['hidden_n']} | {pct(row['answered_share'])} | {pct(row['correct_among_answered'])} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    print(render())
