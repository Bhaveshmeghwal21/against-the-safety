"""Print the two results tables of the README from the saved reports (tests/test_readme_tables.py checks that the README still matches).

  python scripts/make_readme_table.py

Local systems and the refuse-everything line come from reports/reference_policies.json (934 episodes of the sealed round-5 test, scorer v3).
The hosted models come from reports/fair_rerun.json (925 shared episodes, hosted models told the search limit and answer format).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'reports' / name).read_text(encoding='utf-8'))


def pct(value):
    return 'none' if value is None else f'{value * 100:.1f}%'


def results_table():
    ref = load('reference_policies.json')['sealed_r5test']
    rows = [('Tiny Aya base', 'base', False), ('Main adapter alone', 'main', False), ('Main + both repairs (submitted)', 'main+FB+V', True),
            ('candF1 + both repairs (alternative)', 'candF1+FB+V', False), ('Refuse every question', 'always_refuse', False)]
    lines = ['| System | Macro success | Correct and cited | Answers when evidence was hidden | Correct among answers shown |', '| --- | ---: | ---: | ---: | ---: |']
    for title, key, bold in rows:
        r = ref[key]
        macro, hidden = pct(r['macro']), f"{r['unsupported']} / {r['hidden_n']}"
        if bold:
            title, macro, hidden = f'**{title}**', f'**{macro}**', f'**{hidden}**'
        lines.append(f"| {title} | {macro} | {pct(r['supported_correct'])} | {hidden} | {pct(r['correct_among_answered'])} |")
    return '\n'.join(lines)


def hosted_table():
    fair = load('fair_rerun.json')['systems']
    rows = [('Main + both repairs', 'main+FB+V'), ('candF1 + both repairs', 'candF1+FB+V'), ('GPT-6 Luna', 'gpt_fair+FB'), ('DeepSeek V4 Flash', 'deepseek_fair+FB')]
    lines = ['| System | Macro success | Answered | Correct among answered | Wrong or unsupported shown, of all cases |', '| --- | ---: | ---: | ---: | ---: |']
    for title, key in rows:
        r = fair[key]
        lines.append(f"| {title} | {pct(r['macro'])} | {pct(r['answered_share'])} | {pct(r['correct_among_answered'])} | {pct(r['wrong_or_unsupported_shown_share'])} |")
    return '\n'.join(lines)


def external_table():
    """Earlier suites the adapters never trained on, with fail-closed applied to every system (reports/external_benchmarks_corrected.json)."""
    ext = load('external_benchmarks_corrected.json')
    refuse = load('reference_policies.json')['external']
    suites = [('2wiki', '2WikiMultihopQA'), ('musique', 'MuSiQue'), ('squad2', 'SQuAD 2.0'), ('rgb', 'RGB'), ('rgb_int', 'RGB integration')]
    lines = ['| Suite (cases) | Refuse every question | Main + fail-closed | GPT-6 Luna + fail-closed | DeepSeek V4 Flash + fail-closed |', '| --- | ---: | ---: | ---: | ---: |']
    for key, title in suites:
        s = ext[key]['systems']
        lines.append(f"| {title} ({ext[key]['coverage']['common']}) | {pct(refuse[key]['always_refuse']['success'])} | {pct(s['main+FB']['success'])} "
                     f"| {pct(s['azure_native+FB']['success'])} | {pct(s['azure_deepseek_native+FB']['success'])} |")
    return '\n'.join(lines)


def multilingual_table():
    """The Hindi / English / Spanish check on XQuAD (reports/xquad_results.json, scripts/report_xquad.py). The submitted system unless a row says otherwise."""
    rep = load('xquad_results.json')['systems']
    langs = ('hi', 'en', 'es')
    row = lambda title, f: f"| {title} | " + ' | '.join(f(rep[l]) for l in langs) + ' |'
    hidden = lambda s: f"{s['hidden_answers']} / {s['n_hidden']}"
    return '\n'.join([
        '| XQuAD, same questions in three languages | ' + ' | '.join(f"{n} ({rep[l]['main+FB+V']['n_answerable']} questions)" for n, l in zip(('Hindi', 'English', 'Spanish'), langs)) + ' |',
        '| --- | ---: | ---: | ---: |',
        row('Correct and cited, when the documents hold the answer', lambda r: pct(r['main+FB+V']['correct_and_cited'])),
        row('Answers although the evidence was hidden', lambda r: hidden(r['main+FB+V'])),
        row('Same, Tiny Aya base model', lambda r: hidden(r['base'])),
        row('Success (refusing every question scores 50.0%)', lambda r: pct(r['main+FB+V']['success'])),
        row('Adapter alone: turns that ended invalid', lambda r: pct(r['main']['invalid'])),
    ])


if __name__ == '__main__':
    print(results_table())
    print()
    print(hosted_table())
    print()
    print(external_table())
    print()
    print(multilingual_table())
