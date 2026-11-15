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


def fresh_test_table():
    """The second sealed test (reports/round6_second_look.json and round6_second_look_extra.json, scripts/report_round6_second_look*.py) next to the first one."""
    sl, extra = load('round6_second_look.json'), load('round6_second_look_extra.json')
    r5 = load('reference_policies.json')['sealed_r5test']
    groups = ('single', 'compound', 'hotpot', 'rgb')

    def row(title, components, macro, system):
        hidden = f"{system['unsupported']} / {system['hidden_n']}" + (f" ({pct(system['unsupported_rate'])})" if system['unsupported'] else '')
        return f"| {title} | " + ' | '.join(pct(components[g]) for g in groups) + f" | {pct(macro)} | {hidden} |"
    fresh, refuse = sl['systems']['main+FB+V'], sl['systems']['always refuse']
    first = r5['main+FB+V']
    return '\n'.join([
        '| | SQuAD single | SQuAD two-part | HotpotQA | RGB | Macro success | Answers on hidden evidence |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        row('Submitted system, fresh test', extra['round6_components']['main+FB+V'], fresh['macro'], fresh),
        row('Submitted system, first test', extra['round5_components']['main+FB+V'], first['macro'], first),
        row('Refuse every question, fresh test', extra['round6_components']['always refuse'], refuse['macro'], refuse),
    ])


def hindi_training_table():
    """Hindi training of main and candF1 (reports/hindi_results.json, scripts/report_hindi.py): every system with both repairs."""
    rep = load('hindi_results.json')
    hi, en = rep['xq2_hi']['systems'], rep['r6t']['systems']
    rows = [('Main + both repairs (submitted)', 'main+FB+V'), ('hiMain + both repairs', 'hiMain+FB+V'), ('candF1 + both repairs', 'candF1+FB+V'),
            ('hiF + both repairs', 'hiF+FB+V'), ('Refuse every question', 'always refuse')]
    lines = [f"| System | Hindi: correct and cited ({hi['main+FB+V']['n_answerable']} questions) | Hindi: answers when evidence was hidden | English: macro success (second sealed test) | English: answers when evidence was hidden |",
             '| --- | ---: | ---: | ---: | ---: |']
    for title, key in rows:
        h, e = hi[key], en[key]
        lines.append(f"| {title} | {pct(h['correct_and_cited'])} | {h['hidden_answers']} / {h['n_hidden']} | {pct(e['macro'])} | {e['unsupported']} / {e['hidden_n']} |")
    return '\n'.join(lines)


def hindi_rules_table():
    """The three registered rules for the Hindi adapters, each with its interval and whether it was met (limits as registered)."""
    rep = load('hindi_results.json')
    pairs = (('hiMain', 'main'), ('hiF', 'candF1'))
    hi, en = rep['xq2_hi']['comparisons'], rep['r6t']['comparisons']
    cell = lambda c, ok: f"{c['difference'] * 100:+.1f} [{c['ci95'][0] * 100:+.1f}, {c['ci95'][1] * 100:+.1f}]: {'met' if ok else 'not met'}"
    rules = (('A1. Hindi, correct and cited (lower end above 0)', lambda n, p: cell(hi[f'{n}+FB+V minus {p}+FB+V']['correct_and_cited'], hi[f'{n}+FB+V minus {p}+FB+V']['correct_and_cited']['ci95'][0] > 0)),
             ('A2. Hindi, answers on hidden evidence (upper end at most +2.0)', lambda n, p: cell(hi[f'{n}+FB+V minus {p}+FB+V']['hidden_answer_rate'], hi[f'{n}+FB+V minus {p}+FB+V']['hidden_answer_rate']['ci95'][1] <= 0.02)),
             ('A3a. English, macro success (lower end at least -3.0)', lambda n, p: cell(en[f'{n}+FB+V/{p}+FB+V']['macro'], en[f'{n}+FB+V/{p}+FB+V']['macro']['ci95'][0] >= -0.03)),
             ('A3b. English, answers on hidden evidence (upper end at most +2.0)', lambda n, p: cell(en[f'{n}+FB+V/{p}+FB+V']['unsupported_rate'], en[f'{n}+FB+V/{p}+FB+V']['unsupported_rate']['ci95'][1] <= 0.02)))
    lines = ['| Registered rule (difference in points, 95% interval) | hiMain minus main | hiF minus candF1 |', '| --- | ---: | ---: |']
    for title, f in rules:
        lines.append(f"| {title} | " + ' | '.join(f(n, p) for n, p in pairs) + ' |')
    return '\n'.join(lines)


def _ci(entry, digits=3):
    low, high = entry['ci95']
    return f"{entry['mean']:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def judge_table():
    """Judge-based answer metrics on the 300-episode sample of the first sealed test (reports/judge_results.json, scripts/report_judge.py)."""
    rep = load('judge_results.json')['systems']
    rows = [('Tiny Aya base', 'base'), ('Main adapter alone', 'main'), ('Main + both repairs (submitted)', 'main+FB+V'),
            ('candF1 + both repairs (alternative)', 'candF1+FB+V')]
    lines = ['| System | Answers given (of 300 episodes) | Faithfulness (higher is better) | Hallucination (lower is better) | Response groundedness | Answer relevancy |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for title, key in rows:
        m = rep[key]['metrics']
        lines.append(f"| {title} | {rep[key]['answers_given']} | {_ci(m['faithfulness']['all'])} | {_ci(m['hallucination']['all'])} | "
                     f"{m['response_groundedness']['all']['mean']:.3f} | {m['answer_relevancy']['all']['mean']:.3f} |")
    return '\n'.join(lines)


def judge_agreement_table():
    """What the judge says about the submitted system's answers, grouped by our own annotation-based outcome."""
    rep = load('judge_results.json')['judge_vs_annotation']['main+FB+V']
    names = {'supported_correct': 'Correct and cited (our scorer)', 'wrong_answer': 'Wrong answer on answerable question (our scorer)',
             'answered on hidden evidence': 'Answered although the evidence was hidden (our flag)'}
    lines = ['| Our label of the submitted system\'s answer | Answers | Mean faithfulness | Share fully faithful | Share with an unsupported or contradicted claim |', '| --- | ---: | ---: | ---: | ---: |']
    for key, title in names.items():
        g = rep[key]
        lines.append(f"| {title} | {g['judged']} | {g['mean_faithfulness']:.3f} | {g['share_fully_faithful']:.3f} | {g['share_any_hallucination']:.3f} |")
    return '\n'.join(lines)


def context_table():
    """Judge-based retrieval metrics of the three raw systems (same search tool, each system writes its own queries)."""
    rep = load('judge_results.json')['systems']
    lines = ['| System | Context relevance (share of shown passages that are relevant) | Context precision (relevant passages ranked first) | Context recall (reference answer found in the passages) |',
             '| --- | ---: | ---: | ---: |']
    for title, key in (('Tiny Aya base', 'base'), ('Main adapter', 'main'), ('candF1', 'candF1')):
        m = rep[key]['metrics']
        lines.append(f"| {title} | {_ci(m['context_relevance']['all'])} | {_ci(m['context_precision']['answerable'])} | {_ci(m['context_recall']['answerable'])} |")
    return '\n'.join(lines)


def numeric_table():
    rep = load('numeric_accuracy.json')['systems']
    lines = ['| System | Numeric-answer episodes (questions) | Answered | Numerically correct | Numeric accuracy | Correct among answered |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for title, s in rep.items():
        lines.append(f"| {title} | {s['episodes']} ({s['questions']}) | {s['answered']} | {s['numerically_correct']} | {pct(s['numeric_accuracy'])} | {pct(s['correct_among_answered'])} |")
    return '\n'.join(lines)


def latency_table():
    rep = load('latency.json')['arms']
    names = {'base': 'Tiny Aya base, same loop, no model self-check', 'submitted': 'Submitted system (main adapter, fail-closed rule, self-check)'}
    lines = ['| System | Requests | p50 (s) | p95 (s) | p99 (s) | Requests per second | Cost per 1,000 requests, busy T4 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, s in rep.items():
        lat = s['latency_ms']
        lines.append(f"| {names[arm]} | {s['requests']} | {lat['p50'] / 1000:.2f} | {lat['p95'] / 1000:.2f} | {lat['p99'] / 1000:.2f} | {s['throughput']['rps']:.3f} | ${s['cost_per_1000_requests_usd_busy_t4']:.2f} |")
    return '\n'.join(lines)


def tool_use_table():
    """AuditKIT tool-call metrics of the raw agents per suite of the first sealed test (reports/tool_use_r5t.json, scripts/report_tool_use.py)."""
    rep = load('tool_use_r5t.json')['trajectories']
    suites = {'r5t_squad': 'SQuAD', 'r5t_hotpot': 'HotpotQA', 'r5t_rgb': 'RGB'}
    lines = ['| System | Suite | Well-formed tool calls | Repeated searches (lower is better) | Loop-free episodes | Searches per episode | Invalid endings (lower is better) |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, title in (('base', 'Tiny Aya base'), ('main', 'Main adapter'), ('candF1', 'candF1')):
        for part, suite in suites.items():
            m = rep[name][part]
            lines.append(f"| {title} | {suite} | {m['tool_call_validity'] * 100:.1f}% | {m['redundant_tool_calls'] * 100:.1f}% | {m['loop_free_episodes'] * 100:.1f}% | "
                         f"{m['mean_searches']:.2f} | {m['invalid_endings'] * 100:.1f}% |")
    return '\n'.join(lines)


def judge_agreement_sentence():
    ag = load('judge_results.json')['judge_agreement_with_annotation_labels']
    return (f"the judge's fully-faithful verdict agrees with our label on {ag['agreement'] * 100:.1f}% of {ag['answers']} answers "
            f"(Cohen's kappa {ag['cohen_kappa']:.2f})")


if __name__ == '__main__':
    print(results_table())
    print()
    print(hosted_table())
    print()
    print(external_table())
    print()
    print(multilingual_table())
    print()
    print(fresh_test_table())
    print()
    print(hindi_training_table())
    print()
    print(hindi_rules_table())
    print()
    print(judge_table())
    print()
    print(judge_agreement_table())
    print()
    print(context_table())
    print()
    print(numeric_table())
    print()
    print(tool_use_table())
