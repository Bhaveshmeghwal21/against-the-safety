"""Compare completed DeepSeek runs to the saved Azure GPT-6 and adapter runs.

No model calls. The shared scored intersection contains only episodes present
and eligible under scorer v2 for every compared system.
"""
import json
from pathlib import Path

import report_azure_compare as A

ROOT = Path(__file__).resolve().parents[1]
REMOTE = {**A.AZ, 'azure_deepseek_exact': ROOT/'runs/deepseek_exact',
          'azure_deepseek_native': ROOT/'runs/deepseek_native'}


def cases_and_verdicts():
    return A.S.load_v([ROOT/'data/processed/verifier/cases_r5test_refs.jsonl',
                       ROOT/'data/processed/verifier/cases_r5test_final.jsonl'],
        [ROOT/'runs/account2_round5/ats-verify-main-r5-r5test-refs/out/verify/main',
         ROOT/'runs/account2_round5/ats-verify-main-r5-r5test-final/out/verify/main'])


def main():
    cases, verdict = cases_and_verdicts()
    data = {}
    for name, base in [('main', A.R5REF), ('candA', A.R5REF), ('candF1', A.R5ALT)]:
        data[name] = sum((A.load_run(base/f'{name}_bench_{p}/lenient', p) for p in A.PARTS), [])
        data[name+'+FB+V'] = sum((A.S.apply_v(A.S.fail_closed([r for r in data[name] if r['part'] == p]),
            f'{name}_bench_{p}', cases, verdict) for p in A.PARTS), [])
    for name, base in REMOTE.items():
        data[name] = sum((A.load_run(base/f'azure_bench_{p}/lenient', p) for p in A.PARTS), [])
        data[name+'+FB'] = A.S.fail_closed(data[name])
    indexed, keys, coverage = A.common_data(data)
    systems = {name: A.r5_stats(rows, keys) for name, rows in indexed.items()}
    pairs = [('azure_deepseek_native+FB', 'main+FB+V'),
             ('azure_deepseek_native+FB', 'candF1+FB+V'),
             ('azure_deepseek_native+FB', 'azure_native+FB'),
             ('azure_deepseek_exact+FB', 'azure_exact+FB'),
             ('azure_deepseek_native', 'main'),
             ('azure_deepseek_exact', 'main')]
    comparisons = A.grouped_intervals(indexed, keys, pairs)
    output = {'scope': 'round-5 scorer-v2 shared intersection, all compared actors', 'coverage': coverage,
              'systems': systems, 'comparisons': comparisons, 'external': {}}
    for part in A.OLD:
        d = {name: A.load_run(base/f'{name}_bench_{part}/lenient', part)
             for name, base in [('main', A.OLDMAIN), ('candA', A.OLDALT)]}
        d['main+FB'] = A.S.fail_closed(d['main'])
        for name, base in REMOTE.items():
            d[name] = A.load_run(base/f'azure_bench_{part}/lenient', part)
            d[name+'+FB'] = A.S.fail_closed(d[name])
        ix, kk, cov = A.common_data(d)
        output['external'][part] = {'coverage': cov, 'systems': {n: A.old_stats(rr, kk) for n, rr in ix.items()}}
    target = ROOT/'reports/deepseek_comparison'
    target.with_suffix('.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    lines = ['# DeepSeek V4 Flash versus GPT-6 Luna and Tiny Aya', '',
        'DeepSeek-V4-Flash was run through the existing Azure deployment in two modes, matching the GPT-6 comparison: original prompt/parser and native tool calls. The returned model name was DeepSeek-V4-Flash; the response did not expose an immutable checkpoint revision. All rows use the same scored episode intersection, scorer v2, same corpus, question variants, evidence masks, top-three results and three-search limit.',
        'Azure model decoding uses deployment defaults; native mode specifies reasoning_effort=none; max completion is 512 including any reasoning. Tiny Aya used greedy decoding and a 160-token cap. FB maps an invalid final turn to a refusal, which can increase success on hidden-evidence cases. V is the saved verifier and applies only to the adapters. The hosted Azure models have FB rows but no V; no adapter was rerun. DeepSeek V4 Flash is an open-weight model hosted by Azure, whereas GPT-6 Luna is closed-source.', '',
        f"Round 5: {coverage['common']} shared scored cases out of {coverage['expected']}; {coverage['provider_unavailable_or_missing']} provider-blocked/missing, {coverage['excluded_on_at_least_one_actor']} excluded by scorer v2 for at least one compared actor.", '',
        '| system | macro | single | compound | HotpotQA | RGB | unsupported / hidden | supported correct | false refusals | invalid |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in ('main', 'main+FB+V', 'candA+FB+V', 'candF1+FB+V', 'azure_exact', 'azure_exact+FB',
              'azure_native', 'azure_native+FB', 'azure_deepseek_exact', 'azure_deepseek_exact+FB',
              'azure_deepseek_native', 'azure_deepseek_native+FB'):
        s = systems[n]; c = s['components']
        lines.append(f"| {n} | {s['macro']:.3f} | {c['single']:.3f} | {c['compound']:.3f} | {c['hotpot']:.3f} | {c['rgb']:.3f} | {s['unsupported']}/{s['hidden_n']} | {s['supported_correct']:.3f} | {s['false_abstain']:.3f} | {s['invalid']:.3f} |")
    lines += ['', '## Paired differences', '', 'First system minus second. 95% bootstrap intervals resample question IDs within dataset (3,000 draws, seed 47); they do not account for article-level correlation, sampling variability across model runs, or the differing verifier and inference settings.', '']
    for pair, comp in comparisons.items():
        parts = []
        for metric in ('macro', 'unsupported_rate', 'supported_correct'):
            v = comp[metric]
            parts.append(f"{metric} {v['difference']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        lines.append(f"- {pair}: " + '; '.join(parts) + '.')
    lines += ['', '## Earlier external suites', '', 'Only raw main and candA (and main with FB applied offline) are available for these suites; no final verifier or candF1 was evaluated. The internal samples are not the official full benchmark leaderboards. The common set is selected separately per suite.', '',
        '| suite | system | common / expected | success | unsupported / hidden | supported correct | false refusals | invalid |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for part, result in output['external'].items():
        cov = result['coverage']
        for name in ('main', 'main+FB', 'candA', 'azure_exact', 'azure_native', 'azure_native+FB',
                     'azure_deepseek_exact', 'azure_deepseek_exact+FB', 'azure_deepseek_native', 'azure_deepseek_native+FB'):
            s = result['systems'][name]
            lines.append(f"| {part} | {name} | {cov['common']}/{cov['expected']} | {s['success']:.3f} | {s['unsupported']}/{s['hidden_n']} | {s['supported_correct']:.3f} | {s['false_abstain']:.3f} | {s['invalid']:.3f} |")
        lines.append(f"| *{part} exclusions* | missing {cov['provider_unavailable_or_missing']}; v2 {cov['excluded_on_at_least_one_actor']} | | | | | | |")
    lines += ['', 'Scorer v2 excludes hidden cases when any compared model sees alternate answer evidence. This equalizes denominators but changes the sample; evidence and alias matching remain imperfect. Recorded provider filters are excluded, not scored as wrong answers or refusals. Three early HTTP-400 filter errors stopped workers before the affected IDs were saved; the uncheckpointed episodes were resumed after the handler was fixed, so recorded filter counts can omit earlier events. No prompts were modified. These are single runs on a previously opened test, not a fresh sealed test or a guarantee of real-world safety.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:20]), flush=True)


if __name__ == '__main__':
    main()
