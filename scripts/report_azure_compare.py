"""Paired Azure-versus-saved-model reports; no model or cloud requests.

Use only after both Azure runs finish. The fixed intersection excludes provider
blocks and any episode excluded by scorer v2 for any compared actor.
"""
import json
from pathlib import Path

import numpy as np
import report_round5 as R
import rescore_round5 as RS
import select_round5 as S

ROOT = Path(__file__).resolve().parents[1]
R5REF = ROOT/'runs/account2_round5/ats-eval-jobs-r5test-ref/out/eval'
R5ALT = ROOT/'runs/account2_round5/ats-eval-jobs-r5test-final/out/eval'
OLDMAIN = ROOT/'runs/kaggle/ats-eval-bench-a/out/eval'
OLDALT = ROOT/'runs/kaggle/ats-eval-bench-b/out/eval'
AZ = {'azure_exact': ROOT/'runs/azure_round5', 'azure_native': ROOT/'runs/azure_round5_native'}
PARTS = ('r5t_squad','r5t_hotpot','r5t_rgb')
OLD = ('2wiki','musique','squad2','rgb','rgb_int')


def load_run(folder, part):
    _, _, rows, mismatches = RS.rescore_run(folder)
    if mismatches:
        raise RuntimeError(f'legacy scorer mismatch in {folder}: {len(mismatches)}')
    return [{**r, 'part': part, 'comp': R.component(part,r['qtype']) if part in PARTS else part} for r in rows]


def by_key(rows):
    d={(r['part'],r['qid'],r['variant']):r for r in rows}
    assert len(d)==len(rows)
    return d


def common_data(data, reference='main'):
    indexed={name:by_key(rows) for name,rows in data.items()}
    expected=set(indexed[reference])
    for name,rr in indexed.items():
        if name.startswith('azure'):
            assert set(rr) <= expected, name
        else:
            assert set(rr)==expected, name
    present=set.intersection(*(set(x) for x in indexed.values()))
    eligible={k for k in present if all(not rows[k]['excluded_v2'] for rows in indexed.values())}
    return indexed,sorted(eligible), {'expected':len(expected),'provider_unavailable_or_missing':len(expected-present),
            'excluded_on_at_least_one_actor':len(present-eligible),'common':len(eligible),
            'per_system':{n:{'present':len(rr),'excluded_v2':sum(r['excluded_v2'] for r in rr.values())} for n,rr in indexed.items()}}


def r5_stats(rr, keys):
    rows=[rr[k] for k in keys]
    return R.stats_from(R.vectors(rows,keys)[1].sum(axis=0), R.vectors(rows,keys)[0]) | {
        'n':len(rows), 'false_abstain':sum(r['outcome_v2']=='false_abstain' for r in rows if r['answerable'])/max(1,sum(r['answerable'] for r in rows)),
        'invalid':sum(r['final']=='invalid' for r in rows)/max(1,len(rows))}


def grouped_intervals(indexed, keys, pairs, reps=3000):
    comps, vectors = {}, {}
    for name,rr in indexed.items():
        comps[name],vectors[name]=R.vectors([rr[k] for k in keys],keys)
    clusters={}
    for i,(part,qid,_) in enumerate(keys):
        clusters.setdefault(part,{}).setdefault(qid,[]).append(i)
    rng=np.random.default_rng(47)
    diffs={p:{k:[] for k in ('macro','unsupported_rate','supported_correct')} for p in pairs}
    for _ in range(reps):
        idx=[]
        for part in sorted(clusters):
            ids=sorted(clusters[part]); picks=rng.choice(len(ids),size=len(ids))
            for pick in picks:idx.extend(clusters[part][ids[pick]])
        stats={name:R.stats_from(v[idx].sum(axis=0),comps[name]) for name,v in vectors.items()}
        for a,b in pairs:
            for metric,values in diffs[(a,b)].items():values.append(stats[a][metric]-stats[b][metric])
    out={}
    for (a,b),metrics in diffs.items():
        sa,sb=r5_stats(indexed[a],keys),r5_stats(indexed[b],keys)
        out[f'{a}/{b}']={m:{'difference':sa[m]-sb[m], 'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for m,v in metrics.items()}
    return out


def old_stats(rr,keys):
    rows=[rr[k] for k in keys]; hidden=[r for r in rows if not r['answerable']]; ans=[r for r in rows if r['answerable']]
    return {'n':len(rows),'success':sum(r['strong_v2'] for r in rows)/len(rows),
            'unsupported':sum(r['outcome_v2'].startswith('unsupported_') for r in hidden),'hidden_n':len(hidden),
            'supported_correct':sum(r['strong_v2'] for r in ans)/max(1,len(ans)),
            'false_abstain':sum(r['outcome_v2']=='false_abstain' for r in ans)/max(1,len(ans)),
            'invalid':sum(r['final']=='invalid' for r in rows)/len(rows)}


def main():
    cases,verdict=S.load_v([ROOT/'data/processed/verifier/cases_r5test_refs.jsonl',ROOT/'data/processed/verifier/cases_r5test_final.jsonl'],
        [ROOT/'runs/account2_round5/ats-verify-main-r5-r5test-refs/out/verify/main',ROOT/'runs/account2_round5/ats-verify-main-r5-r5test-final/out/verify/main'])
    source={}
    for system,base in [('main',R5REF),('candA',R5REF),('candF1',R5ALT)]:
        source[system]=sum((load_run(base/f'{system}_bench_{p}/lenient',p) for p in PARTS),[])
    for name,base in AZ.items():
        source[name]=sum((load_run(base/f'azure_bench_{p}/lenient',p) for p in PARTS),[])
    data=dict(source)
    for name in ('main','candA','candF1'):
        grouped=[]
        for p in PARTS:
            rr=[r for r in source[name] if r['part']==p]
            grouped += S.apply_v(S.fail_closed(rr),f'{name}_bench_{p}',cases,verdict)
        data[name+'+FB+V']=grouped
    for name in AZ:
        data[name+'+FB']=S.fail_closed(source[name])
    indexed,keys,coverage=common_data(data)
    table={n:r5_stats(rr,keys) for n,rr in indexed.items()}
    pairs=[(a,b) for a in ('azure_exact','azure_exact+FB','azure_native','azure_native+FB') for b in ('main','main+FB+V','candF1+FB+V')]
    comparisons=grouped_intervals(indexed,keys,pairs)
    report={'scope':'r5 sealed test; common intersection; scorer v2', 'coverage':coverage,'systems':table,'comparisons':comparisons,'external':{}}
    for part in OLD:
        d={}
        for system,base in [('main',OLDMAIN),('candA',OLDALT)]:
            d[system]=load_run(base/f'{system}_bench_{part}/lenient',part)
        for name,base in AZ.items():
            d[name]=load_run(base/f'azure_bench_{part}/lenient',part)
            d[name+'+FB']=S.fail_closed(d[name])
        d['main+FB']=S.fail_closed(d['main'])
        ix,kk,cov=common_data(d)
        report['external'][part]={'coverage':cov,'systems':{n:old_stats(rr,kk) for n,rr in ix.items()}}
    out=ROOT/'reports/azure_round5_comparison'
    out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Azure GPT-6 Luna vs the saved RAG systems','',
           'Same question/variant keys, index, masks, top 3, passage display and three-search budget. Scorer v2; every compared actor uses the same intersection of scorable episodes. Exclusion remains a heuristic, not a proof of missing evidence. Macro is the unweighted mean of success rates on SQuAD single, SQuAD compound, HotpotQA and RGB; success is a supported correct answer or a correct abstention.',
           'Azure exact uses the original prompt/parser and default temperature/reasoning; native uses function tools, minimal output-format normalization and reasoning_effort=none, as required by this deployment. Tiny Aya uses greedy decoding and 160 generated tokens; Azure has a 512-token completion cap including any reasoning. Each Azure mode is a separate stochastic run.',
           'FB maps invalid actions to abstention. V is the previously saved Tiny Aya verifier, available for the three round-5 adapter systems only. Azure has no V; no adapter was run for this comparison. Provider-blocked episodes are excluded, not scored as wrong answers or abstentions.', '',
           f"Round 5: {coverage['common']} common episodes of {coverage['expected']}; provider-unavailable {coverage['provider_unavailable_or_missing']}; excluded by at least one actor under v2 {coverage['excluded_on_at_least_one_actor']}.", '',
           '| system | macro | single | compound | HotpotQA | RGB | unsupported / hidden | supported correct | false refusals | invalid |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in ('main','main+FB+V','candA+FB+V','candF1+FB+V','azure_exact','azure_exact+FB','azure_native','azure_native+FB'):
        s=table[n];c=s['components']
        lines.append(f"| {n} | {s['macro']:.3f} | {c['single']:.3f} | {c['compound']:.3f} | {c['hotpot']:.3f} | {c['rgb']:.3f} | {s['unsupported']}/{s['hidden_n']} | {s['supported_correct']:.3f} | {s['false_abstain']:.3f} | {s['invalid']:.3f} |")
    lines += ['', '## Paired differences', '', '95% bootstrap intervals resample complete question IDs within each dataset, seed 47; they do not remove article-level correlation.', '']
    for pair,c in comparisons.items():
        a=c['macro'];u=c['unsupported_rate']
        lines.append(f"- {pair}: macro {a['difference']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]; unsupported rate {u['difference']:+.3f} [{u['ci95'][0]:+.3f}, {u['ci95'][1]:+.3f}].")
    lines += ['', '## Earlier external suites', '', 'These use scorer v2 on the common intersection. Only raw main and candA (and main plus fail-closed) were previously evaluated on these suites; the final round-5 verifier and candF1 were not run here. These are internal samples, not official leaderboard scores.', '',
              '| benchmark | system | common / expected | success | unsupported / hidden | supported correct | false refusals | invalid |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p,v in report['external'].items():
        c=v['coverage']
        for name in ('main','main+FB','candA','azure_exact','azure_exact+FB','azure_native','azure_native+FB'):
            s=v['systems'][name]
            lines.append(f"| {p} | {name} | {c['common']}/{c['expected']} | {s['success']:.3f} | {s['unsupported']}/{s['hidden_n']} | {s['supported_correct']:.3f} | {s['false_abstain']:.3f} | {s['invalid']:.3f} |")
        lines.append(f"| *{p} exclusions* | missing {c['provider_unavailable_or_missing']}; v2 {c['excluded_on_at_least_one_actor']} | | | | | | |")
    lines += ['', 'The long-form data records exclusions by model. Models may retrieve different alternate support; applying one shared intersection avoids comparing unequal denominators, but incomplete alias and semantic checks can still mislabel cases.']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
