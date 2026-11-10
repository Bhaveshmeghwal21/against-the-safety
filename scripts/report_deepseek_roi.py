"""Estimate DeepSeek token spend from recorded API responses; no network calls."""
import json
from pathlib import Path

import report_azure_roi as R

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {'exact': ROOT/'runs/deepseek_exact', 'native': ROOT/'runs/deepseek_native'}


def compute(root, parts):
    rows = sum((R.read(root/f'azure_bench_{part}/lenient/episodes.partial.jsonl') for part in parts), [])
    calls = [c for r in rows for c in r['api_calls']]
    inp = sum(c['usage'].get('prompt_tokens', 0) for c in calls)
    cached = sum((c['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) for c in calls)
    writes = sum((c['usage'].get('prompt_tokens_details') or {}).get('cache_write_tokens', 0) for c in calls)
    out = sum(c['usage'].get('completion_tokens', 0) for c in calls)
    assert 0 <= cached <= inp
    # The published Azure DeepSeek table has no separate cache-write rate.
    dollars = ((inp-cached)*.19 + cached*.028 + out*.51)/1_000_000
    return {'episodes':len(rows), 'provider_blocked':sum(bool(r['score'].get('provider_blocked')) for r in rows),
            'recorded_calls':len(calls), 'input_tokens':inp, 'cached_input_tokens':cached,
            'reported_cache_write_tokens':writes, 'output_tokens':out, 'recorded_estimate_usd':dollars,
            'estimated_usd_per_10000':dollars/len(rows)*10_000}


def main():
    r5 = {name:compute(root, R.PARTS) for name, root in MODEL_PATHS.items()}
    all_suites = {name:compute(root, R.ALL_PARTS) for name, root in MODEL_PATHS.items()}
    saved = json.loads((ROOT/'reports/azure_roi.json').read_text(encoding='utf-8'))
    report = {'deepseek_round5':r5, 'deepseek_all_suites':all_suites,
              'gpt6_round5':saved['cloud'], 'local_t4':saved['local'],
              't4_vm_usd_per_hour':saved['gpu_usd_per_hour'],
              't4_always_730h_usd':saved['continuous_730h_gpu_usd'],
              'volume':10000, 'deepseek_usd_per_million':{'input':.19, 'cached_input':.028, 'output':.51}}
    p = ROOT/'reports/deepseek_roi'
    p.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['# DeepSeek V4 Flash cost scenario', '',
        'The scenario scales observed round-5 usage to 10,000 episodes with the same workload and cache pattern. Prices are illustrative retail rates, not the account invoice; quality and verifier settings differ between systems.', '',
        '| system | recorded round-5 cases | estimated cost / 10,000 |', '| --- | ---: | ---: |']
    for name in ('exact','native'):
        s=r5[name];lines.append(f"| DeepSeek via Azure, {name} | {s['episodes']} | ${s['estimated_usd_per_10000']:.2f} |")
    for name in ('exact','native'):
        s=saved['cloud'][name];lines.append(f"| GPT-6 Luna via Azure, {name} | {s['episodes']} | ${s['estimated_usd_per_10000']:.2f} |")
    for name in ('main+FB+V','candF1+FB+V'):
        s=saved['local'][name];lines.append(f"| {name}, ideal T4 busy time | {s['episodes']} | ${s['estimated_busy_gpu_usd_per_10000']:.2f} |")
    lines += ['', 'DeepSeek pricing assumed $0.19 per million uncached input tokens, $0.028 per million cached input tokens and $0.51 per million output tokens, as published in [Microsoft’s Foundry announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-deepseek-v4-flash-and-v4-pro-in-microsoft-foundry/4515174) (updated July 2026). Current actual prices, region, offer, discounts and billing may differ. The published table does not give a separate cache-write rate; this estimate treats writes as ordinary input. Verify the actual meters and charges in Azure Cost Management.',
        'GPT-6 costs and local T4 assumptions are inherited from [the earlier ROI analysis](azure_roi.md). Local estimates assume a T4 at $0.526/hour that is fully utilized during all billed time and stopped immediately afterward; a continuously allocated 730-hour VM is $383.98 before disk and other infrastructure.', '',
        'Output tokens include any billed reasoning. Recorded usage includes successful calls before a later provider block, but excludes pilots, failed calls and responses lost on interruption. User-specific routing and cache patterns can change the estimates.']
    for name,s in r5.items():
        lines.append(f"- DeepSeek {name}: {s['recorded_calls']} recorded calls over {s['episodes']} episodes ({s['provider_blocked']} provider-blocked), {s['input_tokens']:,} input tokens of which {s['cached_input_tokens']:,} cached, {s['output_tokens']:,} output tokens, {s['reported_cache_write_tokens']:,} reported cache-write tokens; estimated recorded spend ${s['recorded_estimate_usd']:.3f}.")
    lines.append(f"- Across the eight suites and both DeepSeek interfaces: {sum(x['recorded_calls'] for x in all_suites.values()):,} recorded calls, estimated ${sum(x['recorded_estimate_usd'] for x in all_suites.values()):.3f} before pilots and unrecorded attempts.")
    p.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
