"""Cost scenario from saved round-5 runs and metered Azure response usage.

Run after the evaluations complete. This does not fetch prices or call a model.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ('r5t_squad', 'r5t_hotpot', 'r5t_rgb')
ALL_PARTS = PARTS + ('2wiki', 'musique', 'squad2', 'rgb', 'rgb_int')
AZURE = {'exact': ROOT / 'runs/azure_round5', 'native': ROOT / 'runs/azure_round5_native'}
VOLUME, T4_RATE = 10_000, .526


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def cloud(root, parts=PARTS):
    rows = sum((read(root / f'azure_bench_{p}/lenient/episodes.partial.jsonl') for p in parts), [])
    calls = [call for row in rows for call in row['api_calls']]
    usage = [c['usage'] for c in calls]
    p = sum(x.get('prompt_tokens', 0) for x in usage)
    cached = sum(x.get('prompt_tokens_details', {}).get('cached_tokens', 0) for x in usage)
    write = sum(x.get('prompt_tokens_details', {}).get('cache_write_tokens', 0) for x in usage)
    output = sum(x.get('completion_tokens', 0) for x in usage)
    assert 0 <= cached + write <= p, (p, cached, write)
    # Categories counted once, using Microsoft's published Global Standard short-context rates.
    dollars = ((p-cached-write)*.10 + cached*.01 + write*.125 + output*.50)/1_000_000
    return {'episodes': len(rows), 'blocked': sum(bool(r['score'].get('provider_blocked')) for r in rows),
            'calls': len(calls), 'input_tokens': p, 'cached_read_tokens': cached, 'cache_write_tokens': write,
            'output_tokens_including_reasoning': output, 'estimated_recorded_usd': dollars,
            'estimated_usd_per_10000': dollars / len(rows) * VOLUME}


def local(model, root, cases_file, summary_file):
    rows = sum((read(root / f'{model}_bench_{p}/lenient/scores.jsonl') for p in PARTS), [])
    cases = [c for c in read(cases_file) if c['run'] in {f'{model}_bench_{p}' for p in PARTS}]
    needs_v = sum(not (c['deterministic']['cites_unshown'] or c['deterministic']['answer_part_absent_from_cited']) for c in cases)
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    verifier_s = summary['wordings']['A']['seconds'] / summary['cases']
    actor_s = sum(r['seconds'] for r in rows)
    generation_tokens = sum(r['output_tokens'] for r in rows)
    gpu_s = actor_s + needs_v * verifier_s
    return {'episodes':len(rows), 'actor_seconds_amortized':actor_s, 'actor_output_tokens':generation_tokens,
            'effective_actor_output_tokens_per_second':generation_tokens/actor_s,
            'answer_cases':len(cases), 'verifier_calls_estimated':needs_v, 'verifier_seconds_per_case_batched':verifier_s,
            'estimated_gpu_seconds_per_episode':gpu_s/len(rows),
            'estimated_gpu_hours_per_10000':gpu_s/len(rows)*VOLUME/3600,
            'estimated_busy_gpu_usd_per_10000':gpu_s/len(rows)*VOLUME/3600*T4_RATE}


def main():
    cloud_data = {name: cloud(root) for name, root in AZURE.items()}
    cloud_all = {name: cloud(root, ALL_PARTS) for name, root in AZURE.items()}
    local_data = {
      'main+FB+V':local('main',ROOT/'runs/account2_round5/ats-eval-jobs-r5test-ref/out/eval',
          ROOT/'data/processed/verifier/cases_r5test_refs.jsonl',ROOT/'runs/account2_round5/ats-verify-main-r5-r5test-refs/out/verify/main/summary.json'),
      'candF1+FB+V':local('candF1',ROOT/'runs/account2_round5/ats-eval-jobs-r5test-final/out/eval',
          ROOT/'data/processed/verifier/cases_r5test_final.jsonl',ROOT/'runs/account2_round5/ats-verify-main-r5-r5test-final/out/verify/main/summary.json'),
      # candA has no verifier run of its own. Its answers are in the main verifier run's case file, so that run's average
      # time per case is used. This is an estimate (actor time measured, checker time averaged), the same method as the rows above.
      'candA+FB+V':local('candA',ROOT/'runs/account2_round5/ats-eval-jobs-r5test-ref/out/eval',
          ROOT/'data/processed/verifier/cases_r5test_refs.jsonl',ROOT/'runs/account2_round5/ats-verify-main-r5-r5test-refs/out/verify/main/summary.json')}
    local_scores = json.loads((ROOT/'reports/round5_corrected.json').read_text(encoding='utf-8'))['systems']
    cloud_scores = json.loads((ROOT/'reports/azure_round5_comparison_corrected.json').read_text(encoding='utf-8'))['systems']
    for name, value in local_data.items():
        value['cost_per_1000_successes'] = value['estimated_busy_gpu_usd_per_10000'] / 10 / local_scores[name]['macro']
    for name, value in cloud_data.items():
        value['cost_per_1000_successes'] = value['estimated_usd_per_10000'] / 10 / cloud_scores[f'azure_{name}+FB']['macro']
    training = json.loads((ROOT/'runs/kaggle/ats-train-both/out/main/train_report.json').read_text(encoding='utf-8'))
    result={'volume_per_month':VOLUME, 'gpu_usd_per_hour':T4_RATE, 'continuous_730h_gpu_usd':730*T4_RATE,
            'training_report':training, 'cloud':cloud_data, 'all_eight_suites':cloud_all, 'local':local_data}
    out=ROOT/'reports/azure_roi'
    out.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Azure versus self-hosting: illustrative ROI', '',
      'Workload: 10,000 episodes per month with the same mix, average token lengths, tool calls and cache-hit pattern as the three round-5 test sets. The quality comparison is in [the corrected replay](azure_round5_comparison_corrected.md); the systems have different quality and the estimates are not a like-for-like outcome guarantee.', '',
      '**Checked 13 November 2026:** These estimates use all 938 recorded episodes per system, rather than the 931 eligible quality-comparison cases. Scoring corrections leave recorded token usage and GPU time unchanged, so the dollar estimates are unchanged. They describe the saved workload and checker decisions; a new run with revised prompts or checker coverage needs new metering.', '',
      'Cost per 1,000 successful episodes divides the cost per 1,000 episodes by the corrected macro success rate. Macro weights the four components equally, so this is a workload-specific approximation.', '',
      '| System | Recorded episodes | Estimated GPU hours / 10,000 | Estimated compute or token cost / 10,000 | Cost per 1,000 successful episodes |', '| --- | ---: | ---: | ---: | ---: |']
    for n,v in local_data.items():
        lines.append(f"| {n}, busy T4 | {v['episodes']} | {v['estimated_gpu_hours_per_10000']:.2f} | ${v['estimated_busy_gpu_usd_per_10000']:.2f} | ${v['cost_per_1000_successes']:.2f} |")
    for n,v in cloud_data.items():
        lines.append(f"| Azure GPT-6 Luna, {n} + fail-closed | {v['episodes']} | provider-managed | ${v['estimated_usd_per_10000']:.2f} | ${v['cost_per_1000_successes']:.2f} |")
    lines += ['', '## Inputs and limitations', '',
      f'- A Linux Standard_NC4as_T4_v3 GPU VM in East US is estimated at **${T4_RATE:.3f}/hour** using the [Azure Retail Prices API](https://prices.azure.com/api/retail/prices), queried for the East US VM SKU and consumption price; 730 hours continuously costs **${730*T4_RATE:.2f}/month** for the VM. The table assumes perfectly stoppable, fully utilized busy GPU hours; a real always-on or lightly used service can cost much more. It excludes disk, network, storage, retrieval server, CPUs beyond the VM, operations and taxes; availability and startup credits do not guarantee quota.',
      '- Saved actor times apportion each model batch across episodes; saved verifier runtime is a separately batched average. The estimated checker workload counts answers that passed deterministic checks, matching the product pipeline. These are throughput estimates, not individual-request latency; batch size, prompt length and utilization affect cost.',
      '- Azure uses Microsoft published Global Standard short-context prices per million tokens: $0.10 ordinary input, $0.01 cached input, $0.125 cache writes and $0.50 output. Each token is assigned once to its category; completion tokens include reasoning. Regional deployment, contract terms, credits and current rates can differ. [GPT-6 prices](https://azure.microsoft.com/en-us/blog/gpt-6-astra-sol-and-luna-for-production-agents-in-microsoft-foundry/), [cache behavior](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching), [token categories](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing).',
      '- Estimates include recorded successful API calls, including successful calls before a later provider block. Failed requests, pilots, and any in-flight calls lost in an interrupted run may be absent. This is not the Azure invoice. Check actual usage and billing in the Azure portal. Cache behavior at other traffic levels may differ.',
      '- The local model has a non-commercial license; this is an illustrative non-commercial workload, not an assertion that commercial deployment is permitted.', '']
    for n,v in local_data.items():
        lines.append(f"- {n}: actor {v['effective_actor_output_tokens_per_second']:.1f} output tokens/s effective batched throughput; actor plus estimated verifier {v['estimated_gpu_seconds_per_episode']:.2f} GPU seconds/episode; {v['verifier_calls_estimated']}/{v['episodes']} episodes invoke the checker in this approximation.")
    for n,v in cloud_data.items():
        lines.append(f"- Azure {n}: {v['calls']} recorded calls, {v['input_tokens']:,} input tokens ({v['cached_read_tokens']:,} cache reads, {v['cache_write_tokens']:,} cache writes), {v['output_tokens_including_reasoning']:,} output tokens; recorded usage estimated ${v['estimated_recorded_usd']:.3f} for {v['episodes']} episodes, including {v['blocked']} provider-blocked episodes.")
    lines += ['', f"Across all eight suites and both interfaces, the {sum(v['calls'] for v in cloud_all.values()):,} recorded calls are estimated at **${sum(v['estimated_recorded_usd'] for v in cloud_all.values()):.3f}** at the assumed prices. This excludes pilots and any unrecorded attempts or calls; actual billing may differ."]
    lines += ['', f"Main-adapter training report: {training['wall_seconds']/3600:.2f} hours for one historical training run, equivalent to ${training['wall_seconds']/3600*T4_RATE:.2f} at this hypothetical T4 rental rate. This is not the actual Kaggle bill and excludes all other experiments, development, data preparation, and evaluation."]
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
