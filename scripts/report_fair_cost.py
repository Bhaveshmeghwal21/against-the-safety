"""Estimate the fair reruns' token cost using the recorded earlier rate assumptions."""
import json
from pathlib import Path
import report_azure_roi as gpt
import report_deepseek_roi as deepseek

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {
        'gpt_pilot': gpt.cloud(ROOT / 'runs/azure_native_fair_pilot'),
        'gpt_full': gpt.cloud(ROOT / 'runs/azure_native_fair'),
        'deepseek_full': deepseek.compute(ROOT / 'runs/deepseek_native_fair', gpt.PARTS),
        'note': 'Estimates use the previously recorded per-token prices, not an Azure invoice. '
                'Failed or interrupted unrecorded requests, regional pricing and discounts can differ.'}
    total = (report['gpt_pilot']['estimated_recorded_usd'] + report['gpt_full']['estimated_recorded_usd']
             + report['deepseek_full']['recorded_estimate_usd'])
    report['estimated_recorded_total_usd'] = total
    (ROOT / 'reports/fair_rerun_cost.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
