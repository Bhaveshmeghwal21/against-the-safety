"""Summarize the saved three-suite hosted pilot without making model calls."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    suites = {}
    for part in ('r5t_squad', 'r5t_hotpot', 'r5t_rgb'):
        path = ROOT / 'runs/azure_native_fair_pilot' / f'azure_bench_{part}/lenient/episodes.partial.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != 20 or len({(r['score']['qid'], r['score']['variant']) for r in rows}) != 20:
            raise ValueError(f'Expected 20 unique pilot episodes in {part}')
        outcomes = Counter((r['transcript'] or {}).get('stop', 'provider_blocked') for r in rows)
        suites[part] = {'episodes': len(rows), 'stops': dict(outcomes),
                        'provider_blocked': sum(bool(r['score'].get('provider_blocked')) for r in rows)}
    exceeded = sum(s['stops'].get('budget_exceeded', 0) for s in suites.values())
    report = {'episodes': 60, 'search_budget_exceeded': exceeded,
              'provider_blocked': sum(s['provider_blocked'] for s in suites.values()),
              'pilot_passed': exceeded < 2, 'suites': suites,
              'note': 'Provider blocks were not treated as model failures. This pilot used GPT-6 Luna only.'}
    (ROOT / 'reports/fair_pilot.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['pilot_passed']:
        raise SystemExit('Pilot stop rule failed; do not start the full rerun.')


if __name__ == '__main__':
    main()
