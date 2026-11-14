"""Compare completed format-aware hosted runs with the saved systems, scorer v3."""
import json
from pathlib import Path

import remediate_audit as audit
import report_azure_compare as azure
import report_deepseek_compare as deepseek
import azure_rag_eval as runner
from report_reference_policies import answered_stats, sync_wrapped

ROOT = Path(__file__).resolve().parents[1]


def validate_fair_config(config, expected_model):
    expected = {'model': expected_model, 'mode': 'native', 'search_budget': 3, 'top_k': 3,
                'passage_chars': 800, 'native_instructions': runner.NATIVE_INSTRUCTIONS}
    bad = [name for name, value in expected.items() if config.get(name) != value]
    if bad:
        raise ValueError('Fair-run configuration mismatch: ' + ', '.join(bad))


def main():
    specs = {
        'main': (azure.R5REF, 'main'), 'candF1': (azure.R5ALT, 'candF1'),
        'gpt_old': (deepseek.REMOTE['azure_native'], 'azure'),
        'deepseek_old': (deepseek.REMOTE['azure_deepseek_native'], 'azure'),
        'gpt_fair': (ROOT / 'runs/azure_native_fair', 'azure'),
        'deepseek_fair': (ROOT / 'runs/deepseek_native_fair', 'azure'),
    }
    for name, (root, label) in specs.items():
        for part in azure.PARTS:
            folder = root / f'{label}_bench_{part}/lenient'
            for filename in ('scores.jsonl', 'transcripts.jsonl'):
                path = folder / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
            if name in ('gpt_fair', 'deepseek_fair'):
                config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
                validate_fair_config(config, runner.VERSION if name == 'gpt_fair' else 'DeepSeek-V4-Flash')
                rows = runner.read_jsonl(folder / 'episodes.partial.jsonl')
                if len(rows) != config['episodes'] or len({(r['score']['qid'], r['score']['variant']) for r in rows}) != len(rows):
                    raise ValueError(f'Incomplete or duplicate fair-run episodes: {folder}')
    sources = audit.squad_sources()
    cases, verdicts = deepseek.cases_and_verdicts()
    rows = {name: sum((audit.load(root / f'{label}_bench_{part}/lenient', part, sources)[2]
                       for part in azure.PARTS), []) for name, (root, label) in specs.items()}
    all_wrapped = audit.wrappers(rows, cases, verdicts)
    names = ['main+FB+V', 'candF1+FB+V']
    names += [name + suffix for name in ('gpt_old', 'gpt_fair', 'deepseek_old', 'deepseek_fair') for suffix in ('', '+FB')]
    systems = {name: sync_wrapped(all_wrapped[name]) for name in ['main', *names]}
    indexed, keys, coverage = audit.fixed_common(systems)
    summary = {name: {**azure.r5_stats(indexed[name], keys), **answered_stats([indexed[name][key] for key in keys])} for name in names}
    pairs = [(model + '+FB', base) for model in ('gpt_fair', 'deepseek_fair') for base in ('main+FB+V', 'candF1+FB+V')]
    pairs += [('gpt_fair+FB', 'gpt_old+FB'), ('deepseek_fair+FB', 'deepseek_old+FB')]
    differences = audit.clustered(indexed, keys, pairs)
    report = {'scorer': 'v3', 'coverage': coverage, 'systems': summary, 'comparisons': differences,
              'method': 'Shared eligible episodes; SQuAD resampled by article, other suites by question; seed 47, 3000 draws. '
                        'The adapters learned the answer format from training; hosted models were explicitly told it in the fair run. '
                        'The original round-five test was already opened, so this is a retrospective system comparison, not a fresh blind test. '
                        'Replay of saved local searches matched current results, but cannot establish equivalence for unseen queries.'}
    out = ROOT / 'reports/fair_rerun'
    out.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# Format-aware hosted comparison', '', report['method'], '',
             f"Shared eligible episodes: {coverage['common']}. Provider-blocked or missing cases are excluded.", '',
             '| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answered | Wrong or unsupported shown / all cases |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    pct = lambda x: '—' if x is None else f'{100*x:.1f}%'
    for name, value in summary.items():
        lines.append(f"| {name} | {pct(value['macro'])} | {pct(value['supported_correct'])} | "
                     f"{value['unsupported']}/{value['hidden_n']} | {pct(value['answered_share'])} | {pct(value['correct_among_answered'])} | "
                     f"{pct(value['wrong_or_unsupported_shown_share'])} |")
    lines += ['', '## Paired differences', '', 'First system minus second; 95% clustered bootstrap intervals.']
    for pair, metrics in differences.items():
        parts = [f"{metric}: {value['difference']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]"
                 for metric, value in metrics.items()]
        lines.append(f"- {pair}: " + '; '.join(parts))
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:8]))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
