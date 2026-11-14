"""Report a completed Round 6 test after the committed development selection.

This script does not start a model or change the registered test. It refuses to
read test outputs until Amendment 3 (development selection) and the exact selection report are committed.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

import report_azure_compare as azure
import report_round5 as r5
import rescore_round5 as RS
import select_round5 as old_select
from report_reference_policies import always_refuse, answered_stats
from select_round6 import score_transcript, valid, verdicts

ROOT = Path(__file__).resolve().parents[1]
PARTS = ('r6t_squad', 'r6t_hotpot', 'r6t_rgb')


def committed_selection(path):
    prereg = 'research/2026-11-14-round6-preregistration.md'
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    def committed(name):
        cmd = ['git', '-c', f'safe.directory={ROOT.as_posix()}', 'show', f'HEAD:{name}']
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=False)
        if result.returncode:
            raise ValueError(f'Not committed at HEAD: {name}')
        return result.stdout.decode('utf-8').replace('\r\n', '\n')
    committed_prereg = committed(prereg)
    heading = re.search(r'(?mi)^## Amendment 3:[ \t]*development selection[ \t]*$', committed_prereg)
    if not heading:
        raise ValueError('Commit the preregistration Amendment 3: development selection before reading test outputs')
    saved = committed(rel)
    match = re.search(r'(?m)^Selection report SHA256: `([0-9a-f]{64})`[ \t]*$', committed_prereg[heading.end():])
    if not match or match.group(1) != hashlib.sha256(saved.encode('utf-8')).hexdigest():
        raise ValueError('Committed Amendment 3 does not match the committed selection report')
    if saved != path.read_text(encoding='utf-8').replace('\r\n', '\n'):
        raise ValueError('The local development selection differs from the committed selection')
    selection = json.loads(saved)
    chosen = selection.get('chosen_checker')
    if chosen not in ('V2m', 'V2b') or not selection['checker_gates'][chosen]['passes']:
        raise ValueError('The development selection did not adopt a checker; do not open the test')
    return selection


def checked(row, case, probability, threshold):
    if row['final'] != 'answer':
        return old_select.fail_closed([row])[0]
    if case is None:
        raise ValueError('Missing verifier case for an answer')
    accepted = valid(case)
    if accepted:
        if probability is None or not math.isfinite(probability):
            raise ValueError('Missing or invalid verifier probability')
        accepted = probability >= threshold
    if accepted:
        return row
    outcome = 'false_abstain' if row['answerable'] else 'correct_abstain'
    good = not row['answerable']
    return {**row, 'final': 'abstain', 'outcome_v2': outcome, 'outcome_v3': outcome,
            'good_v2': good, 'good_v3': good, 'strong_v2': good, 'strong_v3': good}


def load_actor(root, actor):
    result = []
    for part in PARTS:
        folder = root / f'{actor}_bench_{part}/lenient'
        _, _, rows, mismatches = RS.rescore_run(folder)
        if mismatches:
            raise ValueError(f'Saved and rescored outcomes differ in {folder}')
        for row in rows:
            component = 'hotpot' if part.endswith('hotpot') else 'rgb' if part.endswith('rgb') else row['qtype']
            row.update(part=part, comp=component)
            row.update({f'{field}_v2': row[f'{field}_v3'] for field in ('outcome', 'good', 'strong', 'excluded')})
        result.extend(rows)
    return result


def cluster_intervals(indexed, keys, comparisons):
    _, questions = RS.world('bench:r6t_squad')
    clusters = {}
    for index, (part, qid, _) in enumerate(keys):
        group = questions[qid].paragraphs[0].title.rsplit(' (part ', 1)[0] if part == 'r6t_squad' else qid
        clusters.setdefault(part, {}).setdefault(group, []).append(index)
    comps, template = r5.vectors([indexed[next(iter(indexed))][key] for key in keys], keys)
    grouped = {}
    for name, rows in indexed.items():
        _, vector = r5.vectors([rows[key] for key in keys], keys)
        grouped[name] = {part: np.array([vector[idx].sum(axis=0) for idx in groups.values()]) for part, groups in clusters.items()}
    rng = np.random.default_rng(47)
    totals = {name: np.zeros((3000, template.shape[1])) for name in indexed}
    for part, groups in sorted(clusters.items()):
        picks = rng.integers(0, len(groups), (3000, len(groups)))
        for name in indexed:
            totals[name] += grouped[name][part][picks].sum(axis=1)
    sampled = {name: [r5.stats_from(row, comps) for row in sums] for name, sums in totals.items()}
    observed = {name: azure.r5_stats(rows, keys) for name, rows in indexed.items()}
    return {f'{a}/{b}': {metric: {'difference': observed[a][metric] - observed[b][metric],
                                  'ci95': np.quantile([x[metric] - y[metric] for x, y in zip(sampled[a], sampled[b])], [.025, .975]).tolist()}
                        for metric in ('macro', 'unsupported_rate', 'supported_correct')} for a, b in comparisons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='completed local r6-system test out directory')
    ap.add_argument('--azure', required=True, help='completed GPT hosted eval root')
    ap.add_argument('--deepseek', required=True, help='completed DeepSeek hosted eval root')
    ap.add_argument('--selection', default='reports/round6_selection.json')
    ap.add_argument('--out', default='reports/round6_final')
    args = ap.parse_args()
    selection = committed_selection(ROOT / args.selection)
    selected = selection['chosen_checker']
    threshold = selection['checker_gates'][selected]['chosen']['threshold']
    recover = selection['second_look_enabled']
    recovery_threshold = selection['second_look_gate']['chosen']['threshold'] if recover else None
    root = ROOT / args.run
    cases = {(c['run'], c['qid'], c['variant']): c for c in RS.read(root / 'cases/answers.jsonl')}
    preds = {name: verdicts(root / f'verify/{name}/verdicts_A.jsonl') for name in ('V', selected)}
    raw = {actor: load_actor(root / 'eval', actor) for actor in ('base', 'main', 'candF1')}
    data = dict(raw)
    for actor, checker, tau, label in (('main', 'V', .5, 'main+FB+V'),
                                       ('main', selected, threshold, 'main+FB+V2'),
                                       ('candF1', selected, threshold, 'candF1+FB+V2')):
        built = []
        for row in raw[actor]:
            key = (f"{actor}_bench_{row['part']}", row['qid'], row['variant'])
            built.append(checked(row, cases.get(key), preds[checker].get(key), tau))
        data[label] = built
    if recover:
        contexts = RS.read(root / 'cases/recover_contexts.jsonl')
        recovered = RS.read(root / 'cases/forced_all.jsonl')
        by_id = {c['id']: c for c in recovered}
        if len(by_id) != len(recovered) or len({c['id'] for c in contexts}) != len(contexts):
            raise ValueError('Duplicate forced cases or recovery contexts')
        for actor in ('main', 'candF1'):
            expected_contexts = set()
            for part in PARTS:
                run = f'{actor}_bench_{part}'
                transcripts = RS.read(root / f'eval/{run}/lenient/transcripts.jsonl')
                for transcript in transcripts:
                    shown = transcript.get('provided_evidence', []) or any(s.get('retrieved', []) for s in transcript.get('steps', [])[:-1])
                    if transcript['final']['kind'] in ('abstain', 'invalid') and shown:
                        expected_contexts.add((run, transcript['qid'], transcript['variant']))
            actor_contexts = [c for c in contexts if c['system'] == actor]
            context_keys = [(c['run'], c['qid'], c['variant']) for c in actor_contexts]
            if len(context_keys) != len(set(context_keys)) or set(context_keys) != expected_contexts:
                raise ValueError(f'Incomplete recovery contexts for {actor}')
            expected_ids = {c['id'] for c in actor_contexts}
            if expected_ids:
                attempted = RS.read(root / f'forced/forced_{actor}.jsonl')
                ids = [item['id'] for item in attempted]
                if len(ids) != len(set(ids)) or set(ids) != expected_ids:
                    raise ValueError(f'Incomplete forced outputs for {actor}')
        probabilities = verdicts(root / f'verify_forced/{selected}/verdicts_A.jsonl')
        for actor in ('main', 'candF1'):
            updates = {}
            for context in contexts:
                if context['system'] != actor or context['id'] not in by_id:
                    continue
                candidate = by_id[context['id']]
                key = (candidate['run'], candidate['qid'], candidate['variant'])
                expected = (context['run'] + '_forced', context['qid'], context['variant'])
                if key != expected:
                    raise ValueError('Forced answer identity mismatch')
                if not valid(candidate):
                    continue
                p = probabilities.get(key)
                if p is None or not math.isfinite(p):
                    raise ValueError('Missing forced verifier probability')
                if p < recovery_threshold:
                    continue
                transcript = context['transcript']
                replacement = {**transcript, 'final': {'kind': 'answer', 'answer': candidate['answer'], 'citations': candidate['citations']}}
                score = score_transcript(context['run'], replacement)
                updates[(context['run'], context['qid'], context['variant'])] = score
            altered = []
            for row in data[actor + '+FB+V2']:
                key = (f"{actor}_bench_{row['part']}", row['qid'], row['variant'])
                if key in updates and row['final'] == 'abstain':
                    r = updates[key]
                    r.update(part=row['part'], comp=row['comp'])
                    r.update({f'{field}_v2': r[f'{field}_v3'] for field in ('outcome', 'good', 'strong', 'excluded')})
                    row = r
                altered.append(row)
            data[actor + '+FB+V2+second look'] = altered
    data['always_refuse'] = always_refuse(raw['main'])
    for label, path in (('GPT-6 Luna', args.azure), ('DeepSeek V4 Flash', args.deepseek)):
        data[label] = old_select.fail_closed(load_actor(ROOT / path, 'azure'))
    indexed = {name: azure.by_key(rows) for name, rows in data.items()}
    present = set.intersection(*(set(values) for values in indexed.values()))
    for key in present:
        if len({r[key]['excluded_v3'] for r in indexed.values()}) != 1:
            raise ValueError(f'Model-dependent eligibility: {key}')
    keys = sorted(k for k in present if not indexed['main'][k]['excluded_v3'])
    if not keys:
        raise ValueError('No shared eligible episodes')
    summaries = {name: {**azure.r5_stats(rows, keys), **answered_stats([rows[k] for k in keys])} for name, rows in indexed.items()}
    candidates = [name for name in data if '+FB+V2' in name]
    pairs = [(name, 'main+FB+V') for name in candidates]
    comparisons = cluster_intervals(indexed, keys, pairs)
    for name in candidates:
        comp = comparisons[name + '/main+FB+V']
        comp['headline_passes'] = bool(comp['macro']['ci95'][0] > 0 and comp['unsupported_rate']['difference'] < 0
                                       and comp['unsupported_rate']['ci95'][1] <= .02)
    report = {'scorer': 'v3', 'shared_eligible': len(keys), 'selection': selection,
              'systems': summaries, 'comparisons': comparisons,
              'bootstrap': '3000 paired cluster draws, seed 47; SQuAD by article, HotpotQA and RGB by question'}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pct = lambda v: '—' if v is None else f'{100*v:.1f}%'
    lines = ['# Round 6 sealed test', '', report['bootstrap'], '',
             '| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answered |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, s in summaries.items():
        lines.append(f"| {name} | {pct(s['macro'])} | {pct(s['supported_correct'])} | {s['unsupported']}/{s['hidden_n']} | "
                     f"{pct(s['answered_share'])} | {pct(s['correct_among_answered'])} |")
    lines += ['', '## Preregistered headline rule', '']
    for name in candidates:
        comparison = comparisons[name + '/main+FB+V']
        lines.append(f"- {name}: {'passed' if comparison['headline_passes'] else 'did not pass'}; "
                     f"macro difference {comparison['macro']['difference']:+.3f} {comparison['macro']['ci95']}; "
                     f"hidden-answer-rate difference {comparison['unsupported_rate']['difference']:+.3f} {comparison['unsupported_rate']['ci95']}.")
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    import matplotlib.pyplot as plt
    names = list(summaries)
    figure, axes = plt.subplots(1, 2, figsize=(12, max(4, len(names) * .42)), constrained_layout=True)
    positions = list(range(len(names)))
    axes[0].barh(positions, [summaries[n]['macro'] * 100 for n in names])
    axes[1].barh(positions, [summaries[n]['unsupported_rate'] * 100 for n in names])
    axes[0].set_yticks(positions, names)
    axes[0].invert_yaxis()
    axes[1].set_yticks(positions, [])
    axes[1].invert_yaxis()
    axes[0].set_xlabel('Macro success (%)')
    axes[1].set_xlabel('Answers with hidden evidence (%)')
    figure.suptitle('Round 6: shared eligible episodes')
    target = ROOT / 'docs/figures/round6_final.svg'
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, format='svg')
    plt.close(figure)


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f'Required saved artifact is missing: {exc.filename or exc}') from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
