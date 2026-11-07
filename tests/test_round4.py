"""Check actual-tool pair construction, partial support and train/held-out separation."""

import json
from pathlib import Path

import pytest

from ragharness import corpus as C, episodes, protocol, qlora, round4, training
from ragharness.data import Paragraph, Question
from ragharness.retrieval import MaskedSearch, open_verified_backend
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def environment(tmp_path):
    paragraphs = [Paragraph('Return deadline', ['The return deadline for Nimbus Home devices is 30 days.'])]
    paragraphs += [Paragraph(f'Return topic {i}', [f'Nimbus Home devices return requests involve support service item {i}.']) for i in range(9)]
    q = Question('synthetic-pair', 'What is the return deadline for Nimbus Home devices?', '30 days', 'single', 'test',
                 paragraphs, {'Return deadline': [0]})
    corp = C.build_corpus([q], 47)
    db = tmp_path / 'corpus.sqlite'
    C.build_index(corp, db)
    backend = open_verified_backend(db, corp)
    return q, corp, MaskedSearch(backend), CorpusSearchTool(backend, top_k=3)


@pytest.mark.parametrize('depth', range(4))
def test_pair_changes_fact_availability_at_the_same_decision_depth(environment, depth):
    q, corp, search, tool = environment
    pair = round4.build_pair(q, corp, search, tool, depth, ['Return deadline'])
    assert pair is not None
    pos, neg = pair
    assert pos.complete and not neg.complete
    assert pos.steps[depth].turn.kind == 'answer'
    assert neg.steps[depth].turn.kind == ('search' if depth < 3 else 'abstain')
    assert len(pos.prefill) == len(neg.prefill) == 4
    assert '30 days' not in neg.user_text
    assert all('30 days' not in (s.observation or '') for s in neg.steps)
    if depth:
        assert pos.user_text == neg.user_text
        assert '30 days' in (pos.steps[depth - 1].observation or '')
        seen = {h.chunk_id for h in pos.prefill}
        seen.update(h.chunk_id for s in pos.steps[:depth - 1] for h in s.hits)
        assert len({h.chunk_id for h in pos.steps[depth - 1].hits} - seen) == len({h.chunk_id for h in neg.steps[depth - 1].hits} - seen)
    logs = [episodes.to_eventlog(ep) for ep in pair]
    hf = qlora.to_hf_rows([r for log in logs for r in log.as_dataset_rows('sft')], protocol.system_prompt(tool))
    assert hf


def test_aspect_coverage_requires_every_part(environment):
    q, corp, _, _ = environment
    q.support['Return topic 0'] = [0]
    support = corp.support_chunks(q)
    hits = [__import__('ragharness.builder', fromlist=['_hit'])._hit(corp, c) for c in support.values()]
    text = protocol.format_answer('30 days', [support['Return deadline']])
    ep = episodes.Episode(q.qid, q.question, 'test', frozenset(), prefill=hits,
                          steps=[episodes.Step(text, protocol.parse_turn(text))])
    coverage = round4.aspect_coverage(q, corp, ep)
    assert coverage['retrieved_aspect_recall'] == 1
    assert coverage['cited_aspect_recall'] == .5
    assert not coverage['all_aspects_cited']


@pytest.mark.skipif(not (ROOT / 'reports/round4_data.json').exists(), reason='round-4 data not prepared')
def test_saved_pairs_are_complete_and_groups_disjoint():
    report = json.loads((ROOT / 'reports/round4_data.json').read_text())
    groups = report['groups']
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not set(groups[a]['articles']) & set(groups[b]['articles'])
        assert not set(groups[a]['ids']) & set(groups[b]['ids'])
    rows = [json.loads(l) for l in (ROOT / 'data/processed/training/candE.jsonl').read_text(encoding='utf-8').splitlines()]
    pairs = {}
    for r in rows:
        assert r['meta']['real_tokens'] <= 2560
        if r['meta']['split'] == 'train4':
            pairs.setdefault(r['meta']['pair_id'], []).append(r)
    assert pairs
    assert all(len(v) == 2 for v in pairs.values())
    keys = training.read_keys(ROOT / 'data/processed/training/candE.jsonl')
    logs, metas = [], []
    for split in ('train', 'train4'):
        ll, mm = training.load_episodes(ROOT / 'data/processed/episodes', split)
        logs += ll
        metas += mm
    logs, metas = training.episodes_for_keys(logs, metas, keys)
    actual = qlora.to_hf_rows([r for log in logs for r in log.as_dataset_rows('sft')],
                            training.read_system(ROOT / 'data/processed/training/main.jsonl'), metas, keys)
    pair = lambda r: json.dumps([r['prompt'], r['completion']], sort_keys=True)
    assert sorted(map(pair, actual)) == sorted(map(pair, rows))


def test_higher_score_cannot_bypass_unsupported_answer_guard():
    from copy import deepcopy
    from scripts.select_round4 import choose
    base = {'macro': .50, 'components': {'hotpot': .67}, 'unsupported': 4, 'unsupported_rate': 4 / 132,
            'unsupported_by_family': {'partial': 1, 'fully_missing': 3}, 'invalid': .02,
            'supported_correct': .60, 'hotpot_unsupported': 2}
    records = {n: deepcopy(base) for n in ('main', 'candE1', 'candE2', 'candF1', 'candF2')}
    records['candE1'].update(macro=.90, unsupported=20, unsupported_rate=20 / 132)
    records['candF1'].update(macro=.60)
    assert choose(records) == 'candF1'
    records['candF1']['unsupported_by_family']['partial'] = 2
    assert choose(records) is None


def test_qualifying_zero_error_checkpoint_is_preferred():
    from copy import deepcopy
    from scripts.select_round4 import choose
    base = {'macro': .50, 'components': {'hotpot': .67}, 'unsupported': 4, 'unsupported_rate': 4 / 132,
            'unsupported_by_family': {'partial': 1, 'fully_missing': 3}, 'invalid': .02,
            'supported_correct': .60, 'hotpot_unsupported': 2}
    records = {n: deepcopy(base) for n in ('main', 'candE1', 'candE2', 'candF1', 'candF2')}
    records['candE1']['macro'] = .65
    records['candF1'].update(macro=.60, unsupported=0, unsupported_rate=0, hotpot_unsupported=0,
                            unsupported_by_family={'partial': 0, 'fully_missing': 0})
    assert choose(records) == 'candF1'
    records['candF1']['macro'] = .52  # zero errors cannot win by sacrificing the required utility improvement
    assert choose(records) == 'candE1'


def test_informed_transfer_vetoes_risk_moved_to_other_sources():
    from copy import deepcopy
    from scripts.audit_round4_transfer import deployment_guard
    row = {'unsupported': 2, 'supported_correct': 10, 'success': .5}
    sets = {d: {n: deepcopy(row) for n in ('main', 'candE1')}
            for d in ('hotpot', '2wiki', 'musique', 'squad2', 'rgb', 'rgb_int')}
    assert deployment_guard(sets, 'candE1')['passes']
    sets['hotpot']['candE1']['unsupported'] += 1
    check = deployment_guard(sets, 'candE1')
    assert check['known_risk_sources_preserved']
    assert not check['passes']
    sets['hotpot']['candE1']['unsupported'] -= 1
    sets['hotpot']['candE1']['success'] = .4
    assert not deployment_guard(sets, 'candE1')['passes']
    sets['hotpot']['candE1']['success'] = .7
    sets['squad2']['candE1']['supported_correct'] -= 1
    assert not deployment_guard(sets, 'candE1')['passes']
