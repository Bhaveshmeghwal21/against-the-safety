"""Second-look recovery is opt-in and must fail closed."""
import pytest

from ragharness import protocol, system
from test_system import backend, Script, QUESTION

SEARCH = protocol.format_search('return deadline Nimbus Home')
ABSTAIN = protocol.format_abstain('the return deadline')
GOOD = protocol.format_answer('30 days', ['c00001'])


def make(backend, script, forced, p=0.9, tau_r=0.8):
    force = Script(forced)
    rag = system.RagSystem(None, None, backend, generate=script, support_fn=lambda q, ps, a: p,
                           force_generate=force, recover_threshold=tau_r)
    return rag, force


def test_second_look_turns_a_refusal_into_a_checked_answer(backend):
    rag, force = make(backend, Script(SEARCH, ABSTAIN), GOOD, p=0.93)
    d = rag.ask(QUESTION)
    assert (d.decision, d.answer, d.citations, d.repaired_by, d.model_final) == ('answer', '30 days', ['c00001'], 'second_look', 'abstain')
    assert len(force.calls) == 1


def test_second_look_also_follows_a_repeated_query(backend):
    rag, _ = make(backend, Script(SEARCH, SEARCH), GOOD, p=0.93)
    d = rag.ask(QUESTION)
    assert (d.decision, d.repaired_by, d.model_final) == ('answer', 'second_look', 'invalid')


def test_threshold_is_inclusive_and_blocks_below(backend):
    assert make(backend, Script(SEARCH, ABSTAIN), GOOD, p=0.8)[0].ask(QUESTION).decision == 'answer'
    assert make(backend, Script(SEARCH, ABSTAIN), GOOD, p=0.79)[0].ask(QUESTION).decision == 'abstain'


def test_no_passage_on_screen_means_no_second_look(backend):
    rag, force = make(backend, Script(ABSTAIN), GOOD, p=0.99)
    assert rag.ask(QUESTION).decision == 'abstain' and force.calls == []


@pytest.mark.parametrize('forced', ['garbage', protocol.format_answer('30 days', ['c99999']),
                                     protocol.format_answer('90 days', ['c00001']), ABSTAIN])
def test_unusable_forced_text_stays_a_refusal(backend, forced):
    d = make(backend, Script(SEARCH, ABSTAIN), forced, p=0.99)[0].ask(QUESTION)
    assert d.decision == 'abstain' and d.repaired_by is None and d.reason


def test_withheld_answer_gets_no_second_look(backend):
    rag, force = make(backend, Script(SEARCH, GOOD), GOOD, p=0.2)
    d = rag.ask(QUESTION)
    assert d.repaired_by == 'self_check' and force.calls == []


def test_off_by_default(backend):
    rag = system.RagSystem(None, None, backend, generate=Script(SEARCH, ABSTAIN), support_fn=lambda q, ps, a: 0.99)
    assert rag.ask(QUESTION).decision == 'abstain'


def test_recovery_context_replays_only_prior_turns(tmp_path, monkeypatch):
    import json
    from scripts import build_recover_contexts as recover

    class Corpus:
        def chunk(self, cid):
            assert cid == 'c00001'
            return {'title': 'Return deadline', 'text': 'The return deadline is 30 days.'}

    monkeypatch.setattr(recover.RS, 'world', lambda kind: (Corpus(), {}))
    folder = tmp_path / 'main_bench_sample' / 'lenient'
    folder.mkdir(parents=True)
    transcript = {'qid': 'q1', 'variant': 'open', 'question': QUESTION, 'max_chars': 800,
                  'provided_evidence': [], 'final': {'kind': 'abstain'}, 'steps': [
                      {'kind': 'search', 'assistant': SEARCH, 'observation': '[chunk_id=c00001] Return deadline\n30 days',
                       'retrieved': [{'chunk_id': 'c00001'}]},
                      {'kind': 'abstain', 'assistant': ABSTAIN, 'observation': None, 'retrieved': []}]}
    (folder / 'transcripts.jsonl').write_text(json.dumps(transcript) + '\n', encoding='utf-8')
    result = recover.contexts(tmp_path, {'main'})
    assert len(result) == 1
    prompt = result[0]['prompt']
    assert prompt[1]['content'] == QUESTION
    assert prompt[2]['content'] == SEARCH
    assert prompt[3]['content'] == protocol.format_tool_result(transcript['steps'][0]['observation'])
    assert ABSTAIN not in [message['content'] for message in prompt]
