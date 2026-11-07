"""Paired evidence decisions; no model predictions or evaluation labels are consumed."""

from __future__ import annotations

import hashlib
import random

from . import builder, episodes, policy, protocol, round3
from .data import Question
from .retrieval import visible_text


def public_queries(q: Question, visible: list) -> list[str]:
    candidates = [q.question, policy.keyword_query(q.question, 8), policy.keyword_query(q.question, 12),
                  policy.keyword_query(q.question, 4)]
    # A teacher may choose a useful title only when the title is already public.
    text = q.question + " " + " ".join(h.text for h in visible)
    candidates += [policy.mention(t) for t in q.support if policy.contains(text, policy.mention(t))]
    if ", and " in q.question:
        a, b = q.question.split(", and ", 1)
        candidates += [a + "?", b]
    unique, seen = [], set()
    for candidate in candidates:
        norm = " ".join(candidate.lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(candidate)
    return unique


def hidden_forms(q: Question, titles: list[str]) -> list[str]:
    if q.answer_parts:
        return [a for t in titles for a in q.answer_parts[list(q.support).index(t)] if a]
    return [q.answer, *q.aliases]


def build_pair(q, corp, search, tool, depth: int, hidden_titles: list[str]):
    """Return two actual-tool trajectories and their supervised decision index.

    Before the paired decision, the positive environment reveals the missing support on
    search `depth`, while the negative environment keeps it inaccessible. At depth zero
    the difference is in the supplied passages. Temporary availability is an explicit
    training intervention, analogous to the existing first-search-masked family.
    """
    if depth not in range(4) or not builder.support_visible(q, corp, tool.max_chars):
        return None
    support = corp.support_chunks(q)
    copies = frozenset(c for t in hidden_titles for c in corp.title_copies(t, support[t]))
    negatives = round3.hard_negatives(q, corp, search, corp.support_copies(q), n=8)
    if len(negatives) < 4 or len(support) > 3:
        return None
    full = list(support.values()) + negatives[:4 - len(support)]
    remaining = iter([c for c in negatives if c not in full])
    gap = [next(remaining) if c in {support[t] for t in hidden_titles} else c for c in full]
    order = list(range(4))
    random.Random(int(hashlib.sha1(q.qid.encode()).hexdigest()[:8], 16)).shuffle(order)
    full, gap = [[builder._hit(corp, ids[i]) for i in order] for ids in (full, gap)]
    queries = public_queries(q, gap)
    if len(queries) < 3:
        return None
    needed = set(support.values())
    known = {h.chunk_id for h in gap}
    finishing = next((s for s in queries if needed <= known | {h.chunk_id for h in search.search(s, 3)}), None)
    if depth and finishing is None:
        return None
    prefix = [s for s in queries if s != finishing]
    if len(prefix) < max(0, depth - 1):
        return None
    scheduled = prefix[:depth - 1] + [finishing] if depth else []
    hidden_answer_forms = hidden_forms(q, hidden_titles)
    output = []
    final_extra_mask = frozenset()
    for label in ("positive", "negative"):
        variant = f"r4_d{depth}_h{'-'.join(str(list(support).index(t)) for t in hidden_titles)}_{label}"
        prefill = full if label == "positive" and depth == 0 else gap
        ep = episodes.Episode(q.qid, q.question, variant, copies if label == "negative" else frozenset(),
                              prefill=prefill, max_chars=tool.max_chars)
        observed = {h.chunk_id: h for h in prefill}
        sequence = list(scheduled)
        if label == "negative":
            sequence += [s for s in queries if s not in sequence][:3 - depth]
        for i, query in enumerate(sequence):
            mask = copies if label == "negative" or i < depth - 1 else frozenset()
            if i == depth - 1 and depth:
                if label == "positive":
                    positive_hits = search.search(query, 3)
                    positive_ids = {h.chunk_id for h in positive_hits}
                    novelty = len(positive_ids - set(observed))
                    extra = set()
                    for _ in range(30):
                        negative_hits = search.search(query, 3, copies | frozenset(extra))
                        negative_ids = {h.chunk_id for h in negative_hits}
                        if len(negative_ids - set(observed)) == novelty:
                            break
                        choices = [h.chunk_id for h in negative_hits if h.chunk_id in observed and h.chunk_id not in positive_ids]
                        if not choices or len(negative_ids - set(observed)) > novelty:
                            return None
                        extra.add(choices[-1])
                    else:
                        return None
                    final_extra_mask = frozenset(extra)
                mask |= final_extra_mask
            tool.start_episode(mask, frozenset(), frozenset(observed))
            result = tool.execute(query=query)
            text = protocol.format_search(query)
            ep.steps.append(episodes.Step(text, protocol.parse_turn(text), result.output, list(tool.last_hits)))
            observed.update({h.chunk_id: h for h in tool.last_hits})
        ep.complete = needed <= set(observed)
        if label == "positive":
            if not ep.complete:
                return None
            final = protocol.format_answer(q.answer, list(support.values()))
        else:
            if ep.complete or len(sequence) != 3:
                return None
            if any(builder.answer_in_text(a, visible_text(h.text, tool.max_chars)) is True
                   for a in hidden_answer_forms for h in observed.values()):
                return None
            final = protocol.format_abstain("A required fact is missing from the available passages.")
        ep.steps.append(episodes.Step(final, protocol.parse_turn(final)))
        ep.stop_reason = ep.final.turn.kind
        output.append(ep)
    # Prefixes at depths >0 must differ only in the final search result, not the question or earlier turns.
    if depth:
        assert output[0].user_text == output[1].user_text
        assert [(s.text, s.observation) for s in output[0].steps[:depth - 1]] == [
            (s.text, s.observation) for s in output[1].steps[:depth - 1]]
        previously_seen = {h.chunk_id for h in output[0].prefill}
        previously_seen.update(h.chunk_id for s in output[0].steps[:depth - 1] for h in s.hits)
        assert [len({h.chunk_id for h in ep.steps[depth - 1].hits} - previously_seen) for ep in output] == [
            len({h.chunk_id for h in output[0].steps[depth - 1].hits} - previously_seen)] * 2
    assert output[0].steps[depth].turn.kind == "answer"
    assert output[1].steps[depth].turn.kind == ("abstain" if depth == 3 else "search")
    return output


def aspect_coverage(q, corp, ep) -> dict:
    """Bright-Pro-inspired, equally weighted required-evidence aspects, not its official score.

    Each required support is one aspect; alternate copies can satisfy that aspect.
    Evidence coverage does not establish semantic entailment of arbitrary extra claims.
    """
    supports = corp.support_chunks(q)
    if not supports:
        return {"retrieved_aspect_recall": None, "cited_aspect_recall": None, "all_aspects_cited": None}
    shown = {h.chunk_id for h in ep.retrieved_chunks()}
    cited = set(ep.final.turn.citations or []) if ep.final.turn.kind == "answer" else set()
    facets = [set(corp.title_copies(t, c)) | {c} for t, c in supports.items()]
    retrieved = sum(bool(f & shown) for f in facets) / len(facets)
    citation = sum(bool(f & shown & cited) for f in facets) / len(facets)
    return {"retrieved_aspect_recall": retrieved, "cited_aspect_recall": citation,
            "all_aspects_cited": citation == 1.0}
