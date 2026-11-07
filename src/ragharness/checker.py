"""The self-check V of the round-5 system: does the passage text the model cited state its answer?

Two parts, applied in this order (research/2026-11-07-round5-preregistration.md, priority 1):
  1. deterministic checks: a cited chunk that was never shown to the model, or an answer part (split at ";") whose content words are not all in
     the cited passages;
  2. the main adapter, asked once with wording A, replies SUPPORTED or NOT. The score is the probability of the first token SUPPORTED against the
     first token NOT, taken from the next-token logits (no text is generated). Below THRESHOLD the answer is withheld.

The wording and the threshold are fixed by the pre-registration. `scripts/verify_round5.py` is the batch version used for the experiments; a test
asserts that both share the wording and the deterministic functions give the same answers as `scripts/build_verifier_cases.py`.
"""

from __future__ import annotations

from . import rescore
from .retrieval import Hit, render_observation, visible_text

THRESHOLD = 0.5
POSITIVE, NEGATIVE = "SUPPORTED", "NOT"
WORDING_A = ("You are checking whether an answer is backed by the quoted passages.\n\nQuestion: {question}\n\nPassages:\n{passages}\n\n"
             "Proposed answer: {answer}\n\nIs every part of the proposed answer stated in the passages above? Reply with exactly one word: SUPPORTED or NOT.")


def answer_parts(answer: str) -> list[str]:
    return [p.strip() for p in answer.split(";") if p.strip()]


def contained(part: str, text: str) -> bool | None:
    """Every content token of the answer part occurs in the cited text; None when there is nothing to check (yes/no, only stop words)."""
    toks = rescore.tokens(part)
    if not toks or part.strip().lower() in ("yes", "no"):
        return None
    have = set(rescore.tokens(text))
    return all(t in have for t in toks)


def deterministic_flags(answer: str, citations: list[str], shown: list[Hit], max_chars: int | None = 800) -> dict:
    """`shown` are the passages the model saw (pre-supplied and retrieved). Returns the flags and the cited passages as the tool rendered them."""
    by_id = {h.chunk_id: h for h in shown}
    cited = list(citations or [])
    unseen = [c for c in cited if c not in by_id]
    seen = [c for c in dict.fromkeys(cited) if c in by_id]
    hits = [by_id[c] for c in seen]
    passages = render_observation(hits, frozenset(), max_chars) if hits else ""
    cited_text = " ".join(visible_text(h.text, max_chars) for h in hits)
    missing = [p for p in answer_parts(answer or "") if contained(p, cited_text) is False]
    return {"cites_unshown": bool(unseen) or not seen, "answer_part_absent_from_cited": bool(missing), "absent_parts": missing, "passages": passages}


def support_probability(model, tokenizer, question: str, passages: str, answer: str) -> float:
    """Probability that the model replies SUPPORTED rather than NOT to the wording-A question. One forward pass."""
    import torch

    ip = tokenizer.encode(POSITIVE, add_special_tokens=False)[0]
    ineg = tokenizer.encode(NEGATIVE, add_special_tokens=False)[0]
    assert ip != ineg, "the two answer words must start with different tokens"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": WORDING_A.format(question=question, passages=passages, answer=answer)}],
                                           add_generation_prompt=True, return_dict=True)["input_ids"]
    ids = torch.tensor([list(prompt)], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -1, :].float()
    return float(torch.stack([logits[:, ip], logits[:, ineg]], dim=1).softmax(dim=1)[0, 0])
