"""Checks before any GPU time on the multilingual sets (research/2026-11-15-multilingual-plan.md, section 8, checks 2 and 3). CPU only.

  python scripts/check_xquad.py [--langs hi,en,es] [--tokenizer .cache/tiny-aya-global] [--out reports/xquad_checks.json]

Check 2, search reaches the evidence: the scripted oracle (the perfect policy that built the training data) must find the answer passage within
three searches in at least 80% of the evidence-available episodes of every language. Also reported: how often the very first search
(the question as the query) already puts the answer passage in the top three.
Check 3, prompt length: the worst case is nine passages shown (three searches, three passages each, 800 characters) plus the system prompt,
the longest question and three search turns. It is counted with the Tiny Aya tokenizer and compared with the generator's 4,096-token input limit.
Check 4, Hindi index probe: a Hindi query returns Hindi passages, so the tokenizer option works.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from ragharness import builder, corpus as corpus_mod, data, evaluation, protocol
from ragharness.retrieval import MaskedSearch, open_verified_backend, render_observation
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
INPUT_LIMIT = 4096


def world(lang: str):
    d = ROOT / "data" / "bench" / f"xq_{lang}"
    corp = corpus_mod.load_corpus(d / "corpus.json")
    return corp, open_verified_backend(d / "corpus.sqlite", corp), data.read_questions(d / "questions.jsonl")


def reach(lang: str) -> dict:
    corp, backend, qs = world(lang)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    open_state, hidden_state, reasons, first_hit = Counter(), Counter(), Counter(), 0
    reachable_ids, hidden_ids = [], []
    for q in qs:
        support = set(corp.support_chunks(q).values())
        first_hit += bool(support & {h.chunk_id for h in search.search(q.question, 3)})
        built = {b.variant: b for b in builder.build_question(q, "xq", corp, tool, search)}
        o = built["open"]
        open_state[o.state] += 1
        if o.episode is not None:
            reachable_ids.append(q.qid)
            h = built.get("support_masked")
            if h is not None:
                hidden_state[h.state] += 1
                if h.episode is not None:
                    hidden_ids.append(q.qid)
                else:
                    reasons[h.reject_reason] += 1
        else:
            reasons[o.reject_reason] += 1
    n = len(qs)
    return {"questions": n, "first_search_top3_has_answer_passage": round(first_hit / n, 3),
            "oracle_reaches_within_3_searches": round(len(reachable_ids) / n, 3), "reachable": len(reachable_ids),
            "open_states": dict(open_state), "hidden_evidence_states": dict(hidden_state),
            "hidden_episodes_kept": len(hidden_ids), "reasons": dict(reasons), "reachable_ids": reachable_ids, "hidden_ids": hidden_ids}


def prompt_lengths(lang: str, tok_dir: Path) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    corp, backend, qs = world(lang)
    system = protocol.system_prompt(CorpusSearchTool(backend, top_k=3))
    count = lambda messages: len(tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)["input_ids"])
    from ragharness.retrieval import Hit, visible_text

    chunks = sorted(corp.chunks, key=lambda c: -len(tok.encode(visible_text(c["text"], 800), add_special_tokens=False)))
    longest_q = max(qs, key=lambda q: len(tok.encode(q.question, add_special_tokens=False)))
    nine = [Hit(c["chunk_id"], c["title"], c["text"], 0.0) for c in chunks[:9]]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": longest_q.question}]
    search_turn = protocol.format_search(longest_q.question)
    for i in range(3):
        messages += [{"role": "assistant", "content": search_turn}, {"role": "user", "content": protocol.format_tool_result(render_observation(nine[3 * i:3 * i + 3], frozenset(), 800))}]
    worst = count(messages)
    per_passage = sorted(len(tok.encode(visible_text(c["text"], 800), add_special_tokens=False)) for c in corp.chunks)
    return {"worst_case_tokens": worst, "limit": INPUT_LIMIT, "fits": worst <= INPUT_LIMIT, "passage_tokens_median": per_passage[len(per_passage) // 2],
            "passage_tokens_max": per_passage[-1], "system_prompt_tokens": count([{"role": "system", "content": system}, {"role": "user", "content": "x"}])}


def hindi_probe() -> dict:
    corp, backend, qs = world("hi")
    q = qs[0]
    hits = MaskedSearch(backend).search(q.question, 3)
    support = set(corp.support_chunks(q).values())
    return {"query": q.question, "titles": [h.title for h in hits], "hits": len(hits), "devanagari_titles": all(any("\u0900" <= ch <= "\u097f" for ch in h.title) for h in hits),
            "answer_passage_in_top3": bool(support & {h.chunk_id for h in hits})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,en,es")
    ap.add_argument("--tokenizer", default=str(ROOT / ".cache" / "tiny-aya-global"))
    ap.add_argument("--out", default="reports/xquad_checks.json")
    args = ap.parse_args()
    out = {"reach": {}, "prompt": {}}
    for lang in args.langs.split(","):
        out["reach"][lang] = reach(lang)
        print(lang, json.dumps({k: v for k, v in out["reach"][lang].items() if not k.endswith("_ids")}), flush=True)
        out["prompt"][lang] = prompt_lengths(lang, Path(args.tokenizer))
        print(lang, json.dumps(out["prompt"][lang]), flush=True)
    if "hi" in args.langs.split(","):
        out["hindi_index_probe"] = hindi_probe()
        print(json.dumps(out["hindi_index_probe"], ensure_ascii=False), flush=True)
    (ROOT / args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
