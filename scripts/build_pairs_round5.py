"""Build preference pairs from sampled rollouts (plan section 27, priority 2). Rules: research/2026-11-07-round5-preregistration.md.

  python scripts/build_pairs_round5.py --samples runs/account2_round5/ats-sample-r5-pairs/out/samples --out data/processed/training/dpo_r5.jsonl

Reads only sampled TRAINING rollouts and the local corpora. For each rollout the context is the message list before its final step,
and the rejected text is that final step. The reference action depends only on what the context shows (see the note).
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ragharness import builder, corpus as corpus_mod, data, episodes, policy, protocol, rescore, rows
from ragharness.retrieval import visible_text

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 3
MAX_TOKENS = 2560


def load_world(set_name: str):
    if set_name == "r5train":
        bench = ROOT / "data" / "bench" / "r5train"
        return corpus_mod.load_corpus(bench / "corpus.json"), {q.qid: q for q in data.read_questions(bench / "questions.jsonl")}
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    return corp, {q.qid: q for q in splits["train"]}


def norm_query(text: str) -> str:
    return " ".join(text.lower().split())


def context_messages(r: dict, system: str) -> list[dict]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["user_text"]}]
    for s in r["steps"][:-1]:
        msgs += [{"role": "assistant", "content": s["text"]}, {"role": "user", "content": protocol.format_tool_result(s["observation"])}]
    return msgs


def reference(r: dict, q, corp, shown: set[str]):
    """(kind, text, reason) of the reference action at the final context, or (None, None, why) when the rollout is not a usable pair."""
    final = r["final"]
    support = corp.support_chunks(q)
    facets = [set(corp.title_copies(t, c)) | {c} for t, c in support.items()]
    complete = all(f & shown for f in facets)
    depth = len(r["steps"]) - 1
    max_chars = r["max_chars"]
    if not complete:
        # an alternate passage set that holds the whole answer makes "unsupported" ambiguous: skip (training-side false-unsupported fix)
        if rescore.fully_visible(q, corp, shown, max_chars):
            return None, None, "alternate_evidence_visible"
    if r["answerable"] and complete:
        good = final["kind"] == "answer" and r["outcome"] == "supported_correct" and r["all_aspects_cited"]
        if good:
            return None, None, "already_good"
        cites = [sorted(f & shown, key=lambda c: (c not in support.values(), c))[0] for f in facets]
        cites = list(dict.fromkeys(cites))
        return "answer", protocol.format_answer(q.answer, cites), "answerable_complete"
    # not complete (answerable or hidden)
    if final["kind"] == "abstain" and (depth >= BUDGET or not r["answerable"]):
        return None, None, "abstain_is_fine"
    if final["kind"] == "abstain" and r["answerable"] and depth < BUDGET:
        pass  # early abstention on an answerable episode: reference is another search
    elif final["kind"] not in ("answer", "invalid"):
        return None, None, "not_a_final_error"
    if depth < BUDGET:
        observed = {c: builder._hit(corp, c) for c in shown}
        used = {norm_query(s["query"]) for s in r["steps"][:-1] if s["query"]}
        query, _ = policy.next_query(q, support, observed, used, depth, prefilled=bool(r["prefill"]))
        if query is not None:
            return "search", protocol.format_search(query), "incomplete_search"
    return "abstain", protocol.format_abstain(episodes.ABSTAIN_TEXT.format(question=q.question)), "incomplete_abstain"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="folder holding <system>/samples_<set>.jsonl")
    ap.add_argument("--out", default="data/processed/training/dpo_r5.jsonl")
    ap.add_argument("--report", default="reports/round5_pairs.json")
    ap.add_argument("--tokenizer", default="runs/kaggle/ats-train-both/out/main/adapter")
    ap.add_argument("--max-pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer), local_files_only=True)
    system = json.loads((ROOT / "data" / "processed" / "training" / "main.jsonl").read_text(encoding="utf-8").splitlines()[0])["prompt"][0]["content"]
    files = sorted(Path(args.samples).rglob("samples_*.jsonl"))
    worlds, candidates, reasons, seen = {}, [], Counter(), set()
    n_rollouts = Counter()
    for path in files:
        set_name = path.stem.removeprefix("samples_")
        corp, qmap = worlds.setdefault(set_name, load_world(set_name))
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            n_rollouts[(r["system"], set_name)] += 1
            q = qmap[r["qid"]]
            shown = set(r["prefill"]) | {c for s in r["steps"][:-1] for c in s["hits"]}
            kind, chosen, why = reference(r, q, corp, shown)
            reasons[why] += 1
            if kind is None:
                continue
            rejected = r["steps"][-1]["text"]
            if not rejected.strip() or rejected.strip() == chosen.strip():
                reasons["same_or_empty"] += 1
                continue
            prompt = context_messages(r, system)
            key = hashlib.sha1(json.dumps([prompt, chosen, rejected], ensure_ascii=False).encode()).hexdigest()
            if key in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(key)
            row = {"prompt": prompt, "chosen": [{"role": "assistant", "content": chosen}], "rejected": [{"role": "assistant", "content": rejected}]}
            try:
                rows.check_row({"prompt": prompt, "completion": row["chosen"]})
                n_c = rows.real_tokens({"prompt": prompt, "completion": row["chosen"]}, tok)[0]
                n_r = rows.real_tokens({"prompt": prompt, "completion": row["rejected"]}, tok, strict=False)[0]
            except AssertionError as e:
                reasons[f"template_check:{str(e)[:40]}"] += 1
                continue
            if max(n_c, n_r) > MAX_TOKENS:
                reasons["overlength"] += 1
                continue
            row["meta"] = {"set": set_name, "system": r["system"], "run": r["run"], "qid": r["qid"], "variant": r["variant"], "answerable": r["answerable"],
                           "depth": len(r["steps"]) - 1, "reason": why, "chosen_kind": kind, "rejected_kind": r["final"]["kind"],
                           "rejected_outcome": r["outcome"], "tokens": max(n_c, n_r)}
            candidates.append(row)

    # caps per (question, variant) and per question, in a seeded order that ignores length and outcome. The pre-registered caps are (2, 4);
    # if they leave fewer than 450 balanced pairs, the next rule (3, 6) is used instead (fixed before any sample was read).
    rng.shuffle(candidates)
    for cap_qv, cap_q in ((2, 4), (3, 6)):
        per_qv, per_q, capped = Counter(), Counter(), []
        for row in candidates:
            m = row["meta"]
            if per_qv[(m["qid"], m["variant"])] >= cap_qv or per_q[m["qid"]] >= cap_q:
                continue
            per_qv[(m["qid"], m["variant"])] += 1
            per_q[m["qid"]] += 1
            capped.append(row)
        pro_answer = [r for r in capped if r["meta"]["chosen_kind"] == "answer"]
        pro_caution = [r for r in capped if r["meta"]["chosen_kind"] != "answer"]
        small, large = sorted([pro_answer, pro_caution], key=len)
        if len(small) + min(len(large), int(1.25 * len(small))) >= 450:
            break
    reasons["caps_used"] = f"{cap_qv},{cap_q}"
    n_small, n_large = len(small), min(len(large), int(1.25 * len(small)))
    if n_small + n_large > args.max_pairs:
        scale = args.max_pairs / (n_small + n_large)
        n_small, n_large = int(n_small * scale), int(n_large * scale)
    selected = small[:n_small] + large[:n_large]
    rng.shuffle(selected)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    report = {
        "rollouts": {f"{s}/{n}": c for (s, n), c in sorted(n_rollouts.items())}, "decisions": dict(reasons),
        "candidates_after_template_and_length": len(candidates), "after_caps": len(capped), "pro_answer": len(pro_answer), "pro_caution": len(pro_caution),
        "selected": len(selected), "selected_by_chosen_kind": dict(Counter(r["meta"]["chosen_kind"] for r in selected)),
        "selected_by_rejected_kind": dict(Counter(r["meta"]["rejected_kind"] for r in selected)),
        "selected_by_set": dict(Counter(r["meta"]["set"] for r in selected)), "selected_by_source_system": dict(Counter(r["meta"]["system"] for r in selected)),
        "selected_by_answerable": dict(Counter(str(r["meta"]["answerable"]) for r in selected)), "selected_by_depth": dict(sorted(Counter(r["meta"]["depth"] for r in selected).items())),
        "questions": len({r["meta"]["qid"] for r in selected}), "mean_tokens": round(sum(r["meta"]["tokens"] for r in selected) / max(1, len(selected))),
        "max_tokens": max((r["meta"]["tokens"] for r in selected), default=0), "sha256": digest, "path": args.out,
    }
    (ROOT / args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
