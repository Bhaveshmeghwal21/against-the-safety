"""Round 5b: abstention-chosen preference pairs (research/2026-11-07-round5b-preregistration.md).

  python scripts/build_pairs_round5b.py contexts --samples <samples folder> --out data/processed/training/force_contexts_r5.jsonl
  python scripts/build_pairs_round5b.py pairs --forced <folder holding forced_<system>.jsonl> \
      --contexts data/processed/training/force_contexts_r5.jsonl --old data/processed/training/dpo_r5.jsonl --out data/processed/training/dpo_r5b.jsonl

Step 1 (CPU) picks decision points from the sampled TRAINING rollouts of round 5: hidden-evidence episodes in which the model itself abstained after at
least one search and no shown passage holds the whole answer. The context is the message list before that abstention.
Step 2 (GPU, scripts/force_answers_round5.py) makes the same system answer at that context, with `<answer>` forced as the first tokens.
Step 3 (CPU) pairs the canonical abstention (chosen) with that forced answer (rejected) and appends the pairs to the unchanged round-5 pair file.
No development or test rollout is sampled; held-out annotations are read to reject leaking training prompts.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pairs_round5 as B  # noqa: E402
from ragharness import episodes, protocol, rows, lineage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_NEW = 160


def tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(ROOT / path), local_files_only=True)


def contexts_step(args) -> None:
    rng = random.Random(args.seed)
    tok = tokenizer(args.tokenizer)
    system = json.loads((ROOT / "data" / "processed" / "training" / "main.jsonl").read_text(encoding="utf-8").splitlines()[0])["prompt"][0]["content"]
    worlds, found, why, seen = {}, [], Counter(), set()
    guard = lineage.PromptGuard(lineage.load_holdouts(ROOT))
    for path in sorted(Path(args.samples).rglob("samples_*.jsonl")):
        set_name = path.stem.removeprefix("samples_")
        corp, qmap = worlds.setdefault(set_name, B.load_world(set_name))
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["answerable"]:
                continue
            q = qmap[r["qid"]]
            shown = set(r["prefill"]) | {c for s in r["steps"][:-1] for c in s["hits"]}
            if B.rescore.fully_visible(q, corp, shown, r["max_chars"]):
                why["alternate_evidence_visible"] += 1
                continue
            depth = len(r["steps"]) - 1
            if r["final"]["kind"] != "abstain" or depth < 1:
                why["not_an_abstention_after_a_search"] += 1
                continue
            prompt = B.context_messages(r, system)
            if guard.matches(prompt):
                why["heldout_evidence_or_question"] += 1
                continue
            key = hashlib.sha1(json.dumps(prompt, ensure_ascii=False).encode()).hexdigest()
            if key in seen:
                why["duplicate_context"] += 1
                continue
            seen.add(key)
            chosen = protocol.format_abstain(episodes.ABSTAIN_TEXT.format(question=q.question))
            row = {"prompt": prompt, "completion": [{"role": "assistant", "content": chosen}]}
            try:
                rows.check_row(row)
                n = rows.real_tokens(row, tok)[0]
            except AssertionError as e:
                why[f"template_check:{str(e)[:40]}"] += 1
                continue
            if n + MAX_NEW > B.MAX_TOKENS:
                why["overlength"] += 1
                continue
            found.append({"id": key[:16], "system": r["system"], "set": set_name, "qid": r["qid"], "variant": r["variant"], "depth": depth,
                          "prompt": prompt, "chosen": chosen, "tokens": n})
    rng.shuffle(found)
    per_qv, per_q, per_set, picked = Counter(), Counter(), Counter(), []
    for c in found:
        if per_qv[(c["qid"], c["variant"])] >= 2 or per_q[c["qid"]] >= 4 or per_set[c["set"]] >= args.per_set:
            why["cap"] += 1
            continue
        per_qv[(c["qid"], c["variant"])] += 1
        per_q[c["qid"]] += 1
        per_set[c["set"]] += 1
        picked.append(c)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for c in picked:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    report = {"found": len(found), "picked": len(picked), "decisions": dict(why), "by_set": dict(per_set),
              "by_system": dict(Counter(c["system"] for c in picked)), "by_depth": dict(sorted(Counter(c["depth"] for c in picked).items())),
              "by_variant": dict(Counter(c["variant"] for c in picked)), "questions": len({c["qid"] for c in picked}),
              "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "path": args.out}
    (ROOT / args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def pairs_step(args) -> None:
    rng = random.Random(args.seed)
    tok = tokenizer(args.tokenizer)
    ctx = [json.loads(l) for l in (ROOT / args.contexts).read_text(encoding="utf-8").splitlines() if l.strip()]
    forced = {}
    for path in sorted(Path(args.forced).rglob("forced_*.jsonl")):
        for l in path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                forced[r["id"]] = r
    why, per_set, new = Counter(), Counter(), []
    for c in ctx:
        f = forced.get(c["id"])
        if f is None:
            why["no_forced_answer"] += 1
            continue
        turn = protocol.parse_turn(f["text"])
        if turn.kind != "answer" or turn.error or not turn.citations:
            why["forced_text_is_not_a_valid_answer"] += 1
            continue
        rejected = protocol.format_answer(turn.answer, list(turn.citations))
        if rejected.strip() == c["chosen"].strip():
            why["same"] += 1
            continue
        row = {"prompt": c["prompt"], "chosen": [{"role": "assistant", "content": c["chosen"]}], "rejected": [{"role": "assistant", "content": rejected}]}
        try:
            n_r = rows.real_tokens({"prompt": c["prompt"], "completion": row["rejected"]}, tok, strict=False)[0]
        except AssertionError as e:
            why[f"template_check:{str(e)[:40]}"] += 1
            continue
        if max(c["tokens"], n_r) > B.MAX_TOKENS:
            why["overlength"] += 1
            continue
        if per_set[c["set"]] >= args.per_set:
            why["set_cap"] += 1
            continue
        per_set[c["set"]] += 1
        row["meta"] = {"set": c["set"], "system": c["system"], "run": None, "qid": c["qid"], "variant": c["variant"], "answerable": False, "depth": c["depth"],
                       "reason": "hidden_abstain_forced_answer", "chosen_kind": "abstain", "rejected_kind": "answer", "rejected_outcome": "forced_unsupported",
                       "tokens": max(c["tokens"], n_r)}
        new.append(row)
    old = [json.loads(l) for l in (ROOT / args.old).read_text(encoding="utf-8").splitlines() if l.strip()]
    selected = old + new
    selected, validation = lineage.clean_rows(selected, lineage.PromptGuard(lineage.load_holdouts(ROOT)))
    rng.shuffle(selected)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"old_pairs": len(old), "old_sha256": hashlib.sha256((ROOT / args.old).read_bytes()).hexdigest(), "new_pairs": len(new), "total": len(selected),
              "training_validation": validation,
              "decisions": dict(why), "new_by_set": dict(Counter(r["meta"]["set"] for r in new)), "new_by_system": dict(Counter(r["meta"]["system"] for r in new)),
              "new_by_depth": dict(sorted(Counter(r["meta"]["depth"] for r in new).items())),
              "total_by_chosen_kind": dict(Counter(r["meta"]["chosen_kind"] for r in selected)),
              "total_by_rejected_kind": dict(Counter(r["meta"]["rejected_kind"] for r in selected)),
              "total_by_set": dict(Counter(r["meta"]["set"] for r in selected)), "questions": len({r["meta"]["qid"] for r in selected}),
              "mean_tokens": round(sum(r["meta"]["tokens"] for r in selected) / max(1, len(selected))), "max_tokens": max(r["meta"]["tokens"] for r in selected),
              "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "path": args.out}
    (ROOT / args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    c = sub.add_parser("contexts")
    c.add_argument("--samples", required=True)
    c.add_argument("--out", default="data/processed/training/force_contexts_r5.jsonl")
    c.add_argument("--report", default="reports/round5b_contexts.json")
    c.add_argument("--per-set", type=int, default=120)
    p = sub.add_parser("pairs")
    p.add_argument("--forced", required=True)
    p.add_argument("--contexts", default="data/processed/training/force_contexts_r5.jsonl")
    p.add_argument("--old", default="data/processed/training/dpo_r5.jsonl")
    p.add_argument("--out", default="data/processed/training/dpo_r5b.jsonl")
    p.add_argument("--report", default="reports/round5b_pairs.json")
    p.add_argument("--per-set", type=int, default=90)
    for s in (c, p):
        s.add_argument("--tokenizer", default="runs/kaggle/ats-train-both/out/main/adapter")
        s.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()
    contexts_step(args) if args.step == "contexts" else pairs_step(args)


if __name__ == "__main__":
    main()
