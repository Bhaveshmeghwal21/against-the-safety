"""Print accepted training episodes for the manual verification gate (plan section 5.3).

Run (after build_episodes.py):  python scripts/inspect_episodes.py [--per-state 5] [--seed 7] [--split train] [--max-chars 420]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-state", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-chars", type=int, default=420)
    ap.add_argument("--variants", default="")
    args = ap.parse_args()
    eps = [json.loads(l) for l in (ROOT / "data" / "processed" / "episodes" / f"episodes_{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    by = defaultdict(list)
    for e in eps:
        by[(e["meta"]["state"], e["meta"]["variant"])].append(e)
    rng = random.Random(args.seed)
    want = set(args.variants.split(",")) if args.variants else None
    for key in sorted(by):
        if want and key[1] not in want:
            continue
        for e in rng.sample(by[key], min(args.per_state, len(by[key]))):
            m = e["meta"]
            print(f"\n=== {m['state']} / {m['variant']}  parent={m['parent_qid']}  type={m['qtype']}  searches={m['n_searches']}  final={m['final']}  gold={e['gold_answer']!r}")
            print("USER:", clip(e["user"], args.max_chars))
            for s in e["steps"]:
                print("  ASSISTANT:", clip(s["assistant"], args.max_chars))
                if s["observation"]:
                    print("  OBS:", clip(s["observation"], args.max_chars))


if __name__ == "__main__":
    main()
