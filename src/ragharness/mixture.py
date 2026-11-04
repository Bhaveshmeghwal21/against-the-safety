"""Training mixtures (plan sections 5.1 and 6.1).

main     all states, balanced by action: abstain rows are the rarest class, so search and answer rows are capped at
         `max_ratio` times the abstain count. Selection is round-robin over parent questions (then states) so no
         question dominates.
control  supported-only tool-episode SFT from the same parent questions: only answerable episodes, so no abstain
         rows and no missing-evidence episodes. Rows are drawn to match the main mixture's processed-token budget
         within `tolerance`, keeping the main mixture's search:answer count ratio.

All sizes use estimated tokens (characters / 3.5) until the real tokenizer is available; the selection logic only
needs relative sizes, and the real counts are reported once known.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from . import rows as rowmod


def annotate(rows: list[dict], tokenizer=None) -> list[dict]:
    """Attach token counts to each row: exact when a tokenizer is given, otherwise character estimates."""
    for r in rows:
        if tokenizer is not None:
            r["meta"]["est_tokens"], r["meta"]["est_target_tokens"] = rowmod.real_tokens(r, tokenizer)
            r["meta"]["tokens_exact"] = True
        else:
            r["meta"]["est_tokens"], r["meta"]["est_target_tokens"] = rowmod.est_tokens(r)
    return rows


_CHUNK_ID = re.compile(r"c\d{5}")


def drop_overlap(rows: list[dict], protected: set[str]) -> tuple[list[dict], int]:
    """Remove every row whose prompt or completion mentions a protected chunk ID (dev/test support passages)."""
    kept, dropped = [], 0
    for r in rows:
        blob = " ".join(m["content"] for m in r["prompt"][1:]) + " " + r["completion"][0]["content"]
        if protected.intersection(_CHUNK_ID.findall(blob)):
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


def _round_robin(pool: list[dict], rng: random.Random, limit: int) -> list[dict]:
    """Take up to `limit` rows, cycling over parent questions so each contributes before any contributes twice."""
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_parent[r["meta"]["parent_qid"]].append(r)
    parents = sorted(by_parent)
    rng.shuffle(parents)
    for p in parents:
        rng.shuffle(by_parent[p])
    chosen: list[dict] = []
    while len(chosen) < limit and any(by_parent.values()):
        for p in parents:
            if by_parent[p] and len(chosen) < limit:
                chosen.append(by_parent[p].pop())
    return chosen


def build_main(rows: list[dict], seed: int, max_cap_tokens: int, max_ratio: float = 2.0) -> list[dict]:
    rng = random.Random(seed)
    usable = [r for r in rows if r["meta"]["est_tokens"] <= max_cap_tokens]
    by_kind = defaultdict(list)
    for r in usable:
        by_kind[r["meta"]["kind"]].append(r)
    cap = int(max_ratio * len(by_kind["abstain"]))
    chosen: list[dict] = list(by_kind["abstain"])
    for kind in ("search", "answer"):
        chosen += _round_robin(by_kind[kind], rng, cap)
    rng.shuffle(chosen)
    return chosen


def build_control(rows: list[dict], main: list[dict], seed: int, max_cap_tokens: int, tolerance: float = 0.05) -> tuple[list[dict], dict]:
    rng = random.Random(seed + 1)
    pool = [r for r in rows if r["meta"]["answerable"] and r["meta"]["est_tokens"] <= max_cap_tokens]
    target = sum(r["meta"]["est_tokens"] for r in main)
    n_search = sum(r["meta"]["kind"] == "search" for r in main)
    n_final = len(main) - n_search  # answers and abstentions both become answers in the control
    share_search = n_search / len(main)
    pools = {k: _round_robin([r for r in pool if r["meta"]["kind"] == k], rng, 10**9) for k in ("search", "answer")}
    taken = {"search": 0, "answer": 0}
    chosen: list[dict] = []
    total = 0
    resampled = 0
    while total < target * (1 - tolerance / 2):
        kind = "search" if len(chosen) == 0 or taken["search"] / len(chosen) < share_search else "answer"
        if taken[kind] >= len(pools[kind]):
            resampled += 1
            r = rng.choice(pools[kind])
        else:
            r = pools[kind][taken[kind]]
        taken[kind] += 1
        chosen.append(r)
        total += r["meta"]["est_tokens"]
    rng.shuffle(chosen)
    info = {"target_tokens": target, "control_tokens": total, "relative_difference": round(total / target - 1, 4), "resampled_rows": resampled,
            "pool_rows": {k: len(v) for k, v in pools.items()}}
    return chosen, info


def summarize(rows: list[dict]) -> dict:
    kinds: dict[str, int] = defaultdict(int)
    states: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["meta"]["kind"]] += 1
        states[r["meta"]["state"]] += 1
    return {
        "rows": len(rows),
        "parents": len({r["meta"]["parent_qid"] for r in rows}),
        "by_kind": dict(sorted(kinds.items())),
        "by_state": dict(sorted(states.items())),
        "processed_tokens_est": sum(r["meta"]["est_tokens"] for r in rows),
        "supervised_tokens_est": sum(r["meta"]["est_target_tokens"] for r in rows),
    }
