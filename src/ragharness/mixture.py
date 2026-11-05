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


# --- Second round (plan section 20): two candidates that lower the abstain share and remove the search-depth shortcut. ---
#
# In the main mixture the label of a final turn is predictable from how many searches came before it: 81% of final rows after
# three searches are abstentions (429 of 531) and none after zero or one. The trained adapter copies this on dev (it abstains
# on 54 of 63 final turns after three searches, including 5 where all the support was on screen). Both candidates keep the
# main recipe and token budget; they change only which existing train decisions are selected.

SEARCH_SHARE = 1 / 3  # as in the main mixture (537 of 1,611 rows)
ABSTAIN_SHARE_OF_FINALS = 0.30  # main: 0.50


def _fill_to_budget(groups: dict[str, list[dict]], frac: dict[str, float], target_tokens: int) -> tuple[list[dict], dict]:
    """Take rows from ordered groups so every prefix keeps the target fractions, until the processed-token budget is reached."""
    assert abs(sum(frac.values()) - 1) < 1e-9, frac
    taken = {g: 0 for g in groups}
    chosen: list[dict] = []
    total = 0
    while total < target_tokens:
        open_groups = [g for g in groups if taken[g] < len(groups[g])]
        if not open_groups:
            raise RuntimeError("pool exhausted before the token budget was reached")
        g = max(open_groups, key=lambda g: frac[g] * (len(chosen) + 1) - taken[g])
        r = groups[g][taken[g]]
        taken[g] += 1
        chosen.append(r)
        total += r["meta"]["est_tokens"]
    short = {g: round(frac[g] * len(chosen) - taken[g], 1) for g in groups if frac[g] * len(chosen) - taken[g] > 1.5}
    return chosen, {"taken": taken, "tokens": total, "groups_short_of_target": short}


def _depth(r: dict) -> int:
    return r["meta"]["n_searches"]


def build_cand_a(rows: list[dict], seed: int, max_cap_tokens: int, target_tokens: int) -> tuple[list[dict], dict]:
    """Candidate A: the main mixture's sampling, with abstentions at 30% of final decisions instead of 50%."""
    rng = random.Random(seed + 2)
    usable = [r for r in rows if r["meta"]["est_tokens"] <= max_cap_tokens]
    kind = lambda k: [r for r in usable if r["meta"]["kind"] == k]
    groups = {k: _round_robin(kind(k), rng, 10**9) for k in ("search", "answer", "abstain")}
    finals = 1 - SEARCH_SHARE
    frac = {"search": SEARCH_SHARE, "answer": finals * (1 - ABSTAIN_SHARE_OF_FINALS), "abstain": finals * ABSTAIN_SHARE_OF_FINALS}
    chosen, info = _fill_to_budget(groups, frac, target_tokens)
    rng.shuffle(chosen)
    return chosen, info


def build_cand_b(rows: list[dict], seed: int, max_cap_tokens: int, target_tokens: int,
                 abstain_rate_at_depth: float = 0.40, answer_d0_share: float = 0.09) -> tuple[list[dict], dict]:
    """Candidate B: candidate A's shares, plus
    - depth matching: among final rows after d searches (d = 2, 3), abstentions are `abstain_rate_at_depth` (main: 0.58, 0.81);
    - matched pairs: answer rows after 2 or 3 searches come first from questions whose abstention is also in the set, so the
      same question appears once with its evidence reachable and once with it missing;
    - search recovery: search rows split evenly over the first, second and third search of an episode (main: about 48/31/21%);
    - answer rows with no search (evidence supplied) kept at `answer_d0_share` of rows (main: 0.086), so "when to retrieve" is not traded away.
    """
    rng = random.Random(seed + 3)
    usable = [r for r in rows if r["meta"]["est_tokens"] <= max_cap_tokens]
    of = lambda k, pred: [r for r in usable if r["meta"]["kind"] == k and pred(r)]
    abstain = {d: _round_robin(of("abstain", lambda r, d=d: _depth(r) == d), rng, 10**9) for d in (2, 3)}
    pool_abstain = {d: len(v) for d, v in abstain.items()}
    # parent order of the abstentions, interleaved by depth in proportion, decides which answers are paired first
    abstain_parents: list[str] = []
    for d in (2, 3):
        for r in abstain[d]:
            if r["meta"]["parent_qid"] not in abstain_parents:
                abstain_parents.append(r["meta"]["parent_qid"])
    rank = {p: i for i, p in enumerate(abstain_parents)}

    def paired_first(cands: list[dict]) -> list[dict]:
        paired = sorted((r for r in cands if r["meta"]["parent_qid"] in rank), key=lambda r: (rank[r["meta"]["parent_qid"]], r["meta"]["variant"]))
        rest = _round_robin([r for r in cands if r["meta"]["parent_qid"] not in rank], rng, 10**9)
        return paired + rest

    groups = {
        "search_1st": _round_robin(of("search", lambda r: r["meta"]["step"] == 0), rng, 10**9),
        "search_2nd": _round_robin(of("search", lambda r: r["meta"]["step"] == 1), rng, 10**9),
        "search_3rd": _round_robin(of("search", lambda r: r["meta"]["step"] == 2), rng, 10**9),
        "abstain_d2": abstain[2],
        "abstain_d3": abstain[3],
        "answer_d2": paired_first(of("answer", lambda r: _depth(r) == 2)),
        "answer_d3": paired_first(of("answer", lambda r: _depth(r) == 3)),
        "answer_d0": _round_robin(of("answer", lambda r: _depth(r) == 0), rng, 10**9),
        "answer_d1": _round_robin(of("answer", lambda r: _depth(r) == 1), rng, 10**9),
    }
    finals = 1 - SEARCH_SHARE
    abst = finals * ABSTAIN_SHARE_OF_FINALS
    share_d2 = pool_abstain[2] / (pool_abstain[2] + pool_abstain[3])
    odds = (1 - abstain_rate_at_depth) / abstain_rate_at_depth
    frac = {
        "search_1st": SEARCH_SHARE / 3, "search_2nd": SEARCH_SHARE / 3, "search_3rd": SEARCH_SHARE / 3,
        "abstain_d2": abst * share_d2, "abstain_d3": abst * (1 - share_d2),
        "answer_d2": abst * share_d2 * odds, "answer_d3": abst * (1 - share_d2) * odds,
        "answer_d0": answer_d0_share,
    }
    frac["answer_d1"] = 1 - sum(frac.values())
    assert frac["answer_d1"] > 0, frac
    chosen, info = _fill_to_budget(groups, frac, target_tokens)
    chosen_parents = {r["meta"]["parent_qid"] for r in chosen if r["meta"]["kind"] == "abstain"}
    info["answer_rows_after_2_or_3_searches_paired_with_an_abstention"] = sum(
        r["meta"]["kind"] == "answer" and _depth(r) >= 2 and r["meta"]["parent_qid"] in chosen_parents for r in chosen)
    rng.shuffle(chosen)
    return chosen, info


def depth_table(rows: list[dict]) -> dict:
    """Final decisions by the number of searches before them: {depth: {answer, abstain, abstain_rate}}."""
    out: dict = {}
    for r in rows:
        if r["meta"]["kind"] in ("answer", "abstain"):
            d = out.setdefault(str(_depth(r)), {"answer": 0, "abstain": 0})
            d[r["meta"]["kind"]] += 1
    for d in out.values():
        d["abstain_rate"] = round(d["abstain"] / (d["answer"] + d["abstain"]), 3)
    return dict(sorted(out.items()))
