"""Label-leak and shortcut checks over the accepted training episodes (plan sections 5.2 and 5.3).

Writes reports/shortcut_report_train.json. Run after build_episodes.py:  python scripts/shortcut_report.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK = re.compile(r"\[chunk_id=(c\d{5})\]")
Q = re.compile(r'"query": "((?:[^"\\]|\\.)*)"')


def load(split: str) -> list[dict]:
    p = ROOT / "data" / "processed" / "episodes" / f"episodes_{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def query_of(step: dict) -> str | None:
    m = Q.search(step["assistant"]) if step["assistant"].startswith("[{") else None
    return json.loads('"' + m.group(1) + '"') if m else None


def main() -> None:
    eps = load("train")
    rep: dict = {"n_episodes": len(eps)}

    # 1. Result counts per observation, by final decision
    counts: dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        for s in e["steps"]:
            if s["observation"] is not None:
                n = len(CHUNK.findall(s["observation"]))
                counts[e["meta"]["final"]][n] += 1
    rep["observation_passage_count_by_final"] = {k: dict(sorted(v.items())) for k, v in counts.items()}

    # 2. Queries by kind versus final decision. kind: question / mention (starts uppercase, short) / keywords (lowercase words)
    kinds: dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        question = e["user"].split("\n\nEvidence already provided:")[0]
        for s in e["steps"]:
            qu = query_of(s)
            if qu is None:
                continue
            kind = "question" if qu == question else ("keywords" if qu[:1].islower() else "mention")
            kinds[e["meta"]["final"]][kind] += 1
    rep["query_kind_by_final"] = {k: dict(v) for k, v in kinds.items()}

    # 3. Searches before the final decision, by state
    ns: dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        ns[f'{e["meta"]["state"]}/{e["meta"]["variant"]}'][e["meta"]["n_searches"]] += 1
    rep["searches_by_state_variant"] = {k: dict(sorted(v.items())) for k, v in sorted(ns.items())}

    # 4. Gold answer text present in the cited support passages (answer rows only; yes/no skipped)
    present = total = skipped = 0
    for e in eps:
        fin = e["steps"][-1]["assistant"]
        if not fin.startswith("<answer>"):
            continue
        ans = e["gold_answer"].strip()
        if ans.lower() in {"yes", "no"} or len(ans) < 3:
            skipped += 1
            continue
        cited = re.search(r"<citations>(.*?)</citations>", fin).group(1).split(",")
        text = " ".join(
            p for st in e["steps"] for p in [st["observation"] or ""]
        ) + " " + e["user"]
        spans = re.split(r"(?=\[chunk_id=c\d{5}\])", text)
        cited_text = " ".join(sp for sp in spans if any(f"[chunk_id={c}]" in sp for c in cited))
        total += 1
        present += bool(re.search(rf"(?<!\w){re.escape(ans)}(?!\w)", cited_text, flags=re.IGNORECASE))
    rep["answer_text_in_cited_passages"] = {"present": present, "checked": total, "skipped_yes_no_or_short": skipped}

    # 5. Mask and prefill presence by final decision
    flags: dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        flags[e["meta"]["final"]]["support_masked"] += bool(e["support_masked"])
        flags[e["meta"]["final"]]["first_search_masked"] += bool(e["first_search_masked"])
        flags[e["meta"]["final"]]["prefill"] += "Evidence already provided:" in e["user"]
        flags[e["meta"]["final"]]["total"] += 1
    rep["mask_prefill_by_final"] = {k: dict(v) for k, v in flags.items()}

    # 6. Duplicate final/abstain text (should be templated)
    rep["distinct_abstain_texts"] = len({e["steps"][-1]["assistant"] for e in eps if e["meta"]["final"] == "abstain"})

    out = ROOT / "reports" / "shortcut_report_train.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
