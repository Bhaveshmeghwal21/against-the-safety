"""A blank sheet for a human to check 50 answers of the main adapter (research/2026-11-16-judge-latency-preregistration.md, section 4).

  python scripts/build_hand_label_sheet.py

Draws 25 answered episodes on hidden evidence and 25 answered episodes whose documents hold the answer, from the 300-episode judge sample, seed 61, in a
shuffled order. The sheet shows the question, the system's answer and the passages it was shown. It does not show our label, the judge's verdict or whether
the evidence was hidden; those stay in `data/judge/hand_label_key.json` until the labels are in. The labeller answers two questions per row:
"Do the shown passages support the answer?" and "Is the answer right?".
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIDDEN = ("unsupported_wrong", "unsupported_correct")


def main() -> None:
    cases = [json.loads(l) for l in (ROOT / "data/judge/cases_r5t.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = [c for c in cases if c["system"] == "main" and c["final_kind"] == "answer" and (c["answer"] or "").strip()]
    hidden = sorted((c for c in pool if c["outcome_v3"] in HIDDEN), key=lambda c: (c["part"], c["qid"], c["variant"]))
    answerable = sorted((c for c in pool if c["answerable"]), key=lambda c: (c["part"], c["qid"], c["variant"]))
    rng = random.Random(61)
    chosen = rng.sample(hidden, min(25, len(hidden))) + rng.sample(answerable, min(25, len(answerable)))
    rng.shuffle(chosen)
    key, rows = {}, []
    for i, c in enumerate(chosen, 1):
        rid = f"H{i:02d}"
        key[rid] = {"part": c["part"], "qid": c["qid"], "variant": c["variant"], "annotation_outcome": c["outcome_v3"], "answerable": c["answerable"]}
        passages = "\n\n".join(f"[{cid}]{' (cited)' if cid in c['citations'] else ''} {text}" for cid, text in zip(c["context_ids"], c["contexts"]))
        rows.append({"id": rid, "question": c["question"], "system_answer": c["answer"], "passages_shown": passages,
                     "passages_support_the_answer (yes / partly / no)": "", "answer_is_right (yes / no / cannot tell)": "", "notes": ""})
    (ROOT / "data/judge/hand_label_key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    with (ROOT / "reports/hand_label_sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows ({min(25, len(hidden))} hidden-evidence, {min(25, len(answerable))} answerable)")


if __name__ == "__main__":
    main()
