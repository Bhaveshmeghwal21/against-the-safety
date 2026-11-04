"""How much of the gold supporting text survives a uniform per-passage character limit (plan section 4.1).

A passage limit shortens every prompt. It is safe only if it rarely cuts off a supporting sentence, and the cut
must be uniform (the same rule for support and distractor passages) so passage length cannot reveal the label.

Run (after prepare_data.py):  python scripts/truncation_report.py
"""

import json
from pathlib import Path

from ragharness import data

ROOT = Path(__file__).resolve().parents[1]
LIMITS = (350, 400, 500, 600, 800, 1000, None)


def main() -> None:
    rep: dict = {}
    for split in ("train", "dev", "test"):
        qs = data.read_questions(ROOT / "data" / "processed" / f"questions_{split}.jsonl")
        para_total = 0
        para_len: list[int] = []
        res = {str(lim): {"questions_all_support_visible": 0, "support_paragraphs_cut": 0} for lim in LIMITS}
        for q in qs:
            sup_sents = q.support_sentences()
            for lim in LIMITS:
                ok = True
                for title, sents in sup_sents.items():
                    text = q.paragraph(title).text
                    visible = text if lim is None else text[:lim]
                    if any(s not in visible for s in sents):
                        res[str(lim)]["support_paragraphs_cut"] += 1
                        ok = False
                res[str(lim)]["questions_all_support_visible"] += ok
            para_total += len(sup_sents)
            para_len += [len(p.text) for p in q.paragraphs]
        for lim in LIMITS:
            r = res[str(lim)]
            r["share_questions_kept"] = round(r["questions_all_support_visible"] / len(qs), 3)
        para_len.sort()
        rep[split] = {"questions": len(qs), "support_paragraphs": para_total,
                      "paragraph_chars_mean": round(sum(para_len) / len(para_len)), "paragraph_chars_p90": para_len[int(0.9 * len(para_len))],
                      "limits": res}
    (ROOT / "reports" / "truncation_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8", newline="\n")
    print(f"{'limit':>6s} " + " ".join(f"{s:>22s}" for s in ("train kept / paras cut", "dev kept / paras cut", "test kept / paras cut")))
    for lim in LIMITS:
        row = []
        for s in ("train", "dev", "test"):
            r = rep[s]["limits"][str(lim)]
            row.append(f"{r['share_questions_kept']:.3f} / {r['support_paragraphs_cut']:4d}")
        print(f"{str(lim):>6s} " + " ".join(f"{x:>22s}" for x in row))
    print({s: (rep[s]["paragraph_chars_mean"], rep[s]["paragraph_chars_p90"]) for s in rep})


if __name__ == "__main__":
    main()
