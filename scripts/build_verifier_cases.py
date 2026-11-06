"""Priority 1: the cases the answer checker is run on (plan section 27), built from saved development transcripts only.

  python scripts/build_verifier_cases.py --eval runs/account2_round4/ats-eval-jobs-r4dev/out/eval --system main --out data/verifier/cases_dev.jsonl

One case per answered episode of the chosen system on r4dev and HotpotQA dev. A case holds the question, the answer, the cited passages exactly as
the tool rendered them, the version-2 outcome labels and the deterministic checks (cited chunk never shown; answer text absent from the cited text).
"""

import argparse
import json
from pathlib import Path

from ragharness import rescore
from ragharness.retrieval import Hit, render_observation, visible_text

import rescore_round5 as RS

ROOT = Path(__file__).resolve().parents[1]


def answer_parts(answer: str) -> list[str]:
    return [p.strip() for p in answer.split(";") if p.strip()]


def contained(part: str, text: str) -> bool | None:
    """Every content token of the answer part occurs in the cited text; None when there is nothing to check (yes/no, only stop words)."""
    toks = rescore.tokens(part)
    if not toks or part.strip().lower() in ("yes", "no"):
        return None
    have = set(rescore.tokens(text))
    return all(t in have for t in toks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--system", default="main")
    ap.add_argument("--out", default="data/verifier/cases_dev.jsonl")
    args = ap.parse_args()
    cases, counts = [], {}
    for root in args.eval:
        for folder in sorted(Path(root).rglob("lenient")):
            run = folder.parent.name
            system, kind = RS.kind_of(run)
            if system != args.system or not (folder / "transcripts.jsonl").exists():
                continue
            corp, qmap = RS.world(kind)
            for t in RS.read(folder / "transcripts.jsonl"):
                if t["final"]["kind"] != "answer":
                    continue
                q = qmap[t["qid"]]
                sc = rescore.score(t, q, corp)
                shown = [h["chunk_id"] for h in t.get("provided_evidence", [])] + [h["chunk_id"] for s in t["steps"] for h in s["retrieved"]]
                cited = t["final"]["citations"] or []
                unseen = [c for c in cited if c not in shown]
                seen = [c for c in dict.fromkeys(cited) if c in shown]
                hits = [Hit(c, corp.chunk(c)["title"], corp.chunk(c)["text"], 0.0) for c in seen]
                passages = render_observation(hits, frozenset(), 800) if hits else ""
                cited_text = " ".join(visible_text(corp.chunk(c)["text"], 800) for c in seen)
                parts = answer_parts(t["final"]["answer"] or "")
                missing = [p for p in parts if contained(p, cited_text) is False]
                cases.append({
                    "run": run, "world": kind, "qid": t["qid"], "variant": t["variant"], "question": t["question"], "answer": t["final"]["answer"],
                    "citations": cited, "passages": passages,
                    "labels": {"answerable": sc["answerable"], "outcome_v1": sc["outcome_v1"], "outcome_v2": sc["outcome_v2"], "excluded_v2": sc["excluded_v2"]},
                    "deterministic": {"cites_unshown": bool(unseen) or not seen, "answer_part_absent_from_cited": bool(missing), "absent_parts": missing},
                })
                counts[run] = counts.get(run, 0) + 1
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "by_run": counts, "path": args.out}))


if __name__ == "__main__":
    main()
