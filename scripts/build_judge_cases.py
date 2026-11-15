"""The episodes the judge-based RAG metrics are run on (research/2026-11-16-judge-latency-preregistration.md).

  python scripts/build_judge_cases.py --n 300 --seed 61 --out data/judge

Takes the first sealed test (r5t) as scored with scorer v3, keeps the episodes that are eligible for all three raw systems (base, main, candF1), draws
`--n` of them stratified by question group and answerable/hidden-evidence with a fixed seed, and writes one case per system and episode:
the question, the gold answer, the model's own ending, the answer text, and the passages the system was shown (in the order it first saw them).
The sample is drawn from labels alone: no model output, outcome or score enters the draw. `main+FB+V` and `candF1+FB+V` are not judged separately.
Their answers are the raw answers that the fail-closed rule and the self-check let through, so each case records whether it survived them.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

SYSTEMS = ("base", "main", "candF1")


def allocate(strata: dict, n: int) -> dict:
    """Proportional allocation with largest remainders; every stratum that has episodes gets at least one when n allows it."""
    total = sum(len(v) for v in strata.values())
    quota = {k: n * len(v) / total for k, v in strata.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(strata, key=lambda k: (-(quota[k] - take[k]), str(k)))[: n - sum(take.values())]:
        take[k] += 1
    return take


def draw(keys_by_stratum: dict, n: int, seed: int) -> list:
    rng = random.Random(seed)
    take = allocate(keys_by_stratum, n)
    chosen = []
    for stratum in sorted(keys_by_stratum, key=str):
        pool = sorted(keys_by_stratum[stratum])
        chosen += rng.sample(pool, take[stratum])
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--out", default="data/judge")
    args = ap.parse_args()

    import remediate_audit as audit
    import report_azure_compare as azure
    import report_deepseek_compare as deepseek
    from ragharness import evidence

    sources = audit.squad_sources()
    cases, verdicts = deepseek.cases_and_verdicts()
    roots = {"base": (azure.R5REF, "base"), "main": (azure.R5REF, "main"), "candF1": (azure.R5ALT, "candF1")}
    rows, transcripts, corpora = {}, {}, {}
    for name, (root, label) in roots.items():
        rows[name], transcripts[name] = [], []
        for part in azure.PARTS:
            folder = root / f"{label}_bench_{part}/lenient"
            corrected = audit.load(folder, part, sources)[2]
            ts = audit.A.RS.read(folder / "transcripts.jsonl")
            assert len(corrected) == len(ts)
            rows[name] += corrected
            transcripts[name] += ts
            corpora[part] = audit.A.RS.world("bench:" + part)[0]
    wrapped = audit.wrappers(rows, cases, verdicts)
    kept = {name: {(r["part"], r["qid"], r["variant"]) for r in wrapped[name + "+FB+V"] if r["final"] == "answer"} for name in ("main", "candF1")}

    def key(r):
        return (r["part"], r["qid"], r["variant"])

    by_system = {name: {key(r): (r, t) for r, t in zip(rows[name], transcripts[name], strict=True)} for name in SYSTEMS}
    common = set.intersection(*(set(v) for v in by_system.values()))
    eligible = sorted(k for k in common if not any(by_system[name][k][0]["excluded_v3"] for name in SYSTEMS))
    strata = defaultdict(list)
    for k in eligible:
        r = by_system["main"][k][0]
        strata[(r["comp"], bool(r["answerable"]))].append(k)
    chosen = draw(strata, args.n, args.seed)
    sample = [list(k) for k in chosen]
    digest = hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"test": "r5t (first sealed test)", "scorer": "v3", "seed": args.seed, "n": len(chosen), "eligible_common_episodes": len(eligible),
                "strata": {f"{c}|{'answerable' if a else 'hidden'}": {"eligible": len(v), "sampled": sum(1 for k in chosen if k in set(v))} for (c, a), v in sorted(strata.items(), key=str)},
                "sample_sha256": digest, "episodes": sample}
    (out / "sample_r5t.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    with (out / "cases_r5t.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for k in chosen:
            for name in SYSTEMS:
                r, t = by_system[name][k]
                shown = evidence.shown_texts(t, corpora[k[0]])
                final = t["final"]
                f.write(json.dumps({
                    "system": name, "part": k[0], "qid": k[1], "variant": k[2], "comp": r["comp"], "answerable": bool(r["answerable"]),
                    "question": t["question"], "gold": t.get("gold_answer"), "final_kind": final["kind"], "answer": final.get("answer"),
                    "citations": final.get("citations") or [], "context_ids": list(shown), "contexts": list(shown.values()),
                    "outcome_v3": r["outcome_v3"], "survives_fb_v": (k in kept[name]) if name in kept else None,
                }, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n", "eligible_common_episodes", "strata", "sample_sha256")}, indent=1))


if __name__ == "__main__":
    main()
