"""Priority 0c: rescore saved transcripts under scorer version 2 and report both versions side by side (no model calls).

  python scripts/rescore_round5.py --eval runs/account2_round4/ats-eval-jobs-r4dev/out/eval --out reports/round5_rescore_r4dev
  python scripts/rescore_round5.py --eval <folder> --verify-only      # only check that version 1 reproduces the stored scores

Run folders are named <system>_bench_<name> (data/bench/<name>) or <system>_dev | _test | _test2 (HotpotQA). Per-run rescored rows go to
.cache/rescored/<run>.jsonl; the committed outputs are the summary tables and the list of excluded episode ids.
"""

import argparse
import json
from pathlib import Path

from ragharness import corpus as corpus_mod, data, rescore

ROOT = Path(__file__).resolve().parents[1]
_WORLD: dict = {}


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def world(kind: str):
    """(corpus, {qid: question}) for 'bench:<name>' or 'hotpot:<dev|test|test2>'."""
    if kind in _WORLD:
        return _WORLD[kind]
    if kind.startswith("bench:"):
        d = ROOT / "data" / "bench" / kind[6:]
        corp = corpus_mod.load_corpus(d / "corpus.json")
        qs = data.read_questions(d / "questions.jsonl")
    else:
        man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
        splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
        corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
        name = kind[7:]
        if name == "test2":
            qs = data.read_questions(ROOT / "data" / "processed" / "questions_test2.jsonl")
            corp = corpus_mod.extend_corpus(corp, qs)
        else:
            qs = splits[name]
    _WORLD[kind] = (corp, {q.qid: q for q in qs})
    return _WORLD[kind]


def kind_of(run: str) -> tuple[str, str]:
    """(system, world kind) from a run folder name."""
    if "_bench_" in run:
        system, name = run.split("_bench_", 1)
        return system, f"bench:{name}"
    system, split = run.rsplit("_", 1)
    return system, f"hotpot:{split}"


def rescore_run(folder: Path):
    run = folder.parent.name
    system, kind = kind_of(run)
    corp, qmap = world(kind)
    transcripts, stored = read(folder / "transcripts.jsonl"), read(folder / "scores.jsonl")
    assert len(transcripts) == len(stored), run
    rows, mismatches = [], []
    variant_path = ROOT / "data/bench" / kind[6:] / "prompt_variants.json" if kind.startswith("bench:") else None
    variants = json.loads(variant_path.read_text(encoding="utf-8")) if variant_path and variant_path.exists() else {}
    for t, s in zip(transcripts, stored):
        assert (t["qid"], t["variant"]) == (s["qid"], s["variant"]), run
        q = qmap[t["qid"]]
        if "masked" not in t:
            mask = variants.get(q.qid, {}).get(t["variant"], {}).get("masked")
            if mask is None:
                mask = list(corp.support_copies(q)) if t["variant"] == "support_masked" else []
            t = {**t, "masked": mask}
        r = rescore.score(t, q, corp)
        stored_version = "v3" if s.get("scorer") == "v3" else "v1"
        if r[f"outcome_{stored_version}"] != s["outcome"] or r[f"strong_{stored_version}"] != s.get("evidence_supported_good", r[f"strong_{stored_version}"]):
            mismatches.append((t["qid"], t["variant"], s["outcome"], r[f"outcome_{stored_version}"]))
        rows.append(r)
    return system, kind, rows, mismatches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--out", default="reports/round5_rescore")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    cache = ROOT / ".cache" / "rescored"
    cache.mkdir(parents=True, exist_ok=True)
    results, excluded, problems = {}, {}, {}
    for root in args.eval:
        for folder in sorted(Path(root).rglob("lenient")):
            if not (folder / "transcripts.jsonl").exists():
                continue
            run = folder.parent.name
            system, kind, rows, mismatches = rescore_run(folder)
            problems[run] = len(mismatches)
            if mismatches:
                print(f"{run}: stored scorer differs on replay in {len(mismatches)} of {len(rows)} rows, e.g. {mismatches[:3]}")
            (cache / f"{run}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            results[run] = {"system": system, "world": kind, **{v: rescore.summarize(rows, v) for v in ("v1", "v2", "v3")}}
            excluded[run] = sorted(f"{r['qid']}:{r['variant']}" for r in rows if r["excluded_v2"])
    if args.verify_only:
        print(json.dumps(problems, indent=1))
        return
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps({"results": results, "excluded_v2": excluded, "v1_mismatches": problems}, indent=1), encoding="utf-8")
    lines = ["# Frozen scorers v1/v2 and corrected scorer v3 on saved transcripts", "",
             "v3 preserves exact answers, scores displayed evidence alternatives and fixes eligibility from corpus/masks. Use scripts/remediate_audit.py for source-offset corrections and regenerated comparison tables.", "",
             "v2 = token precision >= 0.5 against the closest gold form, plus exclusion of hidden-evidence episodes where a shown passage holds the answer.", "",
             "| run | scorer | n | single | compound | supported success | unsupported / hidden | false refusals | invalid |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    f = lambda x: "-" if x is None else f"{x:.3f}"
    for run, r in results.items():
        for v in ("v1", "v2", "v3"):
            s = r[v]
            lines.append(f"| {run} | {v} | {s['n']} | {f(s['single'])} | {f(s['compound'])} | {f(s['supported_success'])} | {s['unsupported']}/{s['hidden_n']} | {f(s['false_abstain'])} | {f(s['invalid'])} |")
    lines += ["", "Excluded under v2 (hidden-evidence episodes with a shown passage holding the answer):", ""]
    lines += [f"- {run}: {len(ids)}" + (f" ({', '.join(ids[:12])}{' ...' if len(ids) > 12 else ''})" if ids else "") for run, ids in excluded.items()]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
