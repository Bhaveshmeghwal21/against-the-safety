"""Report the Hindi training experiment (research/2026-11-15-hindi-training-preregistration.md). No model is started.

  python scripts/report_hindi.py --parents runs/kaggle/<account-2 run>/out --new runs/kaggle/<account-1 evaluation run>/out \
      --train runs/kaggle/ats-hi-train-a/out [--out reports/hindi_results]

Reads <root>/eval/<system>_bench_<set>/lenient, <root>/cases/answers.jsonl and <root>/verify/V/verdicts_A.jsonl.
`--parents` holds main and candF1 on xq2_hi, xq2_en, hi_dev and candF1 on r6t_*; `--new` holds hiMain and hiF on xq2_hi, xq2_en, r6t_*; `--train` holds
the training reports and the hi_dev runs of hiMain and hiF; main's r6t run is `runs/kaggle/ats-r6-system-sl/out` (the second-look test).
FB (an invalid turn ends as an abstention) and V (the main adapter, wording A, threshold 0.50) are applied offline to every system alike.
The script refuses to read any xq2 output until the pre-registration is committed at HEAD and unchanged.

Decision rules, per new adapter N against its parent P, both with FB and V (pre-registered):
  A1  xq2_hi: correct and cited, N minus P, 95% interval lower end above zero
  A2  xq2_hi: rate of answers on hidden evidence, N minus P, upper end at most +0.02
  A3  r6t:    macro success, N minus P, lower end at least -0.03 and hidden-answer rate upper end at most +0.02
Intervals: 3,000 paired cluster draws, seed 47 (XQuAD by article; r6t as in scripts/report_round6_second_look.py).
"""

import argparse
import json
import subprocess
from pathlib import Path

import report_azure_compare as azure
import report_round6_second_look as R2
import report_xquad as XQ
import rescore_round5 as RS
import select_round5 as old_select
from report_reference_policies import always_refuse
from select_round6 import verdicts

ROOT = Path(__file__).resolve().parents[1]
PREREG = "research/2026-11-15-hindi-training-preregistration.md"
PAIRS = (("hiMain", "main"), ("hiF", "candF1"))
SYSTEMS = ("main", "candF1", "hiMain", "hiF")
MAIN_R6T = "runs/kaggle/ats-r6-system-sl/out"
A3_MARGIN, HIDDEN_LIMIT = -0.03, 0.02


def committed_preregistration() -> str:
    cmd = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"HEAD:{PREREG}"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Commit the pre-registration before reading any output of the fresh test: {PREREG}")
    saved = result.stdout.decode("utf-8").replace("\r\n", "\n")
    if saved != (ROOT / PREREG).read_text(encoding="utf-8").replace("\r\n", "\n"):
        raise ValueError("The local pre-registration differs from the committed one")
    return saved


class Run:
    """One downloaded kernel output: its evaluation folders, the answer cases and V's verdicts."""

    def __init__(self, path: str):
        self.root = ROOT / path
        cases = self.root / "cases" / "answers.jsonl"
        verdict = self.root / "verify" / "V" / "verdicts_A.jsonl"
        self.cases = {(c["run"], c["qid"], c["variant"]): c for c in RS.read(cases)} if cases.exists() else {}
        self.preds = verdicts(verdict) if verdict.exists() else {}

    def has(self, system: str, bench: str) -> bool:
        return (self.root / "eval" / f"{system}_bench_{bench}" / "lenient" / "transcripts.jsonl").exists()

    def configs(self, system: str, bench: str, with_v: bool = True) -> dict[str, list[dict]]:
        folder = self.root / "eval" / f"{system}_bench_{bench}" / "lenient"
        _, _, raw, mismatches = RS.rescore_run(folder)
        if mismatches:
            raise ValueError(f"Saved and rescored outcomes differ in {folder}")
        out = {system: raw, f"{system}+FB": old_select.fail_closed(raw)}
        if with_v:
            run = f"{system}_bench_{bench}"
            out[f"{system}+FB+V"] = [R2.checked(r, self.cases.get((run, r["qid"], r["variant"])), self.preds.get((run, r["qid"], r["variant"])), R2.TAU_FIRST_CHECK) for r in raw]
        return out

    def summary(self, system: str, bench: str) -> dict:
        return json.loads((self.root / "eval" / f"{system}_bench_{bench}" / "lenient" / "summary.json").read_text(encoding="utf-8")).get("generation", {})


def find(runs: list[Run], system: str, bench: str) -> Run:
    for run in runs:
        if run.has(system, bench):
            return run
    raise FileNotFoundError(f"{system} on {bench}")


def xquad_set(runs: list[Run], bench: str, lang: str, with_v: bool = True) -> dict:
    manifest = json.loads((ROOT / "data" / "bench" / bench / "manifest.json").read_text(encoding="utf-8"))
    article = manifest["article_of"]
    data = {}
    for system in SYSTEMS:
        data.update(find(runs, system, f"{bench}").configs(system, bench, with_v))
    ref = data["main+FB+V"] if with_v else data["main+FB"]
    keyset = lambda rows: {(r["qid"], r["variant"], r["excluded_v3"]) for r in rows}
    for name, rows in data.items():
        if keyset(rows) != keyset(ref):
            raise ValueError(f"model-dependent eligibility in {bench}/{name}")
    data["always refuse"] = always_refuse(data["main"])
    data = {name: [r for r in rows if not r["excluded_v3"]] for name, rows in data.items()}
    result = {"systems": {name: XQ.measures(rows) for name, rows in data.items()}, "comparisons": {}, "lang": lang}
    for new, parent in PAIRS:
        for suffix in ("+FB", "+FB+V") if with_v else ("+FB",):
            result["comparisons"][f"{new}{suffix} minus {parent}{suffix}"] = XQ.paired(data[new + suffix], data[parent + suffix], article)
    return result


def r6t_set(parents: Run, new: Run) -> dict:
    systems, where = {}, {"main": Run(MAIN_R6T), "candF1": parents, "hiMain": new, "hiF": new}
    for name, run in where.items():
        raw = R2.load_rows(run.root, name, R2.TEST_SUFFIXES)
        checked = [R2.checked(r, run.cases.get((R2.run_of(name, r, R2.TEST_SUFFIXES), r["qid"], r["variant"])), run.preds.get((R2.run_of(name, r, R2.TEST_SUFFIXES), r["qid"], r["variant"])),
                              R2.TAU_FIRST_CHECK) for r in raw]
        systems[f"{name}+FB+V"] = checked
        systems[f"{name}+FB"] = old_select.fail_closed(raw)
    systems["always refuse"] = always_refuse(R2.load_rows(where["main"].root, "main", R2.TEST_SUFFIXES))
    indexed = {name: azure.by_key(rows) for name, rows in systems.items()}
    present = set.intersection(*(set(v) for v in indexed.values()))
    if any(len({r[k]["excluded_v3"] for r in indexed.values()}) != 1 for k in present):
        raise ValueError("Model-dependent eligibility on r6t")
    keys = sorted(k for k in present if not indexed["main+FB+V"][k]["excluded_v3"])
    summaries = {name: azure.r5_stats(rows, keys) for name, rows in indexed.items()}
    pairs = [(f"{n}+FB+V", f"{p}+FB+V") for n, p in PAIRS]
    return {"shared_eligible": len(keys), "systems": summaries, "comparisons": R2.cluster_intervals(indexed, keys, pairs, R2.cluster_groups())}


def decide(new: str, parent: str, hindi: dict, english: dict) -> dict:
    c = hindi["comparisons"][f"{new}+FB+V minus {parent}+FB+V"]
    e = english["comparisons"][f"{new}+FB+V/{parent}+FB+V"]
    a1 = c["correct_and_cited"]["ci95"][0] > 0
    a2 = c["hidden_answer_rate"]["ci95"][1] <= HIDDEN_LIMIT
    a3 = e["macro"]["ci95"][0] >= A3_MARGIN and e["unsupported_rate"]["ci95"][1] <= HIDDEN_LIMIT
    return {"A1_hindi_gain": bool(a1), "A2_hindi_safety": bool(a2), "A3_english_not_worse": bool(a3), "published_as_hindi_variant": bool(a1 and a2 and a3)}


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def table(systems: dict, names) -> list[str]:
    lines = ["| system | episodes | success | correct and cited | answers on hidden evidence | false refusals | invalid raw turns | correct among answers shown |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        m = systems[name]
        lines.append(f"| {name} | {m['n']} | {pct(m['success'])} | {pct(m['correct_and_cited'])} | {m['hidden_answers']}/{m['n_hidden']} ({pct(m['hidden_answer_rate'])}) | "
                     f"{pct(m['false_refusal'])} | {pct(m['invalid'])} | {pct(m['correct_among_answered'])} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="reports/hindi_results")
    ap.add_argument("--skip-git-check", action="store_true", help="synthetic tests only")
    args = ap.parse_args()
    if not args.skip_git_check:
        committed_preregistration()
    parents, new, train = Run(args.parents), Run(args.new), Run(args.train)
    hindi = xquad_set([parents, new], "xq2_hi", "hi")
    english = xquad_set([parents, new], "xq2_en", "en")
    r6t = r6t_set(parents, new)
    dev = {}
    for system in SYSTEMS:
        configs = find([parents, train], system, "hi_dev").configs(system, "hi_dev", with_v=False)
        dev[system] = {name: XQ.measures([r for r in rows if not r["excluded_v3"]]) for name, rows in configs.items()}
    decisions = {new_name: decide(new_name, parent, hindi, r6t) for new_name, parent in PAIRS}
    integrity = {}
    for system in SYSTEMS:
        for bench in ("xq2_hi", "xq2_en", "hi_dev", "r6t_squad", "r6t_hotpot", "r6t_rgb"):
            runs = [r for r in ((Run(MAIN_R6T),) if system == "main" and bench.startswith("r6t") else (parents, new, train)) if r.has(system, bench)]
            if runs:
                g = runs[0].summary(system, bench)
                integrity[f"{system}_{bench}"] = {"truncated_inputs": g.get("truncated_inputs"), "max_input_tokens": g.get("max_input_tokens")}
    reports = {n: json.loads((train.root / n / "train_report.json").read_text(encoding="utf-8")) for n in ("hiMain", "hiF") if (train.root / n / "train_report.json").exists()}
    report = {"preregistration": PREREG, "decisions": decisions, "xq2_hi": hindi, "xq2_en": english, "r6t": r6t, "hi_dev": dev, "truncated_prompts": integrity,
              "training": {n: {k: r.get(k) for k in ("rows_trained", "episodes", "processed_tokens", "wall_seconds", "peak_gpu_memory_gb", "train_loss", "optimizer_steps")}
                           for n, r in reports.items()},
              "bootstrap": f"{R2.DRAWS} paired cluster draws, seed {R2.SEED}"}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=float) + "\n", encoding="utf-8")
    ci = lambda m: f"{100 * m['difference']:+.1f} pp [{100 * m['ci95'][0]:+.1f}, {100 * m['ci95'][1]:+.1f}]"
    names = [f"{s}{x}" for s in SYSTEMS for x in ("", "+FB", "+FB+V")] + ["always refuse"]
    lines = ["# Hindi training of main and candF1: results", "", report["bootstrap"] + ". Scorer v3 (XQuAD normalization for Hindi and English).", "", "## Decisions", ""]
    for name, d in decisions.items():
        lines.append(f"- {name}: " + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in d.items()))
    for title, key, data in (("Fresh Hindi test (xq2_hi, 300 XQuAD questions)", "xq2_hi", hindi), ("Same questions in English (xq2_en)", "xq2_en", english)):
        lines += ["", f"## {title}", ""] + table(data["systems"], [n for n in names if n in data["systems"]])
        lines += ["", "| comparison | questions | correct and cited | answers on hidden evidence | success |", "| --- | ---: | ---: | ---: | ---: |"]
        for n, c in data["comparisons"].items():
            lines.append(f"| {n} | {c['questions']} | {ci(c['correct_and_cited'])} | {ci(c['hidden_answer_rate'])} | {ci(c['success'])} |")
    lines += ["", f"## English regression (Round 6 test, {r6t['shared_eligible']} shared eligible episodes)", "",
              "| system | macro success | correct and cited | answers on hidden evidence |", "| --- | ---: | ---: | ---: |"]
    for n, s in r6t["systems"].items():
        if n.endswith("+FB+V") or n == "always refuse":
            lines.append(f"| {n} | {pct(s['macro'])} | {pct(s['supported_correct'])} | {s['unsupported']}/{s['hidden_n']} |")
    lines += ["", "| comparison | macro success | correct and cited | hidden-answer rate |", "| --- | ---: | ---: | ---: |"]
    for n, c in r6t["comparisons"].items():
        lines.append(f"| {n} | {ci(c['macro'])} | {ci(c['supported_correct'])} | {ci(c['unsupported_rate'])} |")
    lines += ["", "## Development set (hi_dev, 120 machine-translated questions; sanity only, no rule depends on it)", ""]
    dev_rows = [row for system in SYSTEMS for row in table(dev[system], [system, f"{system}+FB"])[2:]]
    lines += table(dev["main"], [])[:2] + dev_rows
    truncated = {k: v for k, v in integrity.items() if v["truncated_inputs"]}
    limits = sorted({v["max_input_tokens"] for v in integrity.values() if v["max_input_tokens"] is not None})
    note = "No prompt was truncated in any run, so the differing limits do not matter." if not truncated and len(limits) > 1 else ""
    lines += ["", f"Prompts truncated by the input limit: {truncated or 'none'}. Input limits used: {limits}; main's r6t run is the Round 6 test run. {note}".strip()]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required saved artifact is missing: {exc.filename or exc}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
