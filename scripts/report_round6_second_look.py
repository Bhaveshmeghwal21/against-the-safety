"""Report the Round 6 sealed test for the second look (research/2026-11-15-round6-second-look-preregistration.md).

  python scripts/report_round6_second_look.py --run runs/kaggle/ats-r6-system-sl/out [--out reports/round6_second_look]

No model is started. Everything is read from the saved run: <run>/eval/main_bench_r6t_*/lenient, <run>/cases/{answers,recover_contexts,forced_all}.jsonl,
<run>/verify/V and <run>/verify_forced/V. The script refuses to read the sealed outputs until the pre-registration is committed at HEAD and unchanged.

Systems (all from the one main adapter): main, main+FB (an invalid turn ends as an abstention), main+FB+V (the submitted system: V, wording A, accepts
at 0.50), main+FB+V+second look (a refusal after passages were shown is asked once to answer anyway; V scores that answer and it is shown only at >= 0.90),
and always refuse. Scorer v3. Intervals: 3,000 paired cluster draws, seed 47 (SQuAD by article, HotpotQA and RGB by question).
Decision rule (pre-registered): macro difference (second look minus main+FB+V) has a 95% interval above zero AND the hidden-evidence answer-rate
difference has an upper bound of at most +0.02.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import report_azure_compare as azure
import report_round5 as r5
import rescore_round5 as RS
import select_round5 as old_select
from report_reference_policies import always_refuse, answered_stats
from select_round6 import RECOVERY_THRESHOLDS, score_transcript, valid, verdicts

ROOT = Path(__file__).resolve().parents[1]
PREREG = "research/2026-11-15-round6-second-look-preregistration.md"
TEST_SUFFIXES = ("bench_r6t_squad", "bench_r6t_hotpot", "bench_r6t_rgb")
TAU_SECOND_LOOK = 0.90
TAU_FIRST_CHECK = 0.50
ACTOR = "main"
HEADLINE = "main+FB+V+second look"
REFERENCE = "main+FB+V"
DRAWS, SEED = 3000, 47
MAX_HIDDEN_RATE_RISE = 0.02


def committed_preregistration(path=PREREG):
    """The pre-registration must be committed at HEAD, identical to the local file, and name the registered threshold."""
    cmd = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"HEAD:{path}"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Commit the pre-registration before reading any sealed-test output: {path}")
    saved = result.stdout.decode("utf-8").replace("\r\n", "\n")
    if saved != (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n"):
        raise ValueError("The local pre-registration differs from the committed one")
    if "Second-look threshold: 0.90" not in saved:
        raise ValueError("The committed pre-registration does not state the second-look threshold")
    return saved


def label(suffix):
    return suffix.removeprefix("bench_")


def component(part, qtype):
    """Macro components: SQuAD single, SQuAD compound (two-part), HotpotQA (the test part, or the development HotpotQA split), RGB."""
    return "hotpot" if part in ("r6t_hotpot", "dev") else "rgb" if part == "r6t_rgb" else qtype


def run_name(actor, suffix):
    return f"{actor}_{suffix}"


def run_of(actor, row, suffixes):
    """The run folder name a scored row came from (its part label is the suffix without 'bench_')."""
    suffix = "bench_" + row["part"] if "bench_" + row["part"] in suffixes else row["part"]
    return run_name(actor, suffix)


def load_rows(root, actor, suffixes):
    rows = []
    for suffix in suffixes:
        folder = root / "eval" / run_name(actor, suffix) / "lenient"
        _, _, part_rows, mismatches = RS.rescore_run(folder)
        if mismatches:
            raise ValueError(f"Saved and rescored outcomes differ in {folder}")
        for row in part_rows:
            row.update(part=label(suffix), comp=component(label(suffix), row["qtype"]))
            row.update({f"{field}_v2": row[f"{field}_v3"] for field in ("outcome", "good", "strong", "excluded")})
        rows.extend(part_rows)
    return rows


def checked(row, case, probability, threshold):
    """FB and V as submitted: an invalid turn ends as an abstention, and V withholds an answer it does not support."""
    if row["final"] != "answer":
        return old_select.fail_closed([row])[0]
    if case is None:
        raise ValueError("Missing verifier case for an answer")
    accepted = valid(case)
    if accepted:
        if probability is None or not math.isfinite(probability):
            raise ValueError("Missing or invalid verifier probability")
        accepted = probability >= threshold
    if accepted:
        return row
    outcome = "false_abstain" if row["answerable"] else "correct_abstain"
    good = not row["answerable"]
    return {**row, "final": "abstain", "outcome_v2": outcome, "outcome_v3": outcome,
            "good_v2": good, "good_v3": good, "strong_v2": good, "strong_v3": good}


def second_look_entries(root, actor, suffixes, verifier="V"):
    """One entry per refusal that had passages shown: the forced answer (if it passed the deterministic checks), V's probability and the scored replacement."""
    all_contexts = RS.read(root / "cases/recover_contexts.jsonl")
    contexts = [c for c in all_contexts if c["system"] == actor]
    forced = RS.read(root / "cases/forced_all.jsonl")
    by_id = {c["id"]: c for c in forced}
    ids = {c["id"] for c in contexts}
    if len(by_id) != len(forced) or len({c["id"] for c in all_contexts}) != len(all_contexts):
        raise ValueError("Duplicate forced cases or recovery contexts")
    if not set(by_id) <= {c["id"] for c in all_contexts}:
        raise ValueError("Forced cases that belong to no recovery context")
    expected = set()
    for suffix in suffixes:
        run = run_name(actor, suffix)
        for transcript in RS.read(root / "eval" / run / "lenient" / "transcripts.jsonl"):
            shown = transcript.get("provided_evidence", []) or any(s.get("retrieved", []) for s in transcript.get("steps", [])[:-1])
            if transcript["final"]["kind"] in ("abstain", "invalid") and shown:
                expected.add((run, transcript["qid"], transcript["variant"]))
    keys = [(c["run"], c["qid"], c["variant"]) for c in contexts]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Incomplete recovery contexts for {actor}")
    attempted = [item["id"] for item in RS.read(root / "forced" / f"forced_{actor}.jsonl")]
    if len(attempted) != len(set(attempted)) or set(attempted) != ids:
        raise ValueError(f"Incomplete forced outputs for {actor}")
    probabilities = verdicts(root / "verify_forced" / verifier / "verdicts_A.jsonl")
    entries = []
    for ctx in contexts:
        truth = score_transcript(ctx["run"], ctx["transcript"])
        entry = {"key": (ctx["run"], ctx["qid"], ctx["variant"]), "truth": truth, "valid": False, "p": None, "score": None}
        candidate = by_id.get(ctx["id"])
        if candidate is not None and valid(candidate):
            key = (candidate["run"], candidate["qid"], candidate["variant"])
            if key != (ctx["run"] + "_forced", ctx["qid"], ctx["variant"]):
                raise ValueError("Forced answer identity mismatch")
            p = probabilities.get(key)
            if p is None or not math.isfinite(p):
                raise ValueError("Missing forced verifier probability")
            replacement = {**ctx["transcript"], "final": {"kind": "answer", "answer": candidate["answer"], "citations": candidate["citations"]}}
            entry.update(valid=True, p=p, score=score_transcript(ctx["run"], replacement))
        entries.append(entry)
    return entries


def second_look_table(entries):
    """Descriptive: what the second look adds at each threshold (eligible refusals only). Only the registered threshold counts for the decision."""
    eligible = [e for e in entries if not e["truth"]["excluded_v3"]]
    table = []
    for tau in RECOVERY_THRESHOLDS:
        shown = [e for e in eligible if e["p"] is not None and e["p"] >= tau]
        table.append({"threshold": round(tau, 2),
                      "correct_added": sum(e["truth"]["answerable"] and e["score"]["outcome_v3"] == "supported_correct" for e in shown),
                      "wrong_shown": sum(e["truth"]["answerable"] and e["score"]["outcome_v3"] != "supported_correct" for e in shown),
                      "hidden_added": sum(not e["truth"]["answerable"] for e in shown),
                      "answerable_refusals": sum(e["truth"]["answerable"] for e in eligible),
                      "hidden_refusals": sum(not e["truth"]["answerable"] for e in eligible)})
    return table


def apply_second_look(rows, entries, tau, actor, suffixes):
    updates = {e["key"]: e["score"] for e in entries if e["score"] is not None and e["p"] >= tau}
    out = []
    for row in rows:
        key = (run_of(actor, row, suffixes), row["qid"], row["variant"])
        if key in updates and row["final"] == "abstain":
            new = dict(updates[key])
            new.update(part=row["part"], comp=row["comp"])
            new.update({f"{field}_v2": new[f"{field}_v3"] for field in ("outcome", "good", "strong", "excluded")})
            row = new
        out.append(row)
    return out


def build_systems(root, actor=ACTOR, suffixes=TEST_SUFFIXES, tau=TAU_SECOND_LOOK):
    """The named systems, from one saved run. Returns (systems, second-look entries)."""
    cases = {(c["run"], c["qid"], c["variant"]): c for c in RS.read(root / "cases/answers.jsonl")}
    preds = verdicts(root / "verify/V/verdicts_A.jsonl")
    raw = load_rows(root, actor, suffixes)
    checked_rows = []
    for row in raw:
        key = (run_of(actor, row, suffixes), row["qid"], row["variant"])
        checked_rows.append(checked(row, cases.get(key), preds.get(key), TAU_FIRST_CHECK))
    entries = second_look_entries(root, actor, suffixes)
    systems = {actor: raw, f"{actor}+FB": old_select.fail_closed(raw), REFERENCE: checked_rows,
               HEADLINE: apply_second_look(checked_rows, entries, tau, actor, suffixes), "always refuse": always_refuse(raw)}
    return systems, entries


def shared_keys(systems):
    indexed = {name: azure.by_key(rows) for name, rows in systems.items()}
    present = set.intersection(*(set(v) for v in indexed.values()))
    if len(present) != len(next(iter(indexed.values()))):
        raise ValueError("Systems do not cover the same episodes")
    for key in present:
        if len({r[key]["excluded_v3"] for r in indexed.values()}) != 1:
            raise ValueError(f"Model-dependent eligibility: {key}")
    keys = sorted(k for k in present if not indexed[ACTOR][k]["excluded_v3"])
    if not keys:
        raise ValueError("No eligible episodes")
    return indexed, keys


def cluster_intervals(indexed, keys, comparisons, groups):
    """Paired percentile bootstrap, whole clusters resampled within each part; the same draws for every system."""
    clusters = {}
    for index, (part, qid, _) in enumerate(keys):
        clusters.setdefault(part, {}).setdefault(groups[(part, qid)], []).append(index)
    comps, template = r5.vectors([indexed[next(iter(indexed))][key] for key in keys], keys)
    grouped = {}
    for name, rows in indexed.items():
        _, vector = r5.vectors([rows[key] for key in keys], keys)
        grouped[name] = {part: np.array([vector[idx].sum(axis=0) for idx in members.values()]) for part, members in clusters.items()}
    rng = np.random.default_rng(SEED)
    totals = {name: np.zeros((DRAWS, template.shape[1])) for name in indexed}
    for part, members in sorted(clusters.items()):
        picks = rng.integers(0, len(members), (DRAWS, len(members)))
        for name in indexed:
            totals[name] += grouped[name][part][picks].sum(axis=1)
    sampled = {name: [r5.stats_from(row, comps) for row in sums] for name, sums in totals.items()}
    observed = {name: azure.r5_stats(rows, keys) for name, rows in indexed.items()}
    result = {}
    for a, b in comparisons:
        result[f"{a}/{b}"] = {metric: {"difference": observed[a][metric] - observed[b][metric],
                                       "ci95": np.quantile([x[metric] - y[metric] for x, y in zip(sampled[a], sampled[b])], [.025, .975]).tolist()}
                              for metric in ("macro", "unsupported_rate", "supported_correct")}
    return result


def decision(comparison):
    """The pre-registered adoption rule."""
    passes_macro = comparison["macro"]["ci95"][0] > 0
    passes_hidden = comparison["unsupported_rate"]["ci95"][1] <= MAX_HIDDEN_RATE_RISE
    return {"macro_interval_above_zero": bool(passes_macro), "hidden_rate_rise_at_most_0.02": bool(passes_hidden), "adopt": bool(passes_macro and passes_hidden)}


def cluster_groups():
    """SQuAD questions cluster by article (the title without its part suffix); HotpotQA and RGB by question."""
    groups = {}
    for part in ("r6t_squad", "r6t_hotpot", "r6t_rgb"):
        _, questions = RS.world(f"bench:{part}")
        for qid, question in questions.items():
            groups[(part, qid)] = question.paragraphs[0].title.rsplit(" (part ", 1)[0] if part == "r6t_squad" else qid
    return groups


def integrity(root, actor, suffixes):
    info = {}
    for suffix in suffixes:
        summary = json.loads((root / "eval" / run_name(actor, suffix) / "lenient" / "summary.json").read_text(encoding="utf-8"))
        generation = summary.get("generation", {})
        info[label(suffix)] = {"truncated_inputs": generation.get("truncated_inputs"), "max_input_tokens": generation.get("max_input_tokens")}
    return info


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="completed r6-system out directory for main on the three sealed parts")
    ap.add_argument("--out", default="reports/round6_second_look")
    ap.add_argument("--skip-git-check", action="store_true", help="for tests on synthetic runs only; never for the real report")
    args = ap.parse_args()
    if not args.skip_git_check:
        committed_preregistration()
    root = ROOT / args.run
    systems, entries = build_systems(root)
    indexed, keys = shared_keys(systems)
    summaries = {name: {**azure.r5_stats(rows, keys), **answered_stats([rows[k] for k in keys])} for name, rows in indexed.items()}
    pairs = [(HEADLINE, REFERENCE), (REFERENCE, f"{ACTOR}+FB"), (HEADLINE, f"{ACTOR}+FB"), (REFERENCE, ACTOR)]
    comparisons = cluster_intervals(indexed, keys, pairs, cluster_groups())
    verdict = decision(comparisons[f"{HEADLINE}/{REFERENCE}"])
    table = second_look_table(entries)
    registered = next(r for r in table if r["threshold"] == TAU_SECOND_LOOK)
    report = {"scorer": "v3", "shared_eligible": len(keys), "systems": summaries, "comparisons": comparisons, "decision": verdict,
              "registered_second_look_threshold": TAU_SECOND_LOOK, "registered_threshold_counts": registered,
              "second_look_table_descriptive": table, "integrity": integrity(root, ACTOR, TEST_SUFFIXES),
              "bootstrap": f"{DRAWS} paired cluster draws, seed {SEED}; SQuAD by article, HotpotQA and RGB by question"}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Round 6 sealed test: the second look checked by V", "", report["bootstrap"] + f". Shared eligible episodes: {len(keys)}.", "",
             "| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answered |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {pct(s['macro'])} | {pct(s['supported_correct'])} | {s['unsupported']}/{s['hidden_n']} | "
                     f"{pct(s['answered_share'])} | {pct(s['correct_among_answered'])} |")
    lines += ["", "## Pre-registered decision", ""]
    c = comparisons[f"{HEADLINE}/{REFERENCE}"]
    lines.append(f"- {HEADLINE} versus {REFERENCE}: {'ADOPT' if verdict['adopt'] else 'DO NOT ADOPT'}; "
                 f"macro difference {c['macro']['difference']:+.3f} (95% interval {c['macro']['ci95'][0]:+.3f} to {c['macro']['ci95'][1]:+.3f}); "
                 f"hidden-answer-rate difference {c['unsupported_rate']['difference']:+.3f} (95% interval {c['unsupported_rate']['ci95'][0]:+.3f} to {c['unsupported_rate']['ci95'][1]:+.3f}).")
    lines += ["", "## Other paired comparisons (descriptive)", "", "| Comparison | Macro | Correct and cited | Hidden-answer rate |", "| --- | ---: | ---: | ---: |"]
    ci = lambda m: f"{100 * m['difference']:+.1f} pp [{100 * m['ci95'][0]:+.1f}, {100 * m['ci95'][1]:+.1f}]"
    for name, comp in comparisons.items():
        lines.append(f"| {name} | {ci(comp['macro'])} | {ci(comp['supported_correct'])} | {ci(comp['unsupported_rate'])} |")
    lines += ["", "## What the second look adds at each threshold (descriptive; only 0.90 was registered)", "",
              "| Threshold | Correct and cited added | Wrong answers shown | Hidden-evidence answers added |", "| ---: | ---: | ---: | ---: |"]
    for row in table:
        lines.append(f"| {row['threshold']:.2f} | {row['correct_added']} | {row['wrong_shown']} | {row['hidden_added']}/{row['hidden_refusals']} |")
    lines += ["", f"Integrity (truncated prompts and input limit per part): {report['integrity']}"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required saved artifact is missing: {exc.filename or exc}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
