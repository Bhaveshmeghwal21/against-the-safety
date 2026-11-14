"""Report the multilingual check (research/2026-11-15-multilingual-plan.md, section 6). No model is started; everything is read from the saved run.

  python scripts/report_xquad.py --run runs/kaggle/ats-r6-system-xq/out [--out reports/xquad_results]

Reads <run>/eval/<system>_bench_xq_<lang>/lenient (base and main), <run>/cases/answers.jsonl and <run>/verify/V/verdicts_A.jsonl. Every transcript is
rescored with scorer v3 (the language tag on each question switches on the XQuAD normalization) and compared with the scores stored by the run.
FB (an invalid turn ends as an abstention) and V (the self-check, wording A, threshold 0.5) are applied offline, as in rounds 5 and 6.
Intervals are paired percentile bootstraps over the 48 Wikipedia articles (3,000 draws, seed 47).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import rescore_round5 as RS
import select_round5 as old_select
from report_reference_policies import always_refuse, answered_stats
from select_round6 import valid, verdicts
from ragharness import textnorm

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("hi", "en", "es")
NAMES = {"hi": "Hindi", "en": "English", "es": "Spanish"}
HEADLINE = "main+FB+V"
DRAWS, SEED = 3000, 47
THRESHOLD = 0.5


def checked(row, case, probability):
    """The submitted system's repair: an invalid turn ends as an abstention, and V withholds an answer it does not support."""
    if row["final"] != "answer":
        return old_select.fail_closed([row])[0]
    if case is None:
        raise ValueError("missing verifier case for an answer")
    ok = valid(case)
    if ok:
        if probability is None or not math.isfinite(probability):
            raise ValueError("missing or invalid verifier probability")
        ok = probability >= THRESHOLD
    if ok:
        return row
    outcome = "false_abstain" if row["answerable"] else "correct_abstain"
    good = not row["answerable"]
    return {**row, "final": "abstain", "outcome_v2": outcome, "outcome_v3": outcome, "good_v2": good, "good_v3": good, "strong_v2": good, "strong_v3": good}


def load_run(root: Path, system: str, lang: str):
    folder = root / "eval" / f"{system}_bench_xq_{lang}" / "lenient"
    _, _, rows, mismatches = RS.rescore_run(folder)
    transcripts = RS.read(folder / "transcripts.jsonl")
    for r, t in zip(rows, transcripts):
        r["answer"] = t["final"].get("answer") if t["final"]["kind"] == "answer" else None
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    return rows, mismatches, summary


def measures(rows):
    ans = [r for r in rows if r["answerable"]]
    hid = [r for r in rows if not r["answerable"]]
    cc = sum(r["outcome_v3"] == "supported_correct" for r in ans)
    ca = sum(r["outcome_v3"] == "correct_abstain" for r in hid)
    hidden_answers = sum(r["outcome_v3"].startswith("unsupported_") for r in hid)
    shown = answered_stats(rows)
    rate = lambda a, b: a / b if b else None
    return {"n": len(rows), "n_answerable": len(ans), "n_hidden": len(hid),
            "success": rate(cc + ca, len(rows)), "correct_and_cited": rate(cc, len(ans)),
            "hidden_answers": hidden_answers, "hidden_answer_rate": rate(hidden_answers, len(hid)),
            "false_refusal": rate(sum(r["outcome_v3"] == "false_abstain" for r in ans), len(ans)),
            "invalid": rate(sum(r["outcome_v3"] == "invalid" for r in rows), len(rows)),
            "answered": shown["answered"], "answered_share": shown["answered_share"], "correct_among_answered": shown["correct_among_answered"]}


def cluster_table(rows, article):
    """Per article: [correct and cited, answerable, hidden answers, hidden, good, all] over the given rows."""
    table = {}
    for r in rows:
        t = table.setdefault(article[r["qid"]], np.zeros(6))
        t += [r["answerable"] and r["outcome_v3"] == "supported_correct", r["answerable"],
              (not r["answerable"]) and r["outcome_v3"].startswith("unsupported_"), not r["answerable"], r["good_v3"], 1]
    return table


def ratios(v):
    return np.array([v[0] / v[1] if v[1] else np.nan, v[2] / v[3] if v[3] else np.nan, v[4] / v[5] if v[5] else np.nan])


def paired(rows_a, rows_b, article):
    """Difference (a minus b) of correct-and-cited, hidden-evidence answer rate and success, on the questions both have, with an article bootstrap."""
    keys = lambda rows: {(r["qid"], r["variant"]) for r in rows}
    common = keys(rows_a) & keys(rows_b)
    qids = {q for q, v in common if (q, "open") in common and (q, "support_masked") in common}
    pick = lambda rows: [r for r in rows if r["qid"] in qids]
    ta, tb = cluster_table(pick(rows_a), article), cluster_table(pick(rows_b), article)
    names = sorted(ta)
    A, B = np.array([ta[n] for n in names]), np.array([tb[n] for n in names])
    observed = ratios(A.sum(axis=0)) - ratios(B.sum(axis=0))
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(names), (DRAWS, len(names)))
    diffs = np.array([ratios(A[p].sum(axis=0)) - ratios(B[p].sum(axis=0)) for p in picks])
    lo, hi = np.nanquantile(diffs, [0.025, 0.975], axis=0)
    labels = ("correct_and_cited", "hidden_answer_rate", "success")
    return {"questions": len(qids), "articles": len(names),
            **{m: {"difference": float(observed[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, m in enumerate(labels)}}


def script_check(rows, questions):
    """Hindi only: how many shown answers hold no Devanagari letter, next to the share of gold answers that hold none (names, numbers)."""
    answered = [r for r in rows if r["final"] == "answer" and r.get("answer") is not None]
    none = sum(not textnorm.has_script_letters(r["answer"], "hi") for r in answered)
    gold = [questions[r["qid"]].answer for r in rows if r["answerable"] and r["qid"] in questions]
    gold_none = sum(not textnorm.has_script_letters(a, "hi") for a in gold)
    return {"answers_shown": len(answered), "no_devanagari_letter": none, "share": none / len(answered) if answered else None,
            "gold_answers": len(gold), "gold_no_devanagari_letter": gold_none, "gold_share": gold_none / len(gold) if gold else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="reports/xquad_results")
    ap.add_argument("--langs", default="hi,en,es")
    args = ap.parse_args()
    root = ROOT / args.run
    langs = [l for l in args.langs.split(",")]
    cases = {(c["run"], c["qid"], c["variant"]): c for c in RS.read(root / "cases" / "answers.jsonl")}
    preds = verdicts(root / "verify" / "V" / "verdicts_A.jsonl")
    data, article, gen, audit, mism, scripts = {}, {}, {}, {}, {}, {}
    for lang in langs:
        manifest = json.loads((ROOT / "data" / "bench" / f"xq_{lang}" / "manifest.json").read_text(encoding="utf-8"))
        article.update(manifest["article_of"])
        _, questions = RS.world(f"bench:xq_{lang}")
        raw = {}
        for system in ("base", "main"):
            rows, mismatches, summary = load_run(root, system, lang)
            raw[system] = rows
            mism[f"{system}_{lang}"] = len(mismatches)
            gen[f"{system}_{lang}"] = summary.get("generation", {})
            audit[f"{system}_{lang}"] = summary.get("auditkit", {})
        built = {"base": raw["base"], "base+FB": old_select.fail_closed(raw["base"]), "main": raw["main"], "main+FB": old_select.fail_closed(raw["main"])}
        built[HEADLINE] = [checked(r, cases.get((f"main_bench_xq_{lang}", r["qid"], r["variant"])), preds.get((f"main_bench_xq_{lang}", r["qid"], r["variant"])))
                           for r in raw["main"]]
        built["always refuse"] = always_refuse(raw["main"])
        # eligibility comes from the corpus and the mask, never from a model: it must agree across systems
        for name, rows in built.items():
            if {(r["qid"], r["variant"], r["excluded_v3"]) for r in rows} != {(r["qid"], r["variant"], r["excluded_v3"]) for r in built["main"]}:
                raise ValueError(f"model-dependent eligibility in {lang}/{name}")
        data[lang] = {name: [r for r in rows if not r["excluded_v3"]] for name, rows in built.items()}
        if lang == "hi":
            scripts = {name: script_check(data[lang][name], questions) for name in ("base", "main", HEADLINE)}
    results = {l: {name: measures(rows) for name, rows in systems.items()} for l, systems in data.items()}
    comparisons = {}
    for lang in langs:
        if lang != "en" and "en" in langs:
            comparisons[f"{HEADLINE}: {NAMES[lang]} minus English"] = paired(data[lang][HEADLINE], data["en"][HEADLINE], article)
            comparisons[f"main: {NAMES[lang]} minus English"] = paired(data[lang]["main"], data["en"]["main"], article)
        comparisons[f"{NAMES[lang]}: {HEADLINE} minus base"] = paired(data[lang][HEADLINE], data[lang]["base"], article)
        comparisons[f"{NAMES[lang]}: main minus base"] = paired(data[lang]["main"], data[lang]["base"], article)
    truncated = {k: v.get("truncated_inputs") for k, v in gen.items()}
    report = {"scorer": "v3 with XQuAD normalization (research/2026-11-15-multilingual-plan.md)", "systems": results, "comparisons": comparisons,
              "stored_scores_differ_on_replay": mism, "truncated_prompts": truncated, "max_input_tokens": {k: v.get("max_input_tokens") for k, v in gen.items()},
              "hindi_answer_script": scripts, "auditkit_raw_systems": audit,
              "bootstrap": f"{DRAWS} paired draws over Wikipedia articles, seed {SEED}"}
    if any(mism.values()):
        print("WARNING: stored scores differ from the replay:", mism)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x: None) + "\n", encoding="utf-8")
    write_markdown(report, out.with_suffix(".md"), langs)
    print(out.with_suffix(".md").read_text(encoding="utf-8"))


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def write_markdown(report, path, langs):
    lines = ["# Multilingual check: Hindi, English and Spanish on XQuAD", "", report["bootstrap"] + ". Scorer v3 with the XQuAD normalization.", ""]
    for lang in langs:
        lines += [f"## {NAMES[lang]}", "", "| system | episodes | success | correct and cited | answers on hidden evidence | refuses when evidence is there | correct among answers shown | invalid |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, m in report["systems"][lang].items():
            lines.append(f"| {name} | {m['n']} | {pct(m['success'])} | {pct(m['correct_and_cited'])} | {m['hidden_answers']}/{m['n_hidden']} ({pct(m['hidden_answer_rate'])}) | "
                         f"{pct(m['false_refusal'])} | {pct(m['correct_among_answered'])} | {pct(m['invalid'])} |")
        lines.append("")
    lines += ["## Paired comparisons (difference, 95% interval)", "", "| comparison | questions | correct and cited | answers on hidden evidence | success |", "| --- | ---: | ---: | ---: | ---: |"]
    ci = lambda c: f"{100 * c['difference']:+.1f} pp [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]"
    for name, c in report["comparisons"].items():
        lines.append(f"| {name} | {c['questions']} | {ci(c['correct_and_cited'])} | {ci(c['hidden_answer_rate'])} | {ci(c['success'])} |")
    lines += ["", f"Prompts truncated by the input limit: {report['truncated_prompts']}", f"Stored scores that differ on replay: {report['stored_scores_differ_on_replay']}"]
    if report["hindi_answer_script"]:
        lines += ["", "Hindi answers with no Devanagari letter (names and numbers legitimately have none):", ""]
        for name, s in report["hindi_answer_script"].items():
            lines.append(f"- {name}: {s['no_devanagari_letter']} of {s['answers_shown']} answers shown ({pct(s['share'])}); gold answers: {s['gold_no_devanagari_letter']} of {s['gold_answers']} ({pct(s['gold_share'])})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required saved artifact is missing: {exc.filename or exc}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
