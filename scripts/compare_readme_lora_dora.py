"""README LoRA metrics versus all completed DoRA runs, preserving test scope."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/readme_lora_vs_all_dora"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows] + [""]


def pct(v):
    return f"{v*100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme-source", type=Path, default=ROOT / "README.md",
                        help="optional publication copy when unrelated README edits remain unstaged")
    args = parser.parse_args()
    hashes = {}

    def load(name):
        path = ROOT / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8-sig"))

    hashes["README.md"] = hashlib.sha256(args.readme_source.read_bytes()).hexdigest()
    comparison = load("reports/evidence_order_all_previous_comparison.json")
    rows = comparison["benchmark_rows"]
    lora_names = ("Main LoRA", "candA LoRA", "candF1 LoRA")
    dora_names = ("Modal DoRA", "Kaggle DoRA lr5e-5", "Evidence-order DoRA")
    lookup = {(r["model"], r["configuration"]): r for r in rows}
    lines = ["# README LoRA results versus all completed DoRA runs", "",
             "Prepared 14 November 2026 from local completed records. No new training or inference.", "",
             "**Original Modal DoRA improves on raw main LoRA, but no DoRA checkpoint beats the "
             "best recorded LoRA configuration.** Kaggle lr5e-5 and the latest evidence-order continuation "
             "are below main, candA and candF1 on the shared benchmark macro. The comparison must use "
             "the same test suite and the same output-repair setting.", "",
             "## Match the README's result to the correct test", ""]
    lines += table(["README result", "Original population/metric", "Matching DoRA evaluation?"], [
        ["First-round main 58.3%, control 40.4%", "151 headline HotpotQA episodes, out of 287 total", "No"],
        ["Fresh test2 main 60.1%, candA 65.7%, control 28.0%", "143 headline HotpotQA episodes, out of 273 total", "No"],
        ["External main 49.9%, candA 54.7%, control 29.3%", "1,131 pooled episodes, five older benchmark suites", "No"],
        ["Round-3 main 48.9%, candD 55.5%, candA 54.9%", "739-episode fresh suite, macro of six sets", "No"],
        ["Round-5 main+FB+V 51.7%, candF1+FB+V 57.0%", "938 inputs; four-component macro after scorer-v2 exclusions", "Yes, all three DoRA runs"],
    ])
    lines += ["A 58.3% HotpotQA headline score cannot be subtracted from a 27.82% four-component "
              "macro to estimate a DoRA effect. Even SQuAD/RGB names recur across different question "
              "samples, variants and scoring versions. Control and the round-3 patch lack matching "
              "round-5 rows; missing results are not zero scores.", "",
              "## Direct comparison on the same 938 input episodes", "",
              "FB = fail-closed finalization. V = the fixed main-LoRA verifier with wording A, threshold 0.5. "
              "The repaired DoRA systems use that LoRA verifier; they are not evaluations of a DoRA-trained verifier.", ""]
    lines += table(["Model", "Raw macro ↑", "+FB macro ↑", "+FB+V macro ↑"],
                   [[name, *[pct(lookup[name, config]["macro"]) for config in ("raw", "FB", "FB+V")]]
                    for name in (*lora_names, *dora_names)])
    lines += ["### Each DoRA checkpoint minus each previous LoRA checkpoint", "",
              "Macro differences below are percentage points, DoRA minus LoRA. Positive favors DoRA. "
              "Compare raw with raw and repaired with the same repair configuration.", ""]
    deltas = []
    for dora in dora_names:
        for lora in lora_names:
            values = [lookup[dora, c]["macro"] - lookup[lora, c]["macro"] for c in ("raw", "FB", "FB+V")]
            deltas.append({"dora": dora, "lora_reference": lora, "raw_delta_pp": values[0]*100,
                           "fb_delta_pp": values[1]*100, "fb_v_delta_pp": values[2]*100})
    lines += table(["DoRA checkpoint", "LoRA reference", "Raw Δ pp", "+FB Δ pp", "+FB+V Δ pp"],
                   [[r["dora"], r["lora_reference"], *[f"{r[k]:+.2f}" for k in ("raw_delta_pp", "fb_delta_pp", "fb_v_delta_pp")]] for r in deltas])
    lines += ["### Answer quality and missing-evidence behavior", ""]
    for config in ("raw", "FB+V"):
        lines += [f"**{config}**", ""]
        lines += table(["Model", "Supported correct / answerable ↑", "Unsupported / hidden ↓", "False abstain / answerable ↓"],
                       [[name, f"{round(lookup[name,config]['supported_correct']*563)}/563 ({pct(lookup[name,config]['supported_correct'])})",
                         f"{lookup[name,config]['unsupported']}/{lookup[name,config]['hidden_n']} ({pct(lookup[name,config]['unsupported_rate'])})",
                         f"{round(lookup[name,config]['false_abstain']*563)}/563 ({pct(lookup[name,config]['false_abstain'])})"]
                        for name in (*lora_names, *dora_names)])
    lines += ["On raw outcomes, original Modal DoRA exceeds main LoRA by **3.28 pp macro**, "
              "while unsupported answers increase from **108/364 to 117/364**. With FB+V, "
              "Modal DoRA is **1.65 pp below main**, with **64/364** unsupported answers versus main's **37/364**. "
              "The later DoRA runs move further toward answering: the latest answers **306/374** hidden-evidence "
              "episodes. It is not a better operating point for the repository's evidence-grounding objective.", "",
              "### Per-component metrics, raw and repaired", ""]
    lines += table(["Model", "Setting", "SQuAD single", "SQuAD two-part", "HotpotQA", "RGB"],
                   [[name, config, *[pct(lookup[name,config][k]) for k in ("single", "compound", "hotpot", "rgb")]]
                    for config in ("raw", "FB+V") for name in (*lora_names, *dora_names)])
    lines += ["## The README's original LoRA metrics, retained as a separate panel", "",
              "These first-round HotpotQA metrics are quoted from the machine-readable source linked by "
              "the README. They are useful history, but no completed DoRA evaluation on this same split "
              "has been downloaded. The smaller cohort and older audit adapter prevent a method ranking "
              "against the newer round-5 AuditKIT means.", ""]
    first = load("reports/scorecard_test_lenient.json")
    specs = [
        ("Supported correct answer ↑", "grounding", "supported_correct (answerable episodes)"),
        ("Unsupported answer ↓", "abstention (documents-only requirement)", "unsupported_answer_rate (unanswerable; lower is better)"),
        ("False abstain ↓", "abstention (documents-only requirement)", "false_abstain (answerable; lower is better)"),
        ("AuditKIT evidence-set recall ↑", "what_to_retrieve", "auditkit_evidence_set_recall"),
        ("AuditKIT citation support ↑", "grounding", "auditkit_citation_support"),
        ("Answer F1 when answered ↑", "grounding", "answer_f1_when_answered"),
        ("Tool-call validity ↑", "tool_call_reliability (AuditKIT, Track 3)", "tool_call_validity"),
        ("Loop-free ↑", "tool_call_reliability (AuditKIT, Track 3)", "agent_loop_detection (1.0 = loop-free)"),
        ("Redundant calls ↓", "tool_call_reliability (AuditKIT, Track 3)", "redundant_tool_calls (lower is better)"),
    ]
    entries = []
    for title, group, metric in specs:
        values = []
        for name in ("main", "control"):
            value = first["systems"][name][group][metric]
            values.append(f"{value['value']:.3f} ({value['k']}/{value['n']})" if isinstance(value, dict) else f"{value:.3f}")
        entries.append([title, *values])
    lines += table(["Original HotpotQA metric", "Main LoRA", "Control LoRA"], entries)
    lines += ["For orientation only, the same-named metrics from the three raw DoRA evaluations are "
              "shown below with their actual cohorts. This is a **different test and audit population**; "
              "no cross-panel delta is computed.", ""]
    audit = comparison["auditkit_snapshots"]
    mapping = [("evidence_set_recall", "rag_stress"), ("citation_support", "rag_stress"),
               ("f1_score", "answerable_answered"), ("tool_call_validity", "agent"),
               ("agent_loop_detection", "agent"), ("redundant_tool_calls", "agent")]
    lines += table(["Round-5 AuditKIT metric", *dora_names],
                   [[metric, *[f"{audit[name]['configurations']['dora']['cohorts'][cohort]['stats'][metric]['mean']:.4f} "
                               f"(n={audit[name]['configurations']['dora']['cohorts'][cohort]['stats'][metric]['count']})"
                               for name in dora_names]] for metric, cohort in mapping])
    external = load("reports/benchmarks_lenient.json")
    lines += ["### README five-external-benchmark panel", ""]
    entries = []
    for suite, data in external["benchmarks"].items():
        entries.append([suite, *[f"{data['metrics'][name]['ends well (all scored episodes)']['value']:.3f}" for name in ("main", "candA", "control")], "Not measured"])
    entries.append(["All five pooled (1,131)", *[f"{external['pooled']['ends_well'][name]['value']:.3f}" for name in ("main", "candA", "control")], "Not measured"])
    lines += table(["Older external benchmark", "Main LoRA", "candA LoRA", "Control LoRA", "All three DoRA"], entries)
    lines += ["The old pooled 0.499/0.547 scores are weighted by those five suites' episode counts. "
              "The current macro weights four round-5 components equally. Their numerical values cannot "
              "be used to conclude LoRA or DoRA won on the old external suite.", "",
              "## What the available results establish", "",
              "- **Best raw adapter macro on the shared suite:** candA LoRA, 47.34%.",
              "- **Best repaired adapter macro:** candF1 LoRA+FB+V, 56.97%; it failed the original "
              "unsupported-answer non-inferiority rule, so it remains a published alternative.",
              "- **Historical selected system:** main LoRA+FB+V, 51.74%, under the registered selection rule.",
              "- **Best of your DoRA runs:** original Modal DoRA, 44.71% raw and 50.09% with FB+V.",
              "- **Latest evidence-order DoRA:** 27.82% raw and 45.41% repaired; its increased answer rate "
              "does not compensate for answering without evidence.", "",
              "These are checkpoint comparisons, not proof that LoRA is inherently superior to DoRA. "
              "Data, continuation, LR, runtime/hardware and optimization histories vary. Even original "
              "main LoRA versus Modal DoRA was not a fully controlled multi-seed method study; equal "
              "direction rank also leaves DoRA with extra magnitude parameters.", "",
              "All direct rows use the same 938 input episodes, but retained counts and hidden-evidence "
              "denominators vary by checkpoint (scorer-v2 exclusions). Rankings and deltas are descriptive; "
              "this report adds no paired intervals or common-intersection rescore. "
              "Control and patch require new, matching evaluations to enter the current suite's ranking.", "",
              "## Sources and downloadable tables", "",
              "[README](../README.md) · [Original LoRA scorecard](scorecard_test_lenient.md) · "
              "[External LoRA report](benchmarks_lenient.md) · [Round-5 LoRA report](round5b_final.md) · "
              "[Full latest study](evidence_order_all_previous_comparison.md).", "",
              "[DoRA-minus-LoRA macro deltas CSV](readme_lora_vs_all_dora.csv) · "
              "[Full-precision source snapshot](readme_lora_vs_all_dora.json).", ""]
    with DEST.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    DEST.with_suffix(".json").write_text(json.dumps({"report_date": "2026-11-14", "source_sha256": hashes,
        "matching_suite": "round-5, 938 input episodes", "dora_minus_lora_macro_pp": deltas,
        "matching_benchmark_rows": [r for r in rows if r["model"] != "Base"],
        "original_readme_scorecard": first, "original_external_benchmarks": external,
        "cross_suite_deltas_computed": False}, indent=2), encoding="utf-8")
    DEST.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {DEST.name}: 9 DoRA/LoRA pairs, 27 macro deltas, README metrics with original cohorts")


if __name__ == "__main__":
    main()
