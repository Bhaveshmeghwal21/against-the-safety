"""Compare the completed evidence-order run with preserved historical results.

No model calls. Preserves populations and keeps incompatible tests separate.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "runs/kaggle/evaluation/evidence-order-v369426017/dora-evaluation"
PARENT = ROOT / "runs/kaggle/evaluation/dora_lr5e5-evaluation-fixed"
DEST = ROOT / "reports/evidence_order_all_previous_comparison"
CONFIGS = ("raw", "FB", "FB+V")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, data):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(str(v) for v in row) + " |" for row in data] + [""]


def pct(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def pp(value):
    return f"{value * 100:+.2f}"


def source_link(path):
    import os
    if path.is_relative_to(LATEST):
        return (Path("evidence_order_evaluation") / path.relative_to(LATEST)).as_posix()
    return Path(os.path.relpath(path, DEST.parent)).as_posix()


def main():
    sources = {}

    def load(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return read(path)

    preserved = load(ROOT / "reports/dora_evaluation_data.json")
    history = load(ROOT / "reports/round5b_final.json")
    earlier_r5 = load(ROOT / "reports/round5_final.json")
    assert all(history["systems"][k] == v for k, v in earlier_r5["systems"].items())
    latest = load(LATEST / "reports/computed_round5.json")
    parent = load(PARENT / "reports/computed_round5.json")
    modal = preserved["dora_benchmark"]
    assert history["episodes"] == latest["episodes"] == parent["episodes"] == modal["episodes"] == 938
    assert load(LATEST / "COMPLETE.json")["scorer"] == "v2"
    load(LATEST / "auditkit/AUDIT_COMPLETE.json")
    audits = [preserved["auditkit"], load(PARENT / "auditkit/auditkit_metrics.json"),
              load(LATEST / "auditkit/auditkit_metrics.json")]
    assert len({a["auditkit_version"] for a in audits}) == 1
    assert all(a["adapter_contract"] == "recorded-v2" and a["input_episodes"] == 938 for a in audits)
    assert all(a["metric_source_sha256"] == audits[0]["metric_source_sha256"] for a in audits)
    assert all(a["model_identity"]["checkpoints"] == audits[0]["model_identity"]["checkpoints"] for a in audits)
    assert all(a["model_identity"]["checker"] == "main/A/0.5" for a in audits)
    labels = ["Modal DoRA", "Kaggle DoRA lr5e-5", "Evidence-order DoRA"]
    families = [("Base", history, "base"), ("Main LoRA", history, "main"), ("candA LoRA", history, "candA"),
                ("candF1 LoRA", history, "candF1"), (labels[0], modal, "dora"),
                (labels[1], parent, "dora"), (labels[2], latest, "dora")]
    rows = []
    snapshots = {}
    for label, report, key in families:
        snapshots[label] = report
        for config, suffix in zip(CONFIGS, ("", "+FB", "+FB+V")):
            if key + suffix not in report["systems"]:
                continue
            s = report["systems"][key + suffix]
            assert abs(s["macro"] - sum(s["components"].values()) / len(s["components"])) < 1e-12
            assert abs(s["unsupported_rate"] - s["unsupported"] / s["hidden_n"]) < 1e-12
            rows.append({"model": label, "configuration": config, "input_n": 938,
                         **{k: s[k] for k in ("macro", "n", "hidden_n", "unsupported", "unsupported_rate",
                                             "supported_correct", "false_abstain", "invalid_before_repair")},
                         "answerable_n": s["n"] - s["hidden_n"], **s["components"]})
    with DEST.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Evidence-order DoRA compared with all previous local benchmark records", "",
             "Prepared 14 November 2026. Source: completed Kaggle evaluation `mlsingh1234/evaluation/v5`, "
             "script version `369426017`. This study performs no new inference or training.", "",
             "**The latest approach regresses on overall benchmark success.** Its raw macro is "
             f"{pct(latest['systems']['dora']['macro'])}, versus {pct(parent['systems']['dora']['macro'])} "
             f"for its Kaggle parent and {pct(modal['systems']['dora']['macro'])} for the original Modal DoRA. "
             "It answers more supported questions, but also answers substantially more questions whose evidence is hidden. "
             "The fixed verifier helps, but does not reverse the overall regression.", "",
             "## Scope and comparability", "",
             "The primary comparison covers every locally recorded adapter family on the same 938-input round-5 suite: "
             "base, main, candA, candF1, original Modal DoRA, Kaggle lr5e-5 DoRA, and the latest evidence-order DoRA. "
             "Round 5 and 5b duplicate the same base/main/candA rows; those rows are counted once. "
             "Hosted-model intersections and earlier test/development suites are included separately below. "
             "The latest checkpoint has not been evaluated on those older suites, so they cannot establish its rank there.", "",
             "All three DoRA runs share the pinned base/reference-adapter hashes, fixed main/A/0.5 verifier, "
             "AuditKIT 1.1.3 metric implementation hashes, and recorded-v2 audit contract. "
             "The audit wrapper/source hashes and offline scoring packaging differ across old and new runs; "
             "this is a descriptive comparison, not a claim of identical execution. "
             "Scorer-v2 exclusions vary by checkpoint. No common-intersection recomputation, paired confidence "
             "interval, or significance test is claimed. The test suite was already public when the DoRA experiments ran.", "",
             "The historical Modal audit JSON path was previously overwritten with prose; the full-precision "
             "[preserved snapshot](dora_evaluation_data.json) supplies its audit values. "
             "The lr1e-4 adapter has training metadata but no separate benchmark evaluation available locally.", "",
             "## All comparable round-5 model families", "",
             "Macro is an equal mean of SQuAD single, SQuAD two-part, HotpotQA and RGB success. "
             "FB converts invalid endings to abstentions; V applies the saved main-adapter verifier. "
             "These are inference repairs of one generator's traces, not separate trained models.", ""]
    macro_rows = []
    for label, report, key in families:
        raw = report["systems"][key]
        macro_rows.append([label, *[pct(report["systems"].get(key + suffix, {}).get("macro"))
                                   for suffix in ("", "+FB", "+FB+V")], raw["n"], raw["hidden_n"]])
    lines += table(["Model", "Raw macro ↑", "+FB ↑", "+FB+V ↑", "Retained n", "Hidden n"], macro_rows)
    raw_rank = sorted([(r["macro"], r["model"]) for r in rows if r["configuration"] == "raw"], reverse=True)
    repaired_rank = sorted([(r["macro"], r["model"]) for r in rows if r["configuration"] == "FB+V"], reverse=True)
    rank_raw = next(i + 1 for i, (_, name) in enumerate(raw_rank) if name == labels[2])
    rank_repaired = next(i + 1 for i, (_, name) in enumerate(repaired_rank) if name == labels[2])
    lines += [f"The latest model is descriptively **{rank_raw}/{len(raw_rank)} raw**, and "
              f"**{rank_repaired}/{len(repaired_rank)} with FB+V**, among these families with recorded configurations. "
              "candA has the strongest raw adapter score; candF1 has the strongest repaired adapter score. "
              "The historical selected submission remains main+FB+V: candF1 failed the preregistered "
              "unsupported-answer non-inferiority rule despite its higher macro. See [round5b_final.md](round5b_final.md).", "",
              "## Per-component results for every compatible configuration", ""]
    lines += table(["Model", "Configuration", "SQuAD single", "SQuAD two-part", "HotpotQA", "RGB"],
                   [[r["model"], r["configuration"], *[pct(r[k]) for k in ("single", "compound", "hotpot", "rgb")]] for r in rows])
    lines += ["## Direct DoRA progression and deltas", "",
              "Differences are latest minus the named reference in percentage points (pp), not relative percent change.", ""]
    delta_rows = []
    for reference, report in ((labels[0], modal), (labels[1], parent)):
        for config in latest["systems"]:
            a, b = report["systems"][config], latest["systems"][config]
            delta_rows.append([reference, config, pp(b["macro"] - a["macro"]),
                               pp(b["supported_correct"] - a["supported_correct"]),
                               pp(b["unsupported_rate"] - a["unsupported_rate"]),
                               pp(b["false_abstain"] - a["false_abstain"])])
    lines += table(["Reference", "Configuration", "Macro Δ ↑", "Supported correct Δ ↑", "Unsupported Δ ↓", "False abstain Δ ↓"], delta_rows)
    lines += table(["Model", "Configuration", "Supported correct / answerable", "Unsupported / hidden", "False abstain / answerable", "Raw invalid / input"],
                   [[r["model"], r["configuration"], f"{round(r['supported_correct']*r['answerable_n'])}/{r['answerable_n']} ({pct(r['supported_correct'])})",
                     f"{r['unsupported']}/{r['hidden_n']} ({pct(r['unsupported_rate'])})",
                     f"{round(r['false_abstain']*r['answerable_n'])}/{r['answerable_n']} ({pct(r['false_abstain'])})",
                     f"{round(r['invalid_before_repair']*938)}/938 ({pct(r['invalid_before_repair'])})"]
                    for r in rows if r["model"] in labels])
    lines += ["`invalid_before_repair` records the generator's original invalid endings and therefore repeats "
              "across its repaired configurations. It must not be read as invalid outputs remaining after FB. "
              "Answerable denominators are 563 for all three DoRA runs; hidden denominators differ.", "",
              "The latest raw model gets **192/563** supported correct answers, up from **174/563** in its parent "
              "and **176/563** in Modal. It falsely abstains on only **18/563**, versus **69/563** and **112/563**. "
              "However, hidden-evidence answers rise to **306/374 (81.82%)**, from **227/371 (61.19%)** and "
              "**117/364 (32.14%)**. The model has shifted toward answering regardless of available evidence. "
              "This is an interpretation of observed behavior, not a proven causal explanation of the training change.", "",
              "With FB+V, the latest macro rises from **27.82% to 45.41%** (+17.58 pp), and unsupported answers "
              "fall from **306 to 126** on the same 374 hidden episodes. The cost is supported correct answers "
              "falling from **192 to 179** and false abstentions rising from **18 to 139**. Its repaired unsupported "
              "rate is still **33.69%**, versus **24.53%** for its parent's repaired configuration and **17.58%** for Modal.", "",
              "## All 19 recorded AuditKIT outputs across the three DoRA checkpoints", "",
              "Values remain on the original 0–1 scale; parentheses show each metric's contributing n. "
              "Higher is generally better except redundant tool calls. Lexical overlap and conditional "
              "metrics are proxies and must be interpreted with their cohorts.", ""]
    audit_rows = []
    for config in latest["systems"]:
        lines += [f"### {config}", ""]
        entries = []
        for cohort, data in audits[0]["configurations"][config]["cohorts"].items():
            for metric in data["stats"]:
                stats = [a["configurations"][config]["cohorts"][cohort]["stats"][metric] for a in audits]
                entries.append([f"{cohort}: {metric}", *[f"{s['mean']:.4f} (n={s['count']})" for s in stats],
                                f"{stats[2]['mean']-stats[1]['mean']:+.4f}"])
                audit_rows.append({"configuration": config, "cohort": cohort, "metric": metric,
                                   **{f"{label}_{field}": stat[field] for label, stat in zip(("modal", "parent", "latest"), stats) for field in ("mean", "count")},
                                   "latest_minus_parent": stats[2]["mean"] - stats[1]["mean"]})
        assert len(entries) == 19
        lines += table(["Metric/cohort", *labels, "Latest − parent"], entries)
    with DEST.with_name(DEST.name + "_auditkit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    lines += ["The latest raw evidence-set recall is **0.8028**, below its parent's **0.8970** and Modal's "
              "**0.9449**, all on 563 answerable episodes. Its context coverage also falls to **0.8773**. "
              "First-search hit@3 rises to **0.9810**, but only **211** answerable episodes contribute, versus "
              "304 in the parent and 377 in Modal; it does not establish improved overall retrieval. "
              "Conditional answer F1 falls to **0.6224** on **544** answered cases, from **0.6725** on 485 "
              "and **0.7607** on 433. The cohorts expand as the model answers more often.", "",
              "Citation support (**0.9663**) and lexical groundedness (**0.9768**) are high, and loop-free "
              "behavior (**0.9947**) improves. These measures do not verify that all facts needed to answer "
              "the question were present or that the answer was correct. They therefore do not cancel "
              "the benchmark's evidence-missing regressions.", "",
              "## Hosted-model historical comparisons: different common intersections", "",
              "These published tables compare their original actors on a shared intersection. The latest DoRA "
              "was not included when those intersections were constructed. Do not insert its 937-retained "
              "score into either ranking or infer paired superiority from the numbers below.", ""]
    hosted = {}
    for name in ("azure_round5_comparison", "deepseek_comparison"):
        path = ROOT / f"reports/{name}.json"
        data = load(path)
        hosted[name] = data
        lines += [f"### [{name}]({path.name.replace('.json', '.md')})", "",
                  f"Recorded scope: {data['scope']}; common n={data['coverage']['common']}.", ""]
        lines += table(["System / prompt protocol", "Macro", "Supported correct", "Unsupported / hidden", "n"],
                       [[key, pct(s["macro"]), pct(s["supported_correct"]), f"{s['unsupported']}/{s['hidden_n']}", s["n"]]
                        for key, s in data["systems"].items()])
    lines += ["Provider model names, exact/native protocol settings, availability failures and cost assumptions "
              "are documented in the linked original reports. This study uses those recorded values without "
              "new provider calls or claims about current hosted models.", "",
              "## Earlier test suites: historical context, no latest-checkpoint measurement", ""]
    historical = {}
    for name in ("round3_test", "round3_test_candA", "round4_test", "round4_wrappers", "round5_r4test_wrappers"):
        data = load(ROOT / f"reports/{name}.json")
        historical[name] = data
        lines += [f"### [{name}]({name}.md)", ""]
        if name.startswith("round3"):
            lines += table(["System", "Original three-component macro"], [[k, pct(v)] for k, v in data["macro"].items()])
        else:
            lines += table(["System", "Original macro", "Unsupported / hidden", "Retained n"],
                           [[k, pct(v["macro"]), f"{v['unsupported']}/{v['hidden_n']}", v["n"]] for k, v in data["systems"].items()])
        lines += ["Latest evidence-order DoRA: **not measured on this suite**. Dataset composition and/or scoring "
                  "differ from the round-5 headline; do not compare these macros as repeated measurements of one test.", ""]
    # Include every external benchmark in both parser variants, preserving k/n.
    for name in ("benchmarks_lenient", "benchmarks_strict"):
        data = load(ROOT / f"reports/{name}.json")
        historical[name] = data
        lines += [f"### [{name}]({name}.md)", ""]
        entries = []
        for suite, block in data["benchmarks"].items():
            for system, metrics in block["metrics"].items():
                result = metrics.get("ends well (all scored episodes)")
                if result:
                    entries.append([suite, system, pct(result["value"]), f"{result['k']}/{result['n']}"])
        lines += table(["Earlier external suite", "System", "Original success", "k/n"], entries)
        lines += ["Latest checkpoint: not evaluated here. Strict and lenient parser results are not interchangeable.", ""]
    lines += ["### Earlier task scorecards", ""]
    scorecard_entries = []
    for name in ("scorecard_dev_lenient", "scorecard_test_lenient", "scorecard_test_strict", "scorecard_test2_lenient", "scorecard_test2_strict"):
        data = load(ROOT / f"reports/{name}.json")
        historical[name] = data
        for system, metrics in data["systems"].items():
            supported = metrics["grounding"]["supported_correct (answerable episodes)"]
            unsupported = metrics["abstention (documents-only requirement)"]["unsupported_answer_rate (unanswerable; lower is better)"]
            scorecard_entries.append([f"[{name}]({name}.md)", system,
                                      f"{pct(supported['value'])} ({supported['k']}/{supported['n']})",
                                      f"{pct(unsupported['value'])} ({unsupported['k']}/{unsupported['n']})"])
    lines += table(["Original split/parser", "System", "Supported correct", "Unsupported"], scorecard_entries)
    lines += ["These original scorecards cover base/main/control/candA where present. Their evaluation questions, "
              "variant populations and parsers differ. They are retained as history rather than pooled with the latest test.", "",
              "### Development and selection records", "",
              "Development results must remain separate from final-test rankings. The following records preserve "
              "earlier selection decisions, failed DPO variants, wrapper checks and specialty tests:", ""]
    inventory = []
    for path in sorted((ROOT / "reports").glob("*.json")):
        if any(token in path.stem for token in ("selection", "verifier", "round4_brightpro", "gate_baseline", "retrieval_ceiling", "audit_slice", "round4_dev", "round3_test")):
            data = load(path)
            historical[path.stem] = data
            decision = data.get("selected", data.get("winner", data.get("recommended_default", data.get("headline", "See linked record"))))
            inventory.append([f"[{path.name}]({path.name})", str(decision).replace("|", "/")[:180]])
    lines += table(["Historical record", "Recorded selection/description"], inventory)
    lines += ["The latest training notebook's `DEV_COMPARISON.md`, `dev_comparison.json`, and curriculum/training "
              "reports are **not present in this downloaded evaluation output**. Therefore this study cannot verify "
              "whether the development screen passed, the actual retained curriculum size, or the completed "
              "continuation's LR/epochs/steps. The code defaults (2e-5, one epoch) describe the intended recipe, "
              "not independently verified training metadata. The evaluated checkpoint identity itself is confirmed "
              "by evaluation and audit completion records.", "",
              "## Interpretation and next decision", "",
              "1. **Keep the earlier stronger checkpoint for use.** Original Modal DoRA is strongest among the "
              "three DoRA versions by raw and repaired macro. Across historical adapters, main+FB+V remains "
              "the recorded selection under the original evidence-safety rule.",
              "2. **Treat this latest continuation as a failed overall experiment.** It improves answer rate, "
              "format validity and some citation/tool proxies, but fails the central requirement to withhold "
              "answers when evidence is unavailable. Low false-abstention alone is not a successful result.",
              "3. **Inspect the training and DEV artifacts before spending more GPU time.** Verify the real "
              "action mixture, three-search abstention eligibility, paired/replay token shares, parent hashes, "
              "and the same-episode development screen. The current evidence is insufficient to identify "
              "whether sampling, extra optimization, passage order, or another change caused the regression.",
              "4. **Use controlled development ablations for any follow-up.** Compare equal-budget continuation "
              "with replay only against balanced sampling plus permutation, then choose on held-out development "
              "behavior. Fix the choice before a final benchmark; repeated tuning on these published test "
              "results does not yield a fresh generalization estimate.", "",
              "## Data and provenance", "",
              f"Latest checkpoint: `{audits[2]['model_identity']['dora_checkpoint']['path']}`.", "",
              f"Adapter weights SHA-256: `{audits[2]['model_identity']['dora_checkpoint']['sha256']['adapter_model.safetensors']}`.", "",
              f"[Latest evaluation summary]({source_link(LATEST / 'EVALUATION_SUMMARY.md')}) · "
              f"[Latest full AuditKIT data]({source_link(LATEST / 'auditkit/auditkit_metrics.json')}) · "
              "[Original Modal report](dora_evaluation_report.md) · "
              "[Previous DoRA comparison](dora_kaggle_lr5e5_comparison.md).", "",
              "[Benchmark CSV](evidence_order_all_previous_comparison.csv), "
              "[AuditKIT CSV](evidence_order_all_previous_comparison_auditkit.csv), and "
              "[full precision source snapshots and hashes](evidence_order_all_previous_comparison.json) "
              "support reproducible inspection. All benchmark rate deltas use full-precision inputs.", ""]
    training_data_path = ROOT / "reports/evidence_order_training_data.json"
    training_data = load(training_data_path) if training_data_path.is_file() else None
    if training_data:
        assert training_data["completion"]["adapter_sha256"] == audits[2]["model_identity"]["dora_checkpoint"]["sha256"]
        for i, paragraph in enumerate(lines):
            if paragraph.startswith("The latest training notebook's"):
                lines[i] = ("The training and development metadata have now been downloaded and matched to the evaluated "
                            "adapter hash. Full training used LR 2e-5, one epoch, 541 rows, 34 steps, and 646.8 seconds "
                            "on four L4s. The development screen **failed**: unsupported answers rose from 17/73 "
                            "to 39/73, and evidence-supported good outcomes fell from 64/159 to 54/159. "
                            "See the [verified training snapshot](evidence_order_training_data.json) and "
                            "[cost/efficiency report](dora_lora_cost_efficiency.md). This updates the earlier "
                            "report's missing-metadata limitation; the final benchmark remains a negative result.")
    snapshot = {"report_date": "2026-11-14", "scope": "all locally preserved comparable and historical benchmark records",
                "sources_sha256": sources, "paired_intervals_computed": False, "common_intersection_computed": False,
                "same_dora_base_and_verifier_hashes": True, "same_auditkit_metric_implementation_hashes": True,
                "benchmark_rows": rows, "auditkit_rows": audit_rows, "benchmarks": snapshots,
                "auditkit_snapshots": dict(zip(labels, audits)), "hosted_comparisons": hosted,
                "historical_records": historical, "latest_training_metadata_verified": bool(training_data),
                "latest_training_metadata": training_data}
    DEST.with_suffix(".json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    # Optional local scientific plot; do not download dependencies.
    cache = ROOT / ".cache/report-plot-packages"
    if cache.is_dir():
        sys.path.insert(0, str(cache))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        chart_models = [label for label, _, _ in families if label != "Base"]
        x = np.arange(len(chart_models))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
        for offset, config, color in ((-.19, "raw", "#387bb5"), (.19, "FB+V", "#db8434")):
            selected = [next(r for r in rows if r["model"] == model and r["configuration"] == config) for model in chart_models]
            for ax, metric in zip(axes, ("macro", "unsupported_rate")):
                bars = ax.bar(x + offset, [r[metric] * 100 for r in selected], .38, label=config, color=color)
                ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        for ax, title in zip(axes, ("Macro success (higher is better)", "Unsupported answers / hidden (lower is better)")):
            ax.set_title(title)
            ax.set_xticks(x, chart_models, rotation=28, ha="right", fontsize=8)
            ax.set_ylabel("Percent")
            ax.legend()
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle("Published round-5 results: checkpoint-specific scorer exclusions")
        fig.text(.5, .01, "Descriptive comparison; different retained hidden cohorts; no confidence intervals.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .94))
        chart = ROOT / "docs/figures/evidence_order_all_previous_comparison.png"
        chart.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chart, dpi=180)
        fig.savefig(chart.with_suffix(".svg"))
        plt.close(fig)
        index = lines.index("## Per-component results for every compatible configuration")
        lines[index:index] = ["![Macro success and unsupported-answer trade-off](../docs/figures/evidence_order_all_previous_comparison.png)", ""]
    except (ImportError, OSError) as exc:
        print(f"Plot skipped: {exc}")
    DEST.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {DEST.name}: {len(rows)} benchmark configurations, {len(audit_rows)} AuditKIT metrics, {len(historical)} historical records")


if __name__ == "__main__":
    main()
