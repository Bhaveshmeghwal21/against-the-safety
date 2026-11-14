"""Publish measured resource use and explicitly historical GPU-price scenarios.

Local raw artifacts are optional once the compact source snapshots are published.
No current price lookup, cloud invoice, model inference or GPU launch is performed.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEST = REPORTS / "dora_lora_cost_efficiency"
TRAIN = ROOT / "runs/kaggle/evidence-training-metadata/ats-evidence/dora_evidence_order"
EVAL = ROOT / "runs/kaggle/evaluation/evidence-order-v369426017/dora-evaluation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, data):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in data] + [""]


def main():
    snapshot_path = REPORTS / "evidence_order_training_data.json"
    if TRAIN.is_dir():
        names = {"training": "dora/train_report.json", "curriculum": "data/data_report.json",
                 "development": "dev_comparison.json", "completion": "TRAIN_COMPLETE.json",
                 "identity": "experiment_identity.json"}
        data = {key: read(TRAIN / name) for key, name in names.items()}
        data["sources_sha256"] = {f"runs/kaggle/evidence-training-metadata/ats-evidence/dora_evidence_order/{name}": sha(TRAIN / name) for name in names.values()}
        data["source_notebook"] = "mlsingh1234/tranining-2"
        identity = read(EVAL / "evaluation_identity.json")
        assert data["completion"]["adapter_sha256"] == identity["dora_checkpoint"]["sha256"]
        assert data["training"]["data_sha256"] == data["curriculum"]["output_sha256"]
        assert not data["completion"]["smoke"]
        data["matches_evaluated_adapter"] = True
        snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data = read(snapshot_path)
    original = read(REPORTS / "dora_evaluation_data.json")
    comparison = read(REPORTS / "dora_kaggle_lr5e5_comparison.json")
    all_results = read(REPORTS / "evidence_order_all_previous_comparison.json")
    lora = read(REPORTS / "azure_roi.json")
    training = [original["training"], comparison["new_training"], data["training"]]
    names = ["Original Modal DoRA", "Kaggle DoRA lr5e-5", "Evidence-order DoRA continuation"]
    worlds = [1, 4, 4]
    rate = original["dora_gpu_cost"]["l4_gpu_usd_per_hour"]
    assert rate == .7992
    train_rows = []
    for name, report, world in zip(names, training, worlds):
        seconds = report["wall_seconds"]
        hours = seconds * world / 3600
        tokens = report["processed_tokens"] * report["args"]["epochs"]
        train_rows.append({"model": name, "gpu_count": world, "wall_seconds": seconds,
                           "allocated_gpu_hours": hours, "processed_tokens_all_epochs": tokens,
                           "processed_tokens_per_wall_second": tokens / seconds,
                           "processed_tokens_per_gpu_second": tokens / (seconds * world),
                           "rows": report["rows_trained"], "steps": report["optimizer_steps"],
                           "historical_l4_gpu_only_usd_scenario": hours * rate})
    inference_path = REPORTS / "evidence_order_inference_resources.json"
    if EVAL.is_dir():
        total = {"episodes": 0, "actor_seconds": 0., "input_tokens": 0, "output_tokens": 0, "model_calls": 0}
        source_hashes, suites = {}, []
        for suite in ("squad", "hotpot", "rgb"):
            path = EVAL / f"eval/dora_bench_r5t_{suite}/lenient/scores.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            counts = {"episodes": len(rows), "actor_seconds": sum(r["seconds"] for r in rows),
                      **{key: sum(r[key] for r in rows) for key in ("input_tokens", "output_tokens", "model_calls")}}
            for key in total:
                total[key] += counts[key]
            suites.append({"suite": suite, **counts})
            source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        verifier_seconds, verifier_cases = 0., 0
        for path in sorted((EVAL / "verifier").glob("shard*/summary.json")):
            summary = read(path)
            verifier_seconds += summary["wordings"]["A"]["seconds"]
            verifier_cases += summary["cases"]
            source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        assert total["episodes"] == 938 and verifier_cases > 0
        resources = {**total, "verifier_seconds": verifier_seconds, "verifier_cases": verifier_cases,
                     "by_suite": suites, "sources_sha256": source_hashes,
                     "timing_contract": "sum of per-episode batch-amortized generation seconds; verifier sum of shard wording-A seconds"}
        inference_path.write_text(json.dumps(resources, indent=2), encoding="utf-8")
    resources = read(inference_path)
    old_cost = original["dora_gpu_cost"]
    inference_rows = []
    for label, costs in (("Original Modal DoRA", old_cost), ("Latest evidence-order DoRA", resources)):
        for repaired in (False, True):
            seconds = costs["actor_seconds"] + (costs["verifier_seconds"] if repaired else 0)
            inference_rows.append({"model": label, "configuration": "FB+V" if repaired else "raw",
                                   "timing_contract": "recorded generation busy-time proxy", "episodes": 938,
                                   "gpu_seconds": seconds, "gpu_seconds_per_episode": seconds / 938,
                                   "gpu_hours_per_10000": seconds / 938 * 10000 / 3600,
                                   "historical_l4_gpu_only_usd_per_10000": seconds / 938 * 10000 / 3600 * rate,
                                   "output_tokens_per_actor_second": costs["output_tokens"] / costs["actor_seconds"]})
    csv_rows = [{"phase": "training", **r} for r in train_rows] + [{"phase": "inference", **r} for r in inference_rows]
    fields = list(dict.fromkeys(k for row in csv_rows for k in row))
    with DEST.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    rows = all_results["benchmark_rows"]
    lookup = {(r["model"], r["configuration"]): r for r in rows}
    lines = ["# Cost versus efficiency: LoRA and the three DoRA runs", "",
             "Prepared 14 November 2026 from saved reports. Training GPU counts, wall times and tokens "
             "are measured records. Dollar amounts below are historical scenarios, not invoices or current pricing.", "",
             "## Benchmark effectiveness", ""]
    lines += table(["Model", "Raw macro", "FB+V macro", "Raw unsupported / hidden"],
                   [[name, f"{lookup[name,'raw']['macro']*100:.2f}%", f"{lookup[name,'FB+V']['macro']*100:.2f}%",
                     f"{lookup[name,'raw']['unsupported']}/{lookup[name,'raw']['hidden_n']}"]
                    for name in ("Main LoRA", "candA LoRA", "candF1 LoRA", "Modal DoRA", "Kaggle DoRA lr5e-5", "Evidence-order DoRA")])
    lines += ["All rows use the same 938 input episodes, with checkpoint-specific scorer-v2 exclusions. "
              "Macro is not a pooled success fraction. Dividing cost by macro would not measure cost per "
              "successful episode, so no such composite is reported.", "",
              "## Training resources and GPU-only price scenario", "",
              f"Historical L4 scenario: **${rate:.4f}/GPU-hour**, retained from the Modal report checked "
              "on 12 November 2026. Applying it to Kaggle is a hypothetical equivalent compute cost. "
              "No Kaggle invoice or monetary charge is observed; free-quota use still consumes GPU allocation.", ""]
    lines += table(["Run", "GPUs", "Wall minutes", "Allocated GPU-hours", "Rows / steps", "Tokens/s (wall)", "Tokens/s/GPU", "Historical L4 scenario USD"],
                   [[r["model"], r["gpu_count"], f"{r['wall_seconds']/60:.2f}", f"{r['allocated_gpu_hours']:.4f}",
                     f"{r['rows']}/{r['steps']}", f"{r['processed_tokens_per_wall_second']:.1f}",
                     f"{r['processed_tokens_per_gpu_second']:.1f}", f"${r['historical_l4_gpu_only_usd_scenario']:.3f}"] for r in train_rows])
    total_hours = train_rows[1]["allocated_gpu_hours"] + train_rows[2]["allocated_gpu_hours"]
    lines += [f"The latest continuation inherits the lr5e-5 parent, so its training lineage totals "
              f"**{total_hours:.4f} allocated GPU-hours** (historical equivalent **${total_hours*rate:.3f}**), "
              "not just its 0.7187 incremental hours. Preparation, model loading, development evaluation, "
              "final benchmark, failed/smoke runs, idle allocation, CPU/RAM/storage and network are excluded.", "",
              "The parent four-L4 run has 3.49× lower wall time than the one-L4 Modal training, "
              "but uses more allocated GPU-hours (4.1819 versus 3.6514). Its benchmark score also falls. "
              "This is not a controlled hardware scaling experiment: LR and execution conditions differ.", "",
              "Historical main LoRA training used one T4 for **14,028.9 seconds (3.8969 hours)**, "
              "reported in `azure_roi.json`. The README's illustrative $0.35/T4-hour scenario gives "
              "about **$1.36**; it is not the same rate or GPU as the L4 scenario. Its 2,330,132 "
              "processed tokens/pass differ from DoRA's 2,134,235. No same-hardware, same-token, "
              "multi-seed LoRA-versus-DoRA cost experiment is established.", "",
              "## Recorded inference resources", "",
              "For the latest run, sum all 938 raw score rows' `seconds`, tokens and calls. "
              "The harness charges each episode its share of batch generation time. Sum the four "
              "verifier shards' wording-A seconds for FB+V. Concurrent workers are already summed: "
              "**do not multiply these sums by four again**. These are batch-amortized busy-time "
              "proxies, not user latency or total allocated/billed session duration.", ""]
    lines += table(["Run", "Configuration", "GPU-seconds/episode", "GPU-hours / 10k episodes", "Historical L4 USD / 10k", "Actor output tokens/s"],
                   [[r["model"], r["configuration"], f"{r['gpu_seconds_per_episode']:.3f}",
                     f"{r['gpu_hours_per_10000']:.3f}", f"${r['historical_l4_gpu_only_usd_per_10000']:.2f}",
                     f"{r['output_tokens_per_actor_second']:.2f}"] for r in inference_rows])
    lines += [f"Latest measured actor totals: **{resources['actor_seconds']:.3f} seconds**, "
              f"**{resources['model_calls']} model calls**, **{resources['input_tokens']:,} input tokens**, "
              f"**{resources['output_tokens']:,} output tokens**. Verifier: "
              f"**{resources['verifier_cases']} cases**, **{resources['verifier_seconds']:.1f} seconds**. "
              "A faster, shorter answering policy can reduce generation time while harming evidence safety. "
              "Interpret the resource table with the unsupported-answer rates above.", "",
              "The previous Kaggle lr5e-5 download contains final reports but not its full "
              "raw timing traces; no inference cost is invented for that checkpoint.", "",
              "### Historical LoRA serving estimates from the README's sources", ""]
    lines += table(["System", "Historical T4 GPU-seconds/episode", "Historical T4 USD / 10k episodes"],
                   [[name, f"{cost['estimated_gpu_seconds_per_episode']:.3f}", f"${cost['estimated_busy_gpu_usd_per_10000']:.2f}"]
                    for name, cost in original["historical_cost_scenario"]["local_t4"].items()])
    lines += ["These LoRA scenarios use **$0.526/T4-hour**, batched verifier estimates and their "
              "original serving settings. DoRA uses recorded L4 timings and a different scenario rate. "
              "They are not a controlled price/performance ranking. Hosted API token-price scenarios "
              "remain in [azure_roi.md](azure_roi.md) and [deepseek_roi.md](deepseek_roi.md); no current "
              "API prices are inferred here.", "",
              "## Verified training and development outcome of the latest approach", ""]
    t, c, dev = data["training"], data["curriculum"], data["development"]
    lines += [f"The evaluated adapter hash matches the downloaded training completion record. "
              f"The full continuation used LR **{t['args']['lr']}**, **{t['args']['epochs']:g} epoch**, "
              f"global batch **{t['effective_global_batch_size']}**, **{t['rows_trained']} rows**, "
              f"**{t['optimizer_steps']} steps**, and peak rank-0 memory **{t['peak_rank0_gpu_memory_gb']:.2f} GiB**. "
              f"The curriculum retained **{c['pairs']} pairs** with **{c['by_kind']['answer']} answer**, "
              f"**{c['by_kind']['abstain']} abstain** and **{c['by_kind']['search']} search** rows. "
              "Training loss was 0.0520; lower training loss did not produce better evaluation.", ""]
    lines += table(["DEV metric (159 episodes)", "Parent", "Continuation"],
                   [[key, f"{dev['parent'][key]*100:.2f}%", f"{dev['candidate'][key]*100:.2f}%"]
                    for key in ("supported_correct", "unsupported_answer", "false_abstain", "invalid", "evidence_supported_good")])
    assert dev["passes_screen"] is False
    lines += ["**The development screen failed.** Unsupported answers increased from **17/73 to 39/73** "
              "and evidence-supported good outcomes fell from **64/159 to 54/159**. This negative "
              "development result predates the reported final benchmark. The candidate should not be "
              "promoted under the stated rule. More training or a smaller generation bill is not itself "
              "an efficiency improvement when the required evidence behavior degrades.", "",
              "## Artifacts and formulas", "",
              "- Allocated training GPU-hours = wall seconds × GPU count / 3,600.",
              "- Processed tokens/s = tokens per pass × epochs / training wall seconds.",
              "- Processed tokens/s/GPU = processed tokens/s / GPU count.",
              "- Inference scenario USD/10k = amortized seconds / 938 × 10,000 / 3,600 × historical GPU rate.",
              "- Cumulative continuation training adds the parent cost; inference remains a separate phase.", "",
              "[CSV](dora_lora_cost_efficiency.csv) · [Full data and assumptions](dora_lora_cost_efficiency.json) · "
              "[Verified training/DEV snapshot](evidence_order_training_data.json) · "
              "[Inference resource snapshot](evidence_order_inference_resources.json) · "
              "[All benchmark comparison](evidence_order_all_previous_comparison.md) · "
              "[README LoRA comparison](readme_lora_vs_all_dora.md).", ""]
    source_paths = [REPORTS / p for p in ("dora_evaluation_data.json", "dora_kaggle_lr5e5_comparison.json",
                                         "evidence_order_all_previous_comparison.json", "azure_roi.json",
                                         "evidence_order_training_data.json", "evidence_order_inference_resources.json")]
    snapshot = {"report_date": "2026-11-14", "price_basis": "historical scenario from 2026-11-12; not current quote or invoice",
                "l4_usd_per_gpu_hour": rate, "training": train_rows, "inference": inference_rows,
                "continuation_lineage_gpu_hours": total_hours, "latest_dev_passes_screen": dev["passes_screen"],
                "latest_training": data, "latest_inference": resources,
                "historical_lora_training": lora["training_report"],
                "historical_lora_serving": original["historical_cost_scenario"]["local_t4"],
                "sources_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}}
    DEST.with_suffix(".json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    lines[4:4] = ["![LoRA and DoRA performance and serving cost](../docs/figures/dora_lora_cost_efficiency.svg)", "",
                  "![DoRA training elapsed time and total GPU compute](../docs/figures/dora_training_efficiency.svg)", "",
                  "Generate figures from the published snapshots with `python scripts/make_dora_efficiency_figures.py`.", ""]
    DEST.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("Cost/efficiency report and compact training/inference source snapshots saved.")


if __name__ == "__main__":
    main()
