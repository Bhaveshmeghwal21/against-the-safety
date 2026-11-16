"""Draw README-style SVGs from saved DoRA/LoRA metrics; standard library only."""
import json
from pathlib import Path

import make_azure_figures as F

ROOT = Path(__file__).resolve().parents[1]
COLORS = [F.NAVY, "#507b99", F.TEAL, F.GOLD, "#8055a1", "#b05260"]
LABELS = [("Main LoRA", "Main LoRA + repairs"), ("candA LoRA", "candA LoRA + repairs"),
          ("candF1 LoRA", "candF1 LoRA + repairs"), ("Modal DoRA", "Modal DoRA + repairs"),
          ("Kaggle DoRA lr5e-5", "Kaggle DoRA + repairs"), ("Evidence-order DoRA", "Evidence-order + repairs")]


def bar(svg, left, y, value, maximum, width, color, label):
    assert 0 <= value <= maximum
    length = width * value / maximum
    svg.append(f'<rect x="{left}" y="{y}" width="{length:.2f}" height="23" rx="4" fill="{color}"/>')
    svg.append(F.text(f"{left+length+7:.2f}", y + 17, label, 13, F.INK, 700))


def serving(cost, benchmark):
    results = {(r["model"], r["configuration"]): r for r in benchmark["benchmark_rows"]}
    prices = {
        "Main LoRA": (cost["historical_lora_serving"]["main+FB+V"]["estimated_busy_gpu_usd_per_10000"], "T4"),
        "candF1 LoRA": (cost["historical_lora_serving"]["candF1+FB+V"]["estimated_busy_gpu_usd_per_10000"], "T4"),
        "candA LoRA": (cost["historical_lora_serving"]["candA+FB+V"]["estimated_busy_gpu_usd_per_10000"], "T4"),
    }
    for name, source in (("Modal DoRA", "Original Modal DoRA"), ("Evidence-order DoRA", "Latest evidence-order DoRA")):
        row = next(r for r in cost["inference"] if r["model"] == source and r["configuration"] == "FB+V")
        prices[name] = (row["historical_l4_gpu_only_usd_per_10000"], "L4")
    svg = F.begin("LoRA and DoRA: performance and serving cost", "All six adapters with fail-closed and the fixed main-LoRA verifier. "
                  "Serving costs are estimates from saved run times; the checker part is an average. "
                  "Kaggle lr5e-5 has no timing data. "
                  "Historical T4 and L4 assumptions differ; no equal-hardware cost ranking is established.", 518)
    svg += [F.text(30, 34, "LoRA and DoRA: performance and serving cost", 23, F.INK, 700),
            F.text(30, 58, "938 input episodes · fail-closed + fixed main-LoRA verifier", 13, F.MUTED),
            F.text(245, 100, "Macro success", 16, F.INK, 700),
            F.text(245, 121, "Higher is better", 12, F.MUTED),
            F.text(583, 100, "USD per 10,000 episodes", 16, F.INK, 700),
            F.text(583, 121, "Estimated busy-GPU scenarios", 12, F.MUTED)]
    F.axis(svg, 245, 240, 138, 406, 65, (0, 20, 40, 60))
    F.axis(svg, 583, 235, 138, 406, 45, (0, 15, 30, 45), money=True)
    for i, ((name, label), color) in enumerate(zip(LABELS, COLORS)):
        y = 152 + i * 42
        row = results[name, "FB+V"]
        svg.append(F.text(226, y + 17, label, 13, F.INK, 600, "end"))
        bar(svg, 245, y, row["macro"] * 100, 65, 240, color, f"{row['macro']*100:.2f}%")
        if name in prices:
            amount, hardware = prices[name]
            bar(svg, 583, y, amount, 45, 235, color, f"${amount:.2f}")
            svg.append(F.text(881, y + 17, hardware, 11, F.MUTED))
        else:
            svg.append(F.text(591, y + 17, "Not measured", 13, F.MUTED))
    svg += [F.text(30, 453, "T4: $0.526/GPU-hour. L4: $0.7992/GPU-hour (historical 12 Nov 2026 scenario). Not invoices.", 12, F.MUTED),
            F.text(30, 475, "Different hardware, batching and serving assumptions; excludes idle time, loading and other infrastructure.", 12, F.MUTED),
            F.text(30, 497, "Checkpoint-specific scorer exclusions differ. Lower cost alone does not establish better evidence safety.", 12, F.MUTED), "</svg>"]
    return "\n".join(svg) + "\n"


def training(cost):
    rows = cost["training"]
    assert len(rows) == 3
    minutes = [rows[0]["wall_seconds"] / 60, rows[1]["wall_seconds"] / 60,
               (rows[1]["wall_seconds"] + rows[2]["wall_seconds"]) / 60]
    hours = [rows[0]["allocated_gpu_hours"], rows[1]["allocated_gpu_hours"], cost["continuation_lineage_gpu_hours"]]
    labels = ["Modal DoRA (1 L4)", "Kaggle parent (4 L4)", "Evidence-order lineage"]
    svg = F.begin("DoRA training: elapsed time versus total compute", "Training-only resource records. "
                  "The latest checkpoint includes its four-L4 parent's training plus continuation. "
                  "This is not a controlled scaling experiment and excludes preparation and evaluation.", 395)
    svg += [F.text(30, 34, "DoRA training: elapsed time versus total compute", 23, F.INK, 700),
            F.text(30, 58, "Training-loop measurements · latest checkpoint includes its parent", 13, F.MUTED),
            F.text(245, 100, "Sequential elapsed minutes", 16, F.INK, 700),
            F.text(245, 121, "Training loops only", 12, F.MUTED),
            F.text(583, 100, "Allocated GPU-hours", 16, F.INK, 700),
            F.text(583, 121, "Wall hours × GPU count", 12, F.MUTED)]
    # Explicit units: the shared README helper's percentage ticks are unsuitable.
    for left, maximum, ticks in ((245, 250, (0, 125, 250)), (583, 6, (0, 3, 6))):
        for tick in ticks:
            x = left + 235 * tick / maximum
            svg.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="138" y2="277" stroke="{F.GRID}"/>')
            svg.append(F.text(f"{x:.2f}", 298, tick, 12, F.MUTED, anchor="middle"))
    for i, (label, color) in enumerate(zip(labels, COLORS[3:])):
        y = 156 + i * 43
        svg.append(F.text(226, y + 17, label, 13, F.INK, 600, "end"))
        bar(svg, 245, y, minutes[i], 250, 235, color, f"{minutes[i]:.2f}")
        bar(svg, 583, y, hours[i], 6, 235, color, f"{hours[i]:.4f}")
    svg += [F.text(30, 330, f"Continuation alone: {rows[2]['wall_seconds']/60:.2f} minutes on 4 L4s = {rows[2]['allocated_gpu_hours']:.4f} GPU-hours.", 12, F.MUTED),
            F.text(30, 352, "Different LR, data and continuation histories; not a controlled GPU-scaling or equal-training-budget experiment.", 12, F.MUTED),
            F.text(30, 374, "Preparation, loading, DEV/final evaluation, failed runs and idle allocation are excluded.", 12, F.MUTED), "</svg>"]
    return "\n".join(svg) + "\n"


def main():
    costs = json.loads((ROOT / "reports/dora_lora_cost_efficiency.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "reports/evidence_order_all_previous_comparison.json").read_text(encoding="utf-8"))
    F.FIGURES.mkdir(parents=True, exist_ok=True)
    for name, content in (("dora_lora_cost_efficiency.svg", serving(costs, metrics)),
                          ("dora_training_efficiency.svg", training(costs))):
        path = F.FIGURES / name
        path.write_text(content, encoding="utf-8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()
