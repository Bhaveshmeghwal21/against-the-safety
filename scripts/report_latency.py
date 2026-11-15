"""Turn the Kaggle latency run into a report (research/2026-11-16-judge-latency-preregistration.md, section 6).

  python scripts/report_latency.py --run runs/kaggle/ats-latency-t4/out/latency --out reports/latency

Reads `latency_summary.json` and the per-request files written by `scripts/latency_run.py`. Percentiles and requests per second are AuditKIT's own
(`LatencyStats`, `Throughput`), computed there. Cost per request is the measured seconds on a busy T4 at the rate used in `reports/azure_roi.md`.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T4_DOLLARS_PER_HOUR = 0.526  # Azure Retail Prices API, Standard_NC4as_T4_v3, East US (reports/azure_roi.md)


def seconds(ms: float) -> str:
    return f"{ms / 1000:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kaggle/ats-latency-t4/out/latency")
    ap.add_argument("--out", default="reports/latency")
    args = ap.parse_args()
    run = ROOT / args.run
    summary = json.loads((run / "latency_summary.json").read_text(encoding="utf-8"))
    env = summary["environment"]
    result = {"environment": env, "arms": {}}
    lines = ["# Single-request latency and throughput", "",
             f"One request at a time (batch size 1), greedy decoding, at most {env['max_new_tokens']} new tokens per turn, {env['gpu']}, model in 4-bit, "
             f"Hugging Face `generate` with no serving-engine tuning. {env['episodes_per_part'] * 3} requests from the three r5t suites (seed {env['seed']}), "
             f"{env['warmup']} untimed warm-up requests per arm. Percentiles and requests per second use AuditKIT's `LatencyStats` and `Throughput`. "
             "The base arm also applies the fail-closed rule and the code checks of the self-check (cited passage was shown, answer text is in it), which cost no model call; it skips only the model self-check. The pre-registration is [here](../research/2026-11-16-judge-latency-preregistration.md).", "",
             "| System | Requests | p50 (s) | p95 (s) | p99 (s) | Mean (s) | Max (s) | Requests per second | Mean output tokens | Answers shown after the repairs | Peak GPU memory (GB) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    names = {"base": "Tiny Aya base, same loop, fail-closed rule and code checks, no model self-check", "submitted": "Submitted: main adapter, fail-closed rule, self-check"}
    for arm, s in summary["arms"].items():
        lat = s["latency_ms"]
        cost = lat["mean"] / 1000 / 3600 * T4_DOLLARS_PER_HOUR
        result["arms"][arm] = {**s, "cost_per_1000_requests_usd_busy_t4": round(cost * 1000, 3)}
        lines.append(f"| {names.get(arm, arm)} | {s['requests']} | {seconds(lat['p50'])} | {seconds(lat['p95'])} | {seconds(lat['p99'])} | {seconds(lat['mean'])} | {seconds(lat['max'])} | "
                     f"{s['throughput']['rps']:.3f} | {s['mean_output_tokens']:.0f} | {s['answered']} | {s['peak_gpu_memory_gb']} |")
    lines += ["", "## Where the time goes (mean per request)", "", "| System | Generation (s) | Search (s) | Self-check (s) | Other (s) | Estimated cost per 1,000 requests on a busy T4 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, s in summary["arms"].items():
        m = s["mean_ms"]
        other = s["latency_ms"]["mean"] - sum(m.values())
        lines.append(f"| {names.get(arm, arm)} | {seconds(m['generation_ms'])} | {seconds(m['search_ms'])} | {seconds(m['check_ms'])} | {seconds(other)} | "
                     f"${result['arms'][arm]['cost_per_1000_requests_usd_busy_t4']:.2f} |")
    lines += ["", "## By suite (p50 and p95, seconds)", "", "| System | " + " | ".join(next(iter(summary["arms"].values()))["parts"]) + " |",
              "| --- | " + " | ".join("---:" for _ in next(iter(summary["arms"].values()))["parts"]) + " |"]
    for arm, s in summary["arms"].items():
        lines.append(f"| {names.get(arm, arm)} | " + " | ".join(f"{seconds(v['p50'])} and {seconds(v['p95'])}" for v in s["parts"].values()) + " |")
    lines += ["", f"Cost uses ${T4_DOLLARS_PER_HOUR} per T4 hour (`reports/azure_roi.md`) and assumes the GPU is busy all the time. These are single-request figures on a research stack, "
              "measured on a shared Kaggle host. A serving engine with batching would lower the cost per request and could raise the latency of an individual request."]
    (ROOT / args.out).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (ROOT / args.out).with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
