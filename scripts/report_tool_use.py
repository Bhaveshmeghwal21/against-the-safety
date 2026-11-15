"""Agentic trajectory metrics on the first sealed test, per suite, from the saved runs (no new model run).

  python scripts/report_tool_use.py

Tool-call metrics are AuditKIT's, as each evaluation run saved them in `summary.json` (`auditkit.all`): tool-call validity (share of well-formed search
calls), redundant tool calls (share of searches that repeat an earlier query; lower is better) and agent loop detection (share of episodes with no
repetition loop). They describe the raw agent, before the fail-closed rule, over every episode of the suite. Mean searches per episode and the share of
invalid endings come from the saved transcripts. Task success per question group is scorer v3 with both repairs, from reports/reference_policies.json.
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R5REF = ROOT / "runs/account2_round5/ats-eval-jobs-r5test-ref/out/eval"
R5ALT = ROOT / "runs/account2_round5/ats-eval-jobs-r5test-final/out/eval"
SYSTEMS = {"base": (R5REF, "base"), "main": (R5REF, "main"), "candF1": (R5ALT, "candF1")}
PARTS = {"r5t_squad": "SQuAD", "r5t_hotpot": "HotpotQA", "r5t_rgb": "RGB"}
GROUPS = {"single": "SQuAD single", "compound": "SQuAD two-part", "hotpot": "HotpotQA", "rgb": "RGB"}


def main() -> None:
    result = {"source": "summary.json (AuditKIT) and transcripts.jsonl of the first sealed test runs; raw agents, before the repairs", "trajectories": {}, "task_success": {}}
    for name, (root, label) in SYSTEMS.items():
        result["trajectories"][name] = {}
        for part in PARTS:
            folder = root / f"{label}_bench_{part}" / "lenient"
            ak = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["auditkit"]["all"]
            ts = [json.loads(l) for l in (folder / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            result["trajectories"][name][part] = {
                "episodes": len(ts), "tool_call_validity": ak["tool_call_validity"], "redundant_tool_calls": ak["redundant_tool_calls"],
                "loop_free_episodes": ak["agent_loop_detection"], "mean_searches": statistics.mean(t["n_searches"] for t in ts),
                "invalid_endings": sum(t["final"]["kind"] == "invalid" for t in ts) / len(ts)}
    ref = json.loads((ROOT / "reports/reference_policies.json").read_text(encoding="utf-8"))["sealed_r5test"]
    for name in ("base", "main+FB+V", "candF1+FB+V", "always_refuse"):
        result["task_success"][name] = {**{g: ref[name]["components"][g] for g in GROUPS}, "macro": ref[name]["macro"]}
    (ROOT / "reports/tool_use_r5t.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    lines = ["# Tool use and task success per suite, first sealed test", "",
             "Tool-call metrics are AuditKIT's (saved by each evaluation run) for the raw agent, before the fail-closed rule, over every episode. "
             "Task success is scorer v3 with both repairs (`reports/reference_policies.json`). No new model run.", "",
             "| System | Suite | Episodes | Tool-call validity (higher is better) | Redundant tool calls (lower is better) | Loop-free episodes (higher is better) | Searches per episode | Invalid endings before the fail-closed rule (lower is better) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, parts in result["trajectories"].items():
        for part, m in parts.items():
            lines.append(f"| {name} | {PARTS[part]} | {m['episodes']} | {m['tool_call_validity']:.3f} | {m['redundant_tool_calls'] * 100:.1f}% | {m['loop_free_episodes'] * 100:.1f}% | "
                         f"{m['mean_searches']:.2f} | {m['invalid_endings'] * 100:.1f}% |")
    lines += ["", "| System | " + " | ".join(GROUPS.values()) + " | Macro success |", "| --- | " + " | ".join("---:" for _ in GROUPS) + " | ---: |"]
    for name, s in result["task_success"].items():
        lines.append(f"| {name} | " + " | ".join(f"{s[g] * 100:.1f}%" for g in GROUPS) + f" | {s['macro'] * 100:.1f}% |")
    (ROOT / "reports/tool_use_r5t.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
