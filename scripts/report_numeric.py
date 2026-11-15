"""Numeric accuracy on the first sealed test (research/2026-11-16-judge-latency-preregistration.md, section 5).

  python scripts/report_numeric.py

A numeric-answer episode is an eligible (scorer v3), answerable episode whose gold answer is a single number, optionally with a currency sign, thousands
separators or a percent sign. Numeric accuracy is the share of those episodes where the system answered and the numbers in its answer include the gold
value. Systems are scored on their own eligible episodes, as in the headline table. AuditKIT has no numeric metric, and the datasets carry no numeric type,
so the label is derived from the gold string.
"""

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

GOLD = re.compile(r"^\W*(?:[$€£]\s*)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(?:%|percent|per cent)?\W*$", re.I)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def gold_number(gold: str | None) -> float | None:
    m = GOLD.match(gold or "")
    return float(m.group(1).replace(",", "") + (m.group(2) or "")) if m else None


def numbers_in(text: str | None) -> list[float]:
    out = []
    for token in NUMBER.findall(text or ""):
        try:
            out.append(float(token.replace(",", "").rstrip(".")))
        except ValueError:
            continue
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def score_system(rows: list[dict], answers: dict) -> dict:
    subset = [r for r in rows if not r["excluded_v3"] and r["answerable"] and gold_number(answers[(r["part"], r["qid"], r["variant"])]["gold"]) is not None]
    answered = [r for r in subset if r["final"] == "answer"]
    correct = [r for r in answered
               if gold_number(answers[(r["part"], r["qid"], r["variant"])]["gold"]) in numbers_in(answers[(r["part"], r["qid"], r["variant"])]["answer"])]
    n, a, c = len(subset), len(answered), len(correct)
    return {"episodes": n, "questions": len({r["qid"] for r in subset}), "answered": a, "numerically_correct": c,
            "numeric_accuracy": c / n if n else None, "numeric_accuracy_ci95": wilson(c, n),
            "answered_share": a / n if n else None, "correct_among_answered": c / a if a else None, "correct_among_answered_ci95": wilson(c, a)}


def main() -> None:
    import remediate_audit as audit
    import report_azure_compare as azure
    import report_deepseek_compare as deepseek

    sources = audit.squad_sources()
    cases, verdicts = deepseek.cases_and_verdicts()
    roots = {"base": (azure.R5REF, "base"), "main": (azure.R5REF, "main"), "candF1": (azure.R5ALT, "candF1")}
    rows, answers = {}, {}
    for name, (root, label) in roots.items():
        rows[name] = []
        for part in azure.PARTS:
            folder = root / f"{label}_bench_{part}/lenient"
            rows[name] += audit.load(folder, part, sources)[2]
            for t in audit.A.RS.read(folder / "transcripts.jsonl"):
                answers[(name, part, t["qid"], t["variant"])] = {"gold": t.get("gold_answer"), "answer": (t["final"].get("answer") if t["final"]["kind"] == "answer" else None)}
    wrapped = audit.wrappers(rows, cases, verdicts)
    systems = {"base": "base", "main": "main", "main+FB+V (submitted)": "main+FB+V", "candF1+FB+V (alternative)": "candF1+FB+V"}
    result = {"definition": "eligible answerable episodes whose gold answer is a single number; correct = the answer's numbers include the gold value", "systems": {}}
    for label, key in systems.items():
        raw = key.split("+")[0]
        lookup = {(r["part"], r["qid"], r["variant"]): answers[(raw, r["part"], r["qid"], r["variant"])] for r in wrapped[key]}
        result["systems"][label] = score_system(wrapped[key], lookup)
    (ROOT / "reports/numeric_accuracy.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def ci(x):
        return "n/a" if x is None else f"{100 * x[0]:.0f} to {100 * x[1]:.0f}"

    lines = ["# Numeric accuracy, first sealed test", "",
             "Episodes whose gold answer is a single number (derived from the gold string; the datasets carry no numeric type). A numeric answer counts as correct when "
             "the numbers in the answer include the gold value. Intervals are 95% Wilson intervals. The subset is small, so read the counts.", "",
             "| System | Numeric-answer episodes (questions) | Answered | Numerically correct | Numeric accuracy (95% interval) | Correct among answered (95% interval) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, s in result["systems"].items():
        lines.append(f"| {label} | {s['episodes']} ({s['questions']}) | {s['answered']} | {s['numerically_correct']} | {pct(s['numeric_accuracy'])} ({ci(s['numeric_accuracy_ci95'])}) | "
                     f"{pct(s['correct_among_answered'])} ({ci(s['correct_among_answered_ci95'])}) |")
    (ROOT / "reports/numeric_accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
