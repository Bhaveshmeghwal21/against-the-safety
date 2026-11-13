"""Report refusal and answered-accuracy reference policies from saved runs."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def always_refuse(rows):
    result = []
    for row in rows:
        hidden = not row["answerable"]
        outcome = "correct_abstain" if hidden else "false_abstain"
        result.append({**row, "final": "abstain", "outcome_v2": outcome, "outcome_v3": outcome,
                       "good_v2": hidden, "good_v3": hidden, "strong_v2": hidden, "strong_v3": hidden})
    return result


def answered_stats(rows):
    answers = [row for row in rows if row["final"] == "answer"]
    correct = sum(row["outcome_v3"] == "supported_correct" for row in answers)
    return {"answered": len(answers), "answered_share": len(answers) / len(rows) if rows else None,
            "correct_among_answered": correct / len(answers) if answers else None,
            "wrong_or_unsupported_shown_share": (len(answers) - correct) / len(rows) if rows else None}


def sync_wrapped(rows):
    """The existing offline wrappers update v2 aliases of v3 fields."""
    return [{**row, **{f"{field}_v3": row[f"{field}_v2"] for field in ("outcome", "good", "strong")}}
            for row in rows]


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    import remediate_audit as audit
    import report_azure_compare as azure
    import report_deepseek_compare as deepseek

    required = [ROOT / "reports/round5_test_manifest.json", ROOT / "data/bench/r5t_rgb/questions.jsonl",
                ROOT / "data/processed/verifier/cases_r5test_refs.jsonl",
                ROOT / "runs/account2_round5/ats-eval-jobs-r5test-ref/out/eval"]
    missing = next((path for path in required if not path.exists()), None)
    if missing:
        raise SystemExit(f"Required saved artifact is missing: {missing}")
    sources = audit.squad_sources()
    cases, verdicts = deepseek.cases_and_verdicts()
    specs = {"base": (azure.R5REF, "base"), "main": (azure.R5REF, "main"),
             "candF1": (azure.R5ALT, "candF1"),
             "azure_native": (deepseek.REMOTE["azure_native"], "azure"),
             "azure_deepseek_native": (deepseek.REMOTE["azure_deepseek_native"], "azure")}
    rows = {name: sum((audit.load(root / f"{label}_bench_{part}/lenient", part, sources)[2]
                       for part in azure.PARTS), []) for name, (root, label) in specs.items()}
    wrapped = audit.wrappers(rows, cases, verdicts)
    systems = {name: sync_wrapped(wrapped[name]) for name in
               ("base", "main", "main+FB+V", "candF1+FB+V", "azure_native+FB", "azure_deepseek_native+FB")}
    systems["always_refuse"] = always_refuse(rows["main"])
    sealed = {}
    for name, values in systems.items():
        eligible = [r for r in values if not r["excluded_v3"]]
        indexed = azure.by_key(eligible)
        sealed[name] = {**azure.r5_stats(indexed, sorted(indexed)), **answered_stats(eligible)}

    external = {}
    for part in azure.OLD:
        path = azure.OLDMAIN / f"main_bench_{part}/lenient"
        values = [r for r in audit.load(path, part, sources)[2] if not r["excluded_v3"]]
        external[part] = {}
        for name, candidates in (("main", values), ("always_refuse", always_refuse(values))):
            indexed = azure.by_key(candidates)
            external[part][name] = {**azure.old_stats(indexed, sorted(indexed)), **answered_stats(candidates)}
        print(f"external {part}: {len(values)} eligible episodes", flush=True)

    result = {"scorer": "v3", "scope": "Each system uses its own available eligible episodes.",
              "sealed_r5test": sealed, "external": external}
    (ROOT / "reports/reference_policies.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    def percent(value):
        return "—" if value is None else f"{100 * value:.1f}%"
    lines = ["# Reference policies", "", "Each row uses its own available, scorer-v3 eligible episodes.", "",
             "| System | Macro success | Correct and cited, of answerable | Answers given, of hidden-evidence | Answered at all | Correct among answered |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in sealed.items():
        lines.append(f"| {name} | {percent(s['macro'])} | {percent(s['supported_correct'])} | "
                     f"{s['unsupported']}/{s['hidden_n']} | {percent(s['answered_share'])} | {percent(s['correct_among_answered'])} |")
    lines += ["", "## Earlier benchmark suites", "",
              "| Suite | System | Success | Correct and cited, of answerable | Answers given, of hidden-evidence | Answered at all | Correct among answered |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for part, group in external.items():
        for name, s in group.items():
            lines.append(f"| {part} | {name} | {percent(s['success'])} | {percent(s['supported_correct'])} | "
                         f"{s['unsupported']}/{s['hidden_n']} | {percent(s['answered_share'])} | {percent(s['correct_among_answered'])} |")
    (ROOT / "reports/reference_policies.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]), flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required saved artifact is missing: {exc.filename or exc}") from exc
