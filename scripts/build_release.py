"""Assemble everything that goes to Hugging Face under release/ (git-ignored). Nothing is uploaded here; see publish_release.py.

  python scripts/build_release.py --user BMEscanor

release/
  model_main/        LoRA adapter, 2 epochs, main mixture        (README.md = model card)
  model_control/     LoRA adapter, 2 epochs, supported-only control
  model_hi_main/, model_hi_f/  experimental Hindi adapters (hiMain from main, hiF from candF1); they failed their registered safety rules
  dataset_episodes/  training rows, AgentTune EventLogs, evaluation traces, manifests, scorecards (HotpotQA-derived: CC BY-SA 4.0)
  dataset_nimbus/    the fictional Nimbus Home support suite and its evaluation traces (CC BY 4.0)
  dataset_benchmarks/ traces on five external benchmarks (each folder keeps its source license; RGB is CC BY-NC-SA 4.0)
  space/             static trace viewer

The adapters' adapter_config.json is rewritten so that anyone can load them: the Kaggle path to the base model becomes
CohereLabs/tiny-aya-global at the pinned revision. Tokenizer files are not copied (they belong to the gated base repository).
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
HELD_BACK_REPORTS = ("dora_", "evidence_order_", "readme_lora_vs_all_dora")  # contributed DoRA reports (pull request 1): GitHub only

RUNS = ROOT / "runs" / "kaggle"
TRAIN = RUNS / "ats-train-both" / "out"
EVALS = {  # label -> (split, directory holding <system>_<split>/<parse>/)
    "dev": ("dev", RUNS / "ats-eval-dev" / "out" / "eval"),
    "dev_epoch1": ("dev", RUNS / "ats-eval-dev-ep1" / "out" / "eval"),
    "test": ("test", RUNS / "ats-eval-test" / "out" / "eval"),
    "product": ("product", RUNS / "ats-eval-product-v2" / "out" / "eval"),
    "product_run1": ("product", RUNS / "ats-eval-product" / "out" / "eval"),  # first run, flawed gold answers; kept for disclosure
    "dev_bias": ("dev", RUNS / "ats-eval-dev-bias" / "out" / "eval"),
    "dev_bias2": ("dev", RUNS / "ats-eval-dev-bias2" / "out" / "eval"),
    "dev_round2": ("dev", RUNS / "ats-eval-dev-cands" / "out" / "eval"),  # second round: candA and candB on dev (selection)
    "test2": ("test2", RUNS / "ats-eval-test2" / "out" / "eval"),  # second round: fresh split
    "dev_round3": ("dev", RUNS / "ats-patch-r3" / "out" / "eval"),  # third round: both patches on HotpotQA dev
    "dev_round3_r3dev": ("bench_r3dev", RUNS / "ats-patch-r3" / "out" / "eval"),  # main and both patches on the round-3 development set
    "dev_round3_blend": ("dev", RUNS / "ats-interp-r3b" / "out" / "eval"),  # weight blends of main and candD
    "dev_round3_blend_r3dev": ("bench_r3dev", RUNS / "ats-interp-r3b" / "out" / "eval"),
}
TRAIN_ROUND2 = RUNS / "ats-train-cands" / "out"
TRAIN_ROUND3 = RUNS / "ats-patch-r3" / "out"
A2R4, A2R5 = ROOT / "runs" / "account2_round4", ROOT / "runs" / "account2_round5"  # rounds 4 and 5 ran on the second Kaggle account
EVALS["dev_round4"] = ("dev", A2R4 / "ats-eval-jobs-r4dev" / "out" / "eval")  # round-4 checkpoints on HotpotQA dev
EVALS["dev_round5"] = ("dev", A2R5 / "ats-train-dpo-r5c" / "out" / "eval")  # round-5 preference-trained checkpoints on HotpotQA dev
EVALS["dev_round5b"] = ("dev", A2R5 / "ats-train-dpo2-r5b" / "out" / "eval")  # round-5b checkpoints on HotpotQA dev (present once that kernel is downloaded)
BENCH_EVALS = [RUNS / "ats-eval-bench-a" / "out" / "eval", RUNS / "ats-eval-bench-b" / "out" / "eval",  # plan sections 22 and 23
               RUNS / "ats-eval-jobs-r3test" / "out" / "eval", RUNS / "ats-eval-jobs-r3testA" / "out" / "eval",  # plan section 26: fresh test
               A2R4 / "ats-eval-jobs-r4dev" / "out" / "eval", A2R4 / "ats-eval-jobs-r4test" / "out" / "eval",  # round 4: development and final holdout
               A2R5 / "ats-train-dpo-r5c" / "out" / "eval", A2R5 / "ats-train-dpo2-r5b" / "out" / "eval",  # rounds 5 and 5b: development evaluation
               A2R5 / "ats-eval-jobs-r5test-ref" / "out" / "eval", A2R5 / "ats-eval-jobs-r5test-final" / "out" / "eval",  # round 5: sealed test
               RUNS / "ats-r6-system-xq" / "out" / "eval",  # multilingual check (research/2026-11-15-multilingual-plan.md)
               RUNS / "ats-r6-system-sl" / "out" / "eval",  # round 6 sealed test, second look (research/2026-11-15-round6-second-look-preregistration.md)
               RUNS / "ats-r6-system-hi-base" / "out" / "eval", RUNS / "ats-r6-system-hi-new" / "out" / "eval", RUNS / "ats-hi-train-a" / "out" / "eval"]  # Hindi training (research/2026-11-15-hindi-training-preregistration.md)
BENCHES = ["2wiki", "musique", "squad2", "rgb", "rgb_int", "r3t_hotpot", "r3t_2wiki", "r3t_musique", "r3t_squad2", "r3t_rgb", "r3t_rgb_int",
           "r4dev", "r4test", "r5t_squad", "r5t_hotpot", "r5t_rgb", "r6t_squad", "r6t_hotpot", "r6t_rgb", "xq_hi", "xq_en", "xq_es", "xq2_hi", "xq2_en", "hi_dev"]


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def copy_traces(label: str, split: str, root: Path, dst: Path) -> int:
    n = 0
    if not root.exists():
        return 0
    for system_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(f"_{split}")):
        system = system_dir.name[: -len(split) - 1]
        for mode in ("lenient", "strict"):
            for name in ("transcripts.jsonl", "scores.jsonl", "trajectories.jsonl", "summary.json"):
                src = system_dir / mode / name
                if src.exists():
                    copy(src, dst / "traces" / label / f"{system}__{mode}__{name}")
                    n += 1
        if (system_dir / "summary.json").exists():  # closed-book
            copy(system_dir / "summary.json", dst / "traces" / label / f"{system}__summary.json")
            n += 1
    return n


def build_model(name: str, out: Path, card: str, train: Path = TRAIN, dst_name: str | None = None, folder: str = "adapter") -> None:
    src = train / name / folder
    dst = out / f"model_{dst_name or name}"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"], cfg["revision"] = BASE, REVISION
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8", newline="\n")
    copy(src / "adapter_model.safetensors", dst / "adapter_model.safetensors")
    copy(train / name / "train_report.json", dst / "train_report.json")
    (dst / "README.md").write_text(card, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="BMEscanor")
    ap.add_argument("--out", default="release")
    args = ap.parse_args()
    from ragharness import release_cards as cards

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    for name in ("main", "control"):
        build_model(name, out, cards.model_card(name, args.user))
    build_model("candA", out, cards.cand_a_card(args.user), train=TRAIN_ROUND2, dst_name="cand_a")
    build_model("candD", out, cards.patch_card(args.user), train=TRAIN_ROUND3, dst_name="patch")
    build_model("candF", out, cards.cand_f1_card(args.user), train=A2R4 / "ats-train-evidence-r4" / "out", dst_name="cand_f1", folder="adapter_ep1")  # round 4 checkpoint, the round-5b alternative
    for name, (slug, _, _) in cards.HINDI_ADAPTERS.items():  # experimental Hindi adapters, final (epoch 2) weights; they failed their registered safety rules
        build_model(name, out, cards.hindi_card(name, args.user), train=RUNS / "ats-hi-train-a" / "out", dst_name=slug.replace("-", "_"))

    ds = out / "dataset_episodes"
    shutil.rmtree(ds, ignore_errors=True)
    for name in ("main", "control", "dev", "candA", "candB", "candC", "candD", "candE", "candF", "hiMain", "hiF"):
        copy(ROOT / "data" / "processed" / "training" / f"{name}.jsonl", ds / "training" / f"{name}.jsonl")
    for name in ("dpo_r5", "dpo_r5b", "dpo_r5_clean", "dpo_r5b_clean", "force_contexts_r5"):  # historical artifacts and separately validated clean pairs
        if (ROOT / "data" / "processed" / "training" / f"{name}.jsonl").exists():
            copy(ROOT / "data" / "processed" / "training" / f"{name}.jsonl", ds / "training" / f"{name}.jsonl")
    for f in sorted((ROOT / "data" / "processed" / "verifier").glob("*.jsonl")):  # the cases the self-check was run on
        copy(f, ds / "verifier" / f.name)
    for tag, folder in (("dev_main", A2R5 / "ats-verify-r5-dev"), ("dev_r4cands", A2R5 / "ats-verify-main-r5-devcands"), ("r4test", A2R5 / "ats-verify-main-r5-r4test"),
                        ("r5test_refs", A2R5 / "ats-verify-main-r5-r5test-refs"), ("dev_dpo", A2R5 / "ats-train-dpo-r5c"), ("dev_dpo2", A2R5 / "ats-train-dpo2-r5b"),
                        ("r5test_final", A2R5 / "ats-verify-main-r5-r5test-final")):
        for v in sorted(folder.rglob("verdicts_*.jsonl")):
            copy(v, ds / "verifier" / f"verdicts_{tag}__{v.parent.name}__{v.name}")
    for folder in ("ats-sample-r5-pairs", "ats-force-r5-r5b"):  # sampled rollouts and forced answers
        for f in sorted((A2R5 / folder / "out").rglob("*.jsonl")):
            copy(f, ds / "sampling" / folder / f.relative_to(A2R5 / folder / "out").as_posix().replace("/", "__"))
    copy(ROOT / "data" / "processed" / "episodes" / "eventlogs_train.jsonl", ds / "episodes" / "eventlogs_train.jsonl")
    copy(ROOT / "data" / "processed" / "episodes" / "eventlogs_train3.jsonl", ds / "episodes" / "eventlogs_train3.jsonl")
    copy(ROOT / "data" / "processed" / "episodes" / "eventlogs_trainhi.jsonl", ds / "episodes" / "eventlogs_trainhi.jsonl")
    copy(ROOT / "data" / "bench" / "r3dev" / "manifest.json", ds / "manifests" / "r3dev.json")
    for f in (ROOT / "data" / "manifests").glob("*.json"):
        copy(f, ds / "manifests" / f.name)
    for f in sorted((ROOT / "reports").glob("*")):
        if f.is_file() and not f.name.startswith(HELD_BACK_REPORTS):  # the dataset card does not describe the contributed DoRA reports (pull request 1), so they stay in the GitHub repository only
            copy(f, ds / "reports" / f.name)
    counts = {label: copy_traces(label, split, root, ds) for label, (split, root) in EVALS.items() if not label.startswith("product")}
    (ds / "README.md").write_text(cards.episodes_card(args.user), encoding="utf-8", newline="\n")

    nb = out / "dataset_nimbus"
    shutil.rmtree(nb, ignore_errors=True)
    for f in ("docs.jsonl", "tasks.jsonl", "manifest.json"):
        copy(ROOT / "data" / "product" / f, nb / f)
    counts["product"] = copy_traces("product", "product", EVALS["product"][1], nb)
    counts["product_run1"] = copy_traces("product_run1", "product", EVALS["product_run1"][1], nb)
    copy(ROOT / "reports" / "scorecard_product_lenient.md", nb / "scorecard_lenient.md")
    copy(ROOT / "reports" / "scorecard_product_strict.md", nb / "scorecard_strict.md")
    (nb / "README.md").write_text(cards.nimbus_card(args.user), encoding="utf-8", newline="\n")

    bt = out / "dataset_benchmarks"
    shutil.rmtree(bt, ignore_errors=True)
    for b in BENCHES:
        counts[f"bench_{b}"] = sum(copy_traces(b, f"bench_{b}", root, bt) for root in BENCH_EVALS)
        copy(ROOT / "data" / "bench" / b / "manifest.json", bt / "manifests" / f"{b}.json")
    for f in sorted((ROOT / "reports").glob("benchmarks_*")) + sorted((ROOT / "reports").glob("round3_test*")) + sorted((ROOT / "reports").glob("round4_test*")) \
            + sorted((ROOT / "reports").glob("round5_final*")) + sorted((ROOT / "reports").glob("round5b_final*")) + sorted((ROOT / "reports").glob("round5_system_check*")) + sorted((ROOT / "reports").glob("xquad_*")) \
            + sorted((ROOT / "reports").glob("round6_second_look*")) + sorted((ROOT / "reports").glob("hindi_*")) \
            + sorted((ROOT / "reports").glob("general_capability*")) + sorted((ROOT / "reports").glob("tool_use_r5t*")) + sorted((ROOT / "reports").glob("judge_results*")) + sorted((ROOT / "reports").glob("numeric_accuracy*")) + sorted((ROOT / "reports").glob("latency*")) + sorted((ROOT / "reports").glob("hand_label_sheet*")):
        copy(f, bt / "reports" / f.name)
    for rel in ("sample_r5t.json", "cases_r5t.jsonl", "hand_label_key.json", "judge_calls.jsonl", "judge_calls_retest.jsonl"):  # judge-based metrics (research/2026-11-16-judge-latency-preregistration.md)
        copy(ROOT / "data" / "judge" / rel, bt / "judge" / rel)
    for folder in ("run1", "run1_retest"):
        for rel in ("scores.jsonl", "run_summary.json"):
            copy(ROOT / "data" / "judge" / folder / rel, bt / "judge" / f"{folder}__{rel}")
    copy(ROOT / "data" / "general" / "manifest.json", bt / "general" / "manifest.json")  # general-capability check: the question sets are public datasets and are not redistributed; the per-question predictions are
    for kernel in ("ats-general-t4x2", "ats-general-mmlu"):
        folder = RUNS / kernel / "out" / "general"
        for f in sorted(folder.glob("*.jsonl")) + sorted(folder.glob("summary_*.json")):
            copy(f, bt / "general" / f.name)
    latency = RUNS / "ats-latency-t4" / "out" / "latency"
    for rel in ("latency_summary.json", "requests_base.jsonl", "requests_submitted.jsonl"):
        if (latency / rel).exists():
            copy(latency / rel, bt / "latency" / rel)
    for tag, folder in (("base", RUNS / "ats-r6-system-hi-base" / "out"), ("new", RUNS / "ats-r6-system-hi-new" / "out")):  # Hindi training: the self-check's cases and verdicts
        for rel in ("cases/answers.jsonl", "verify/V/verdicts_A.jsonl"):
            if (folder / rel).exists():
                copy(folder / rel, bt / "hindi_training" / f"{tag}__{rel.replace('/', '__')}")  # the baseline verdicts are the batch-1 rerun on the saved cases (kernel ats-verify-cases-hi-base)
    sl = RUNS / "ats-r6-system-sl" / "out"  # the self-check's verdicts, the forced second-look answers and the checks of those answers
    for rel in ("cases/answers.jsonl", "cases/forced_all.jsonl", "cases/recover_contexts.jsonl", "forced/forced_main.jsonl",
                "verify/V/verdicts_A.jsonl", "verify_forced/V/verdicts_A.jsonl"):
        if (sl / rel).exists():
            copy(sl / rel, bt / "round6_second_look" / rel.replace("/", "__"))
    (bt / "README.md").write_text(cards.bench_card(args.user), encoding="utf-8", newline="\n")

    sp = out / "space"
    shutil.rmtree(sp, ignore_errors=True)
    shutil.copytree(ROOT / "demo" / "space", sp)
    print(json.dumps({"trace_files": counts, "out": str(out)}, indent=1))
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.stat().st_size > 50 * 2**20:
            print("WARNING large file:", p, p.stat().st_size)


if __name__ == "__main__":
    main()
