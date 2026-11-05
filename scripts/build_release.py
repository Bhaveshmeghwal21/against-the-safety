"""Assemble everything that goes to Hugging Face under release/ (git-ignored). Nothing is uploaded here; see publish_release.py.

  python scripts/build_release.py --user BMEscanor

release/
  model_main/        LoRA adapter, 2 epochs, main mixture        (README.md = model card)
  model_control/     LoRA adapter, 2 epochs, supported-only control
  dataset_episodes/  training rows, AgentTune EventLogs, evaluation traces, manifests, scorecards (HotpotQA-derived: CC BY-SA 4.0)
  dataset_nimbus/    the fictional Nimbus Home support suite and its evaluation traces (CC BY 4.0)
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
}


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


def build_model(name: str, out: Path, card: str) -> None:
    src = TRAIN / name / "adapter"
    dst = out / f"model_{name}"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"], cfg["revision"] = BASE, REVISION
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8", newline="\n")
    copy(src / "adapter_model.safetensors", dst / "adapter_model.safetensors")
    copy(TRAIN / name / "train_report.json", dst / "train_report.json")
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

    ds = out / "dataset_episodes"
    shutil.rmtree(ds, ignore_errors=True)
    for name in ("main", "control", "dev"):
        copy(ROOT / "data" / "processed" / "training" / f"{name}.jsonl", ds / "training" / f"{name}.jsonl")
    copy(ROOT / "data" / "processed" / "episodes" / "eventlogs_train.jsonl", ds / "episodes" / "eventlogs_train.jsonl")
    for f in (ROOT / "data" / "manifests").glob("*.json"):
        copy(f, ds / "manifests" / f.name)
    for f in sorted((ROOT / "reports").glob("*")):
        if f.is_file():
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

    sp = out / "space"
    shutil.rmtree(sp, ignore_errors=True)
    shutil.copytree(ROOT / "demo" / "space", sp)
    print(json.dumps({"trace_files": counts, "out": str(out)}, indent=1))
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.stat().st_size > 50 * 2**20:
            print("WARNING large file:", p, p.stat().st_size)


if __name__ == "__main__":
    main()
