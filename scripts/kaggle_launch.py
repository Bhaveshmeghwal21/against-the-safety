"""Push the bundle as a private Kaggle dataset and run one stage as a GPU kernel, from this machine.

  python scripts/kaggle_launch.py dataset                         # create or update the bundle dataset
  python scripts/kaggle_launch.py run smoke                       # push + start the smoke kernel (T4 x2)
  python scripts/kaggle_launch.py run train-both
  python scripts/kaggle_launch.py run eval-dev --adapters-from <kernel-slug>
  python scripts/kaggle_launch.py status <kernel-slug>
  python scripts/kaggle_launch.py output <kernel-slug>            # download outputs to runs/kaggle/<slug>

Credentials are read from ~/.kaggle/kaggle.json into this process only; the CLI's own config dir is .cache/kaggle.
No Hugging Face token is sent anywhere: the model comes from Kaggle Models (cohere-labs/tiny-aya, variant global).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "runs" / "kaggle_bundle"
MODEL_SOURCE = "cohere-labs/tiny-aya/transformers/global/1"
SLUG = "against-the-safety-bundle"


def creds() -> dict:
    kj = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text(encoding="utf-8"))
    return {"KAGGLE_USERNAME": kj["username"], "KAGGLE_KEY": kj["key"], "KAGGLE_CONFIG_DIR": str(ROOT / ".cache" / "kaggle")}


def kaggle(*args: str, check: bool = True) -> str:
    import shutil

    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle"))
    env = {**os.environ, **creds(), "PYTHONUTF8": "1"}
    p = subprocess.run([exe, *args], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def username() -> str:
    return creds()["KAGGLE_USERNAME"]


def make_dataset() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "bundle_for_kaggle.py")], check=True, env={**os.environ, "PYTHONUTF8": "1"})
    meta = {"title": "against-the-safety-bundle", "id": f"{username()}/{SLUG}", "licenses": [{"name": "CC-BY-SA-4.0"}]}
    (BUNDLE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    exists = SLUG in kaggle("datasets", "list", "--mine", "-s", SLUG, check=False)
    if exists:
        print(kaggle("datasets", "version", "-p", str(BUNDLE), "-m", f"bundle {time.strftime('%Y-%m-%d %H:%M')}", "--dir-mode", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(BUNDLE), "--dir-mode", "zip"))  # private by default


def run_stage(stage: str, adapters_from: str | None, extra: str) -> str:
    slug = f"ats-{stage}"
    kdir = ROOT / "runs" / "kaggle" / f"push-{slug}"
    kdir.mkdir(parents=True, exist_ok=True)
    code = (
        "import subprocess, sys, pathlib\n"
        "p = next(pathlib.Path('/kaggle/input').rglob('bootstrap.py'))\n"
        "raise SystemExit(subprocess.call([sys.executable, str(p), " + repr(stage) + (", " + repr(extra) if extra else "") + "]))\n"
    )
    (kdir / "run.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{username()}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{username()}/{SLUG}"], "competition_sources": [],
        "kernel_sources": [f"{username()}/{adapters_from}"] if adapters_from else [],
        "model_sources": [MODEL_SOURCE],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(kaggle("kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4"))
    return slug


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dataset")
    r = sub.add_parser("run")
    r.add_argument("stage")
    r.add_argument("--adapters-from", default=None)
    r.add_argument("--extra", default="")
    s = sub.add_parser("status")
    s.add_argument("slug")
    o = sub.add_parser("output")
    o.add_argument("slug")
    a = ap.parse_args()
    if a.cmd == "dataset":
        make_dataset()
    elif a.cmd == "run":
        print("kernel slug:", run_stage(a.stage, a.adapters_from, a.extra))
    elif a.cmd == "status":
        print(kaggle("kernels", "status", f"{username()}/{a.slug}"))
    elif a.cmd == "output":
        dest = ROOT / "runs" / "kaggle" / a.slug
        dest.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", f"{username()}/{a.slug}", "-p", str(dest)))


if __name__ == "__main__":
    main()
