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
    status = kaggle("datasets", "status", f"{username()}/{SLUG}", check=False).lower()  # `datasets list -s` is a search index and can lag
    exists = any(w in status for w in ("ready", "pending", "processing", "error"))
    if exists:
        print(kaggle("datasets", "version", "-p", str(BUNDLE), "-m", f"bundle {time.strftime('%Y-%m-%d %H:%M')}", "--dir-mode", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(BUNDLE), "--dir-mode", "zip"))  # private by default
    wait_until_served()


def bundle_id() -> str:
    """Content id of the local bundle (written into the zip by bundle_for_kaggle.py)."""
    import zipfile

    with zipfile.ZipFile(BUNDLE / "against-the-safety.zip") as z:
        return z.read("BUNDLE_ID").decode().strip()


def served_bundle_id() -> str:
    """The BUNDLE_ID Kaggle currently serves for the dataset ('' if it cannot be read yet)."""
    import shutil

    tmp = ROOT / ".cache" / "kaggle_probe"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    kaggle("datasets", "download", f"{username()}/{SLUG}", "-f", "BUNDLE_ID", "-p", str(tmp), "--unzip", "--force", check=False)
    f = tmp / "BUNDLE_ID"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


def wait_until_served(timeout: int = 900) -> None:
    """`datasets status` says "ready" for the previous version while a new one is still processing; compare content ids."""
    want, t0 = bundle_id(), time.time()
    while time.time() - t0 < timeout:
        if served_bundle_id() == want:
            print(f"dataset serves bundle {want}")
            return
        time.sleep(10)
    raise SystemExit(f"dataset still does not serve bundle {want} after {timeout}s")


def run_stage(stage: str, adapters_from: str | None, extra: str, tag: str = "") -> str:
    if served_bundle_id() != bundle_id():
        raise SystemExit("the dataset on Kaggle is not the local bundle; run `kaggle_launch.py dataset` first (it waits until the new version is served)")
    slug = f"ats-{stage}" + (f"-{tag}" if tag else "")
    kdir = ROOT / "runs" / "kaggle" / f"push-{slug}"
    kdir.mkdir(parents=True, exist_ok=True)
    code = (
        "import pathlib, subprocess, sys\n"
        "p = next(pathlib.Path('/kaggle/input').rglob('bootstrap.py'))\n"
        f"expect = {bundle_id()!r}\n"
        "idf = p.parent.parent / 'BUNDLE_ID'\n"
        "got = idf.read_text().strip() if idf.exists() else 'missing'\n"
        "if got != expect:\n"
        "    raise SystemExit(f'stale dataset mount: this kernel was launched for bundle {expect}, the mounted one is {got}')\n"
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
    r.add_argument("--tag", default="", help="suffix for the kernel slug, so two runs of one stage do not overwrite each other")
    s = sub.add_parser("status")
    s.add_argument("slug")
    o = sub.add_parser("output")
    o.add_argument("slug")
    a = ap.parse_args()
    if a.cmd == "dataset":
        make_dataset()
    elif a.cmd == "run":
        print("kernel slug:", run_stage(a.stage, a.adapters_from, a.extra, a.tag))
    elif a.cmd == "status":
        print(kaggle("kernels", "status", f"{username()}/{a.slug}"))
    elif a.cmd == "output":
        dest = ROOT / "runs" / "kaggle" / a.slug
        dest.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", f"{username()}/{a.slug}", "-p", str(dest)))


if __name__ == "__main__":
    main()
