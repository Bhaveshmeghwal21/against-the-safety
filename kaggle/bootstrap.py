"""Kaggle notebook bootstrap: unpack the bundle, install pinned libraries for the T4 driver, run a stage.

In a Kaggle notebook (GPU T4, internet on), with the bundle dataset attached and the secret HF_TOKEN set
(the account must have accepted the Tiny Aya gate):

    !python /kaggle/input/against-the-safety-bundle/kaggle/bootstrap.py smoke

Stages: smoke | train-main | train-control | eval-dev-base | eval-dev-adapter | eval-test-base | eval-test-adapter | eval-test-closed
Everything is written under /kaggle/working/out (download it or save the version). Training and evaluation stages need
the smoke stage to have passed once on this driver.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "repo"
OUT = WORK / "out"


def sh(cmd: str, check: bool = True) -> str:
    print("$", cmd, flush=True)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print((p.stdout + p.stderr)[-3000:], flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {cmd}")
    return p.stdout


def driver_major() -> int:
    out = subprocess.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader", shell=True, text=True, capture_output=True).stdout
    m = re.match(r"(\d+)", out.strip())
    return int(m.group(1)) if m else 0


def unpack() -> None:
    if REPO.exists():
        return
    zips = list(Path("/kaggle/input").rglob("against-the-safety.zip"))
    if not zips:
        raise SystemExit("attach the bundle dataset (against-the-safety.zip) to this notebook")
    REPO.mkdir(parents=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(REPO)


def python_for_env() -> str:
    """AgentTune needs Python 3.12+. Use the notebook's interpreter if it qualifies, otherwise build a 3.12 venv with uv."""
    if sys.version_info >= (3, 12):
        return sys.executable
    venv = WORK / "venv312"
    if not (venv / "bin" / "python").exists():
        sh("pip install -q uv")
        sh(f"uv venv --python 3.12 --seed {venv}")
    return str(venv / "bin" / "python")


def install(py: str) -> None:
    marker = WORK / ".installed"
    if marker.exists():
        return
    major = driver_major()
    # torch 2.11.0's default wheel is built for CUDA 13, which needs driver 580+; older drivers use the cu126 build.
    index = "" if major >= 580 else "--index-url https://download.pytorch.org/whl/cu126"
    print(f"NVIDIA driver major {major}: torch index = {index or 'default (CUDA 13)'}; python = {py}")
    sh(f"{py} -m pip install -q torch==2.11.0 {index}")
    sh(f"{py} -m pip install -q --no-deps -r {REPO}/requirements-gpu.lock.txt")
    sh(f"{py} -m pip install -q --no-deps -e {REPO}")
    marker.write_text(json.dumps({"driver_major": major, "python": py}))


def hf_login() -> None:
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:  # noqa: BLE001
        if "HF_TOKEN" not in os.environ:
            raise SystemExit(f"set a Kaggle secret named HF_TOKEN ({type(e).__name__})")


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    extra = " ".join(sys.argv[2:])
    OUT.mkdir(parents=True, exist_ok=True)
    unpack()
    py = python_for_env()
    install(py)
    hf_login()
    os.chdir(REPO)
    os.environ.update({"PYTHONUTF8": "1", "HF_HOME": str(WORK / "hf"), "TOKENIZERS_PARALLELISM": "false"})
    sh("nvidia-smi")
    cmds = {
        "smoke": f"{py} scripts/smoke_test.py --out {OUT}/smoke {extra}",
        "train-main": f"{py} scripts/train.py --set main --out {OUT}/main {extra}",
        "train-control": f"{py} scripts/train.py --set control --out {OUT}/control {extra}",
        "eval-dev-base": f"{py} scripts/run_eval.py --system base --split dev --out {OUT}/eval/base_dev {extra}",
        "eval-dev-adapter": f"{py} scripts/run_eval.py --system adapter --adapter {OUT}/main/adapter --split dev --out {OUT}/eval/main_dev {extra}",
        "eval-test-base": f"{py} scripts/run_eval.py --system base --split test --out {OUT}/eval/base_test {extra}",
        "eval-test-adapter": f"{py} scripts/run_eval.py --system adapter --adapter {OUT}/main/adapter --split test --out {OUT}/eval/main_test {extra}",
        "eval-test-closed": f"{py} scripts/run_eval.py --system closed_book --split test --out {OUT}/eval/closed_test {extra}",
    }
    if stage not in cmds:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(cmds)}")
    sh(cmds[stage], check=False)
    print("done; outputs in", OUT)


if __name__ == "__main__":
    main()
