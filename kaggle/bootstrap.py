"""Kaggle runner: unpack the bundle, build a pinned environment, run one stage.

Used by the kernels that `scripts/kaggle_launch.py` pushes. Inputs the kernel must have attached:
  - dataset  <user>/against-the-safety-bundle   (the zip built by scripts/bundle_for_kaggle.py)
  - model    cohere-labs/tiny-aya/transformers/global/1   (Tiny Aya Global; verified against the pinned HF hashes)
  - for eval stages: the kernel that produced the adapters (kernel_sources)
No Hugging Face token is needed or used: the weights come from Kaggle Models and are hash-checked.

Stages
  smoke                  compatibility test (one T4): environment, tokens, QLoRA step, batched generation, adapter reload
  train-both             main on GPU 0 and control on GPU 1 at the same time
  eval-dev | eval-test | eval-product   base (GPU 0) and main adapter (GPU 1), then closed-book (GPU 0) and control adapter (GPU 1);
                         options in --extra: --ckpt adapter_ep1 (which saved adapter), --systems main,control (subset)

The repo and the Python environment live in /kaggle/temp so they are not saved as notebook output; only /kaggle/working/out is.
"""

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

TMP = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
REPO = TMP / "repo"
OUT = Path("/kaggle/working/out")
EXTRA_PIP: list[str] = []  # nothing beyond the lock file


def sh(cmd: str, check: bool = True, env: dict | None = None) -> str:
    print("$", cmd, flush=True)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, env={**os.environ, **(env or {})})
    print((p.stdout + p.stderr)[-3500:], flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {cmd}")
    return p.stdout


def driver_major() -> int:
    out = subprocess.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader", shell=True, text=True, capture_output=True).stdout
    m = re.match(r"(\d+)", out.strip())
    return int(m.group(1)) if m else 0


def unpack() -> None:
    """Copy the attached bundle dataset to writable scratch space (Kaggle auto-extracts uploaded zips)."""
    import shutil

    if REPO.exists():
        return
    roots = [p.parent for p in Path("/kaggle/input").rglob("pyproject.toml") if (p.parent / "src" / "ragharness").exists()]
    if roots:
        shutil.copytree(roots[0], REPO)
        return
    zips = list(Path("/kaggle/input").rglob("against-the-safety.zip"))
    if not zips:
        raise SystemExit("attach the bundle dataset (against-the-safety-bundle) to this kernel")
    REPO.mkdir(parents=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(REPO)


def find_model() -> str:
    for cand in Path("/kaggle/input").rglob("model.safetensors.index.json"):
        if "tiny-aya" in cand.as_posix() and "global" in cand.as_posix():
            return str(cand.parent)
    raise SystemExit("attach the model cohere-labs/tiny-aya/transformers/global/1 to this kernel")


def has_fts5(py: str) -> bool:
    code = "import sqlite3; sqlite3.connect(':memory:').execute('create virtual table t using fts5(a)')"
    return subprocess.run([py, "-c", code], capture_output=True).returncode == 0


def python_for_env() -> str:
    """AgentTune needs Python 3.12+ and its retrieval backend needs SQLite FTS5.

    Kaggle's /usr/bin/python3 (3.13) passes a fresh-process FTS5 check, yet inside the training process (after torch and
    the other libraries are imported) its sqlite3 reports "no such module: fts5", presumably because another shared
    library supplies libsqlite3. AgentTune's backend turns that error into "no results", so every question looked
    unreachable. A uv-managed Python 3.12 (python-build-standalone) links SQLite with FTS5 statically, and it is the same
    minor version as the local environment, so the kernel always uses it.
    """
    venv = TMP / "venv312"
    if not (venv / "bin" / "python").exists():
        sh("pip install -q uv")
        sh(f"uv venv --python 3.12 --seed {venv}")
    py = str(venv / "bin" / "python")
    if not has_fts5(py):
        raise SystemExit(f"{py} has no SQLite FTS5; retrieval would silently return nothing")
    return py


def install(py: str) -> None:
    marker = TMP / ".installed"
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


def run_parallel(jobs: list[tuple[str, dict, str]]) -> int:
    """jobs: (command, extra env, log name). All start together; returns the worst exit code."""
    procs = []
    for cmd, env, log in jobs:
        f = open(OUT / log, "w")
        print("$ (parallel)", env, cmd, flush=True)
        procs.append((subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **env}), f, log))
    worst = 0
    for p, f, log in procs:
        rc = p.wait()
        f.close()
        print(f"{log}: exit {rc}; tail:\n" + (OUT / log).read_text()[-1500:], flush=True)
        worst = max(worst, rc)
    return worst


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    extra = " ".join(sys.argv[2:])
    OUT.mkdir(parents=True, exist_ok=True)
    unpack()
    model = find_model()
    py = python_for_env()
    install(py)
    os.chdir(REPO)
    base_env = {"PYTHONUTF8": "1", "HF_HOME": str(TMP / "hf"), "TOKENIZERS_PARALLELISM": "false", "TINY_AYA_PATH": model}
    os.environ.update(base_env)
    sh("nvidia-smi")
    adapters = {f"{p.parent.parent.name}/{p.parent.name}": str(p.parent) for p in Path("/kaggle/input").rglob("adapter_config.json")}
    print("adapters found:", adapters)

    g0, g1 = {"CUDA_VISIBLE_DEVICES": "0"}, {"CUDA_VISIBLE_DEVICES": "1"}
    rc = 0
    if stage == "smoke":
        sh(f"{py} scripts/smoke_test.py --out {OUT}/smoke {extra}", check=False)
    elif stage == "train-both":
        rc = run_parallel([
            (f"{py} scripts/train.py --set main --out {OUT}/main {extra}", g0, "train_main.log"),
            (f"{py} scripts/train.py --set control --out {OUT}/control {extra}", g1, "train_control.log"),
        ])
    elif stage in ("eval-dev", "eval-test", "eval-product"):
        split = stage.split("-")[1]
        ckpt, wanted, rest = "adapter", ["base", "main", "control"] + ([] if split == "product" else ["closed"]), extra
        m = re.search(r"--ckpt\s+(\S+)", rest)
        if m:  # which saved adapter of each training run to evaluate: adapter (final) or adapter_ep1, ...
            ckpt, rest = m.group(1), rest.replace(m.group(0), "")
        m = re.search(r"--systems\s+(\S+)", rest)
        if m:
            wanted, rest = m.group(1).split(","), rest.replace(m.group(0), "")
        tag = "" if ckpt == "adapter" else f"_{ckpt}"
        ev = f"{py} scripts/run_eval.py --split {split} {rest}"

        def adapter_for(name: str) -> str:
            key = f"{name}/{ckpt}"
            if key not in adapters:
                raise SystemExit(f"adapter {key!r} not found under /kaggle/input (have {sorted(adapters)}); attach the training kernel")
            return adapters[key]

        waves = [
            [("base", g0, f"{ev} --system base --out {OUT}/eval/base_{split}"),
             ("main", g1, f"{ev} --system adapter --adapter {{main}} --out {OUT}/eval/main{tag}_{split}")],
            [("closed", g0, f"{ev} --system closed_book --out {OUT}/eval/closed_{split}"),
             ("control", g1, f"{ev} --system adapter --adapter {{control}} --out {OUT}/eval/control{tag}_{split}")],
        ]
        for wave in waves:
            jobs = []
            for name, env, cmd in wave:
                if name in wanted:
                    if name in ("main", "control"):
                        cmd = cmd.replace("{" + name + "}", adapter_for(name))
                    jobs.append((cmd, env, f"eval_{name}{tag}_{split}.log"))
            if jobs:
                rc = max(rc, run_parallel(jobs))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    print("done; outputs in", OUT, "exit", rc)


if __name__ == "__main__":
    main()
