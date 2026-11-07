"""Kaggle runner: unpack the bundle, build a pinned environment, run one stage.

Used by the kernels that `scripts/kaggle_launch.py` pushes. Inputs the kernel must have attached:
  - dataset  <user>/against-the-safety-bundle   (the zip built by scripts/bundle_for_kaggle.py)
  - model    cohere-labs/tiny-aya/transformers/global/1   (Tiny Aya Global; verified against the pinned HF hashes)
  - for eval stages: the kernel that produced the adapters (kernel_sources)
No Hugging Face token is needed or used: the weights come from Kaggle Models and are hash-checked.

Stages
  smoke                  compatibility test (one T4): environment, tokens, QLoRA step, batched generation, adapter reload
  train-both             main on GPU 0 and control on GPU 1 at the same time
  train-cands            second round (plan section 20): candA on GPU 0 and candB on GPU 1
  patch                  third round (plan section 24): candC on GPU 0 and candD on GPU 1, both continuing from the main adapter
                         (needs ats-train-both attached); --jobs system:split[:weight],... then evaluates them in the same kernel
  eval-jobs              third round: --jobs system:split[:weight],... over both GPUs (split: dev, test, test2 or bench_<name>)
  sample-r5              fifth round (plan section 27, priority 2): sampled rollouts on training episodes for preference pairs,
                         main on GPU 0 and candA on GPU 1 (needs the initial adapters in the bundle)
  train-dpo              fifth round: DPO on the pairs file, candG (from main) on GPU 0 and candH (from candA) on GPU 1; then --jobs
                         system:split[:weight],... for development evaluation (names candG1/candG2/candH1/candH2 = half and full epoch)
  force-r5               round 5b: the system answers at the abstention contexts of round 5 with `<answer>` forced (main on GPU 0, candA on GPU 1)
  check-system           packaging check: src/ragharness/system.py with main's adapter on sealed-test parts, to compare with the offline evaluation
  train-dpo2             round 5b: DPO on the abstention-augmented pairs, candI (from main) and candK (from the round-4 checkpoint candF1); candI1/candI2/candK1/candK2
  interp                 third round probe (plan section 25): --from main --to candD --alphas 0.2,0.35 --splits bench_r3dev:30,dev:11 builds the
                         blends (1 - alpha) * from + alpha * to inside the kernel and evaluates them
  eval-dev | eval-test | eval-test2 | eval-product
                         systems two at a time over GPU 0 and GPU 1; default base + main, then closed-book + control;
                         options in --extra: --ckpt adapter_ep1 (which saved adapter), --systems a,b,c (which systems, in this
                         order; any adapter name from an attached training kernel, e.g. candA), --bias b1,b2 (main adapter sweep)

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


def eval_jobs(py: str, spec: str, rest: str, adapters: dict[str, str]) -> int:
    """Round 3 (plan section 24): a list of system:split[:weight] jobs spread over the two GPUs.

    `split` is dev | test | test2 | a benchmark folder written as bench_<name> (for example bench_r3dev). `weight` is the expected
    cost (minutes); jobs are placed on the less loaded GPU, longest first, and each GPU runs its jobs one after the other."""
    jobs = []
    for item in spec.split(","):
        name, split, *w = item.split(":")
        weight = float(w[0]) if w else 10.0
        ev = f"{py} scripts/run_eval.py --split {split} {rest}"
        if name == "base":
            cmd = f"{ev} --system base --out {OUT}/eval/base_{split}"
        elif name == "closed":
            cmd = f"{ev} --system closed_book --out {OUT}/eval/closed_{split}"
        else:
            if f"{name}/adapter" not in adapters:
                raise SystemExit(f"adapter {name}/adapter not found (have {sorted(adapters)})")
            cmd = f"{ev} --system adapter --adapter {adapters[f'{name}/adapter']} --out {OUT}/eval/{name}_{split}"
        jobs.append((weight, cmd))
    load, chains = {0: 0.0, 1: 0.0}, {0: [], 1: []}
    for weight, cmd in sorted(jobs, key=lambda j: -j[0]):
        g = min(load, key=lambda k: load[k])
        chains[g].append(cmd)
        load[g] += weight
    print("expected minutes per GPU:", load, flush=True)
    return run_parallel([(" && ".join(c), {"CUDA_VISIBLE_DEVICES": str(g)}, f"eval_jobs_gpu{g}.log") for g, c in chains.items() if c])


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
    adapters.update({f"{p.parent.parent.name}/{p.parent.name}": str(p.parent) for p in (REPO / "initial_adapters").rglob("adapter_config.json")})
    for name in ("candE", "candF"):
        if f"{name}/adapter_ep1" in adapters:
            adapters[f"{name}1/adapter"] = adapters[f"{name}/adapter_ep1"]
        if f"{name}/adapter" in adapters:
            adapters[f"{name}2/adapter"] = adapters[f"{name}/adapter"]
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
    elif stage == "train-cands":  # second round (plan section 20): candA on GPU 0, candB on GPU 1
        rc = run_parallel([
            (f"{py} scripts/train.py --set candA --out {OUT}/candA {extra}", g0, "train_candA.log"),
            (f"{py} scripts/train.py --set candB --out {OUT}/candB {extra}", g1, "train_candB.log"),
        ])
    elif stage in ("evidence-smoke", "train-evidence"):
        smoke = stage == "evidence-smoke"
        names = (("candE", "main", g0), ("candF", "candA", g1))
        options = "--epochs 1 --max-steps 2 --limit-rows 32" if smoke else "--epochs 2"
        for _, initial, _ in names:
            if f"{initial}/adapter" not in adapters:
                raise SystemExit(f"initial adapter missing: {initial}")
        rc = run_parallel([
            (f"{py} scripts/train.py --set {name} --out {OUT}/{name} --init-adapter {adapters[f'{initial}/adapter']} "
             f"--lr 3e-5 --seed 47 --max-length 2560 {options} {extra}", gpu, f"train_{name}.log")
            for name, initial, gpu in names
        ])
        if rc:
            raise SystemExit(rc)
        for name, _, _ in names:
            report = json.loads((OUT / name / "train_report.json").read_text())
            if not __import__('math').isfinite(report["train_loss"]):
                raise SystemExit(f"non-finite training loss: {name}")
            adapters[f"{name}/adapter"] = str(OUT / name / "adapter")
            if (OUT / name / "adapter_ep1").exists():
                adapters[f"{name}1/adapter"] = str(OUT / name / "adapter_ep1")
            adapters[f"{name}2/adapter"] = str(OUT / name / "adapter")
        if smoke:
            spec = "candE:bench_r4dev:1,candF:bench_r4dev:1"
            rc = eval_jobs(py, spec, "--parse lenient --limit-questions 2 --extra-questions 0", adapters)
    elif stage == "patch":  # round 3 (plan section 24): candC on GPU 0 and candD on GPU 1, both continuing from the main adapter; then --jobs
        m = re.search(r"--jobs\s+(\S+)", extra)
        jobs_spec, rest = (m.group(1), extra.replace(m.group(0), "")) if m else ("", extra)
        m = re.search(r"--eval-args\s+\"([^\"]*)\"", rest)
        eval_args, rest = (m.group(1), rest.replace(m.group(0), "")) if m else ("--parse lenient --extra-questions 0", rest)
        if "main/adapter" not in adapters:
            raise SystemExit("attach the kernel that trained the main adapter (ats-train-both)")
        init = adapters["main/adapter"]
        rc = run_parallel([
            (f"{py} scripts/train.py --set candC --out {OUT}/candC --init-adapter {init} {rest}", g0, "train_candC.log"),
            (f"{py} scripts/train.py --set candD --out {OUT}/candD --init-adapter {init} {rest}", g1, "train_candD.log"),
        ])
        for n in ("candC", "candD"):
            if (OUT / n / "adapter" / "adapter_config.json").exists():
                adapters[f"{n}/adapter"] = str(OUT / n / "adapter")
        if jobs_spec:
            rc = max(rc, eval_jobs(py, jobs_spec, eval_args, adapters))
    elif stage == "interp":  # round 3 probe (plan section 25): blends of two adapters, evaluated at once
        m = {k: re.search(rf"--{k}\s+(\S+)", extra) for k in ("from", "to", "alphas", "splits")}
        if not all(m.values()):
            raise SystemExit("interp needs --from NAME --to NAME --alphas a,b,c --splits split:weight,...")
        src, dst, alphas, splits = (m[k].group(1) for k in ("from", "to", "alphas", "splits"))
        rest = extra
        for v in m.values():
            rest = rest.replace(v.group(0), "")
        jobs = []
        for a in alphas.split(","):
            name = f"{dst}_a{int(round(float(a) * 100)):02d}"
            sh(f"{py} scripts/interp_adapters.py --a {adapters[f'{src}/adapter']} --b {adapters[f'{dst}/adapter']} --alpha {a} --out {OUT}/interp/{name}/adapter")
            adapters[f"{name}/adapter"] = str(OUT / "interp" / name / "adapter")
            jobs += [f"{name}:{s.split(':')[0]}:{s.split(':')[1]}" for s in splits.split(",")]
        rc = eval_jobs(py, ",".join(jobs), rest.strip() or "--parse lenient --extra-questions 0", adapters)
    elif stage == "sample-r5":  # round 5 (plan section 27, priority 2): sampled rollouts on training episodes, main on GPU 0, candA on GPU 1
        for n in ("main", "candA"):
            if f"{n}/adapter" not in adapters:
                raise SystemExit(f"initial adapter missing: {n}")
        rc = run_parallel([
            (f"{py} scripts/sample_round5.py --adapter {adapters[f'{n}/adapter']} --tag {n} --out {OUT}/samples/{n} --seed {seed} {extra}", gpu, f"sample_{n}.log")
            for n, gpu, seed in (("main", g0, 5301), ("candA", g1, 5302))
        ])
    elif stage == "train-dpo":  # round 5 (plan section 27, priority 2): candG from main on GPU 0 and candH from candA on GPU 1, same pairs; then dev evaluation
        m = re.search(r"--jobs\s+(\S+)", extra)
        jobs_spec, rest = (m.group(1), extra.replace(m.group(0), "")) if m else ("", extra)
        pairs_file = REPO / "data" / "processed" / "training" / "dpo_r5.jsonl"
        if not pairs_file.exists():
            raise SystemExit("the preference-pair file is missing from the bundle")
        names = (("candG", "main", g0), ("candH", "candA", g1))
        for _, initial, _ in names:
            if f"{initial}/adapter" not in adapters:
                raise SystemExit(f"initial adapter missing: {initial}")
        rc = run_parallel([
            (f"{py} scripts/train_dpo.py --init-adapter {adapters[f'{initial}/adapter']} --pairs {pairs_file} --out {OUT}/{name} --name {name} {rest}",
             gpu, f"train_{name}.log")
            for name, initial, gpu in names
        ])
        if rc:
            raise SystemExit(rc)
        for name, _, _ in names:
            report = json.loads((OUT / name / "train_report.json").read_text())
            adapters[f"{name}2/adapter"] = str(OUT / name / "adapter")
            if (OUT / name / "adapter_half").exists():
                adapters[f"{name}1/adapter"] = str(OUT / name / "adapter_half")
            print(name, "trained", {k: report[k] for k in ("pairs_trained", "steps", "wall_seconds", "seconds_per_pair_pass", "peak_gpu_memory_gb")}, flush=True)
        if jobs_spec:
            rc = eval_jobs(py, jobs_spec, "--parse lenient --extra-questions 0", adapters)
            if rc == 0:  # priority 1: the checker V on every candidate's development answers (main's adapter, wording A), for selection with V on both sides
                cands = ",".join(f"{n}{i}" for n, _, _ in names for i in (1, 2))
                sh(f"{py} scripts/build_verifier_cases.py --eval {OUT}/eval --system {cands} --out {OUT}/verify/cases_candidates.jsonl", check=False)
                sh(f"{py} scripts/verify_round5.py --cases {OUT}/verify/cases_candidates.jsonl --tag main --adapter {adapters['main/adapter']} "
                   f"--out {OUT}/verify/main --wordings A", check=False, env=g0)
    elif stage == "force-r5":  # round 5b: forced answers at the abstention contexts of each system, main on GPU 0 and candA on GPU 1
        contexts = REPO / "data" / "processed" / "training" / "force_contexts_r5.jsonl"
        for n in ("main", "candA"):
            if f"{n}/adapter" not in adapters:
                raise SystemExit(f"initial adapter missing: {n}")
        if not contexts.exists():
            raise SystemExit("the context file is missing from the bundle")
        rc = run_parallel([
            (f"{py} scripts/force_answers_round5.py --adapter {adapters[f'{n}/adapter']} --tag {n} --contexts {contexts} --out {OUT}/forced --seed {seed} {extra}",
             gpu, f"force_{n}.log")
            for n, gpu, seed in (("main", g0, 5303), ("candA", g1, 5304))
        ])
    elif stage == "train-dpo2":  # round 5b: candI from main on GPU 0 and candJ from candA on GPU 1 on the abstention-augmented pair file; then dev evaluation and the checker
        m = re.search(r"--jobs\s+(\S+)", extra)
        jobs_spec, rest = (m.group(1), extra.replace(m.group(0), "")) if m else ("", extra)
        pairs_file = REPO / "data" / "processed" / "training" / "dpo_r5b.jsonl"
        if not pairs_file.exists():
            raise SystemExit("the round 5b preference-pair file is missing from the bundle")
        names = (("candI", "main", g0), ("candK", "candF1", g1))
        for _, initial, _ in names:
            if f"{initial}/adapter" not in adapters:
                raise SystemExit(f"initial adapter missing: {initial}")
        rc = run_parallel([
            (f"{py} scripts/train_dpo.py --init-adapter {adapters[f'{initial}/adapter']} --pairs {pairs_file} --out {OUT}/{name} --name {name} {rest}",
             gpu, f"train_{name}.log")
            for name, initial, gpu in names
        ])
        if rc:
            raise SystemExit(rc)
        for name, _, _ in names:
            report = json.loads((OUT / name / "train_report.json").read_text())
            adapters[f"{name}2/adapter"] = str(OUT / name / "adapter")
            if (OUT / name / "adapter_half").exists():
                adapters[f"{name}1/adapter"] = str(OUT / name / "adapter_half")
            print(name, "trained", {k: report[k] for k in ("pairs_trained", "steps", "wall_seconds", "seconds_per_pair_pass", "peak_gpu_memory_gb")}, flush=True)
        if jobs_spec:
            rc = eval_jobs(py, jobs_spec, "--parse lenient --extra-questions 0", adapters)
            if rc == 0:
                cands = ",".join(f"{n}{i}" for n, _, _ in names for i in (1, 2))
                sh(f"{py} scripts/build_verifier_cases.py --eval {OUT}/eval --system {cands} --out {OUT}/verify/cases_candidates.jsonl", check=False)
                sh(f"{py} scripts/verify_round5.py --cases {OUT}/verify/cases_candidates.jsonl --tag main --adapter {adapters['main/adapter']} "
                   f"--out {OUT}/verify/main --wordings A", check=False, env=g0)
    elif stage == "check-system":  # round 5 packaging: the packaged system (src/ragharness/system.py) with main's adapter on sealed-test parts that main already ran on
        m = re.search(r"--benches\s+(\S+)", extra)
        benches = (m.group(1) if m else "r5t_hotpot:0,r5t_rgb:100").split(",")
        if "main/adapter" not in adapters:
            raise SystemExit("check-system needs the main adapter")
        rc = run_parallel([
            (f"{py} scripts/check_system_round5.py --adapter {adapters['main/adapter']} --bench {b.split(':')[0]} --limit {b.split(':')[1] if ':' in b else 0} --out {OUT}/system_check",
             {"CUDA_VISIBLE_DEVICES": str(i % 2)}, f"check_{b.split(':')[0]}.log")
            for i, b in enumerate(benches)
        ])
    elif stage == "verify-main-r5":  # round 5: the selected checker (main's adapter, wording A) on another cases file, --cases <name under data/processed/verifier>
        m = re.search(r"--cases\s+(\S+)", extra)
        cases = REPO / "data" / "processed" / "verifier" / (m.group(1) if m else "cases_dev.jsonl")
        if "main/adapter" not in adapters or not cases.exists():
            raise SystemExit("verify-main-r5 needs the main adapter and the cases file in the bundle")
        rc = run_parallel([(f"{py} scripts/verify_round5.py --cases {cases} --tag main --adapter {adapters['main/adapter']} --out {OUT}/verify/main --wordings A", g0, "verify_main.log")])
    elif stage == "verify-r5":  # round 5 (plan section 27, priority 1): the answer checker, base on GPU 0 and main's adapter on GPU 1
        cases = REPO / "data" / "processed" / "verifier" / "cases_dev.jsonl"
        if "main/adapter" not in adapters or not cases.exists():
            raise SystemExit("verify-r5 needs the main adapter and the verifier cases in the bundle")
        rc = run_parallel([
            (f"{py} scripts/verify_round5.py --cases {cases} --tag base --out {OUT}/verify/base {extra}", g0, "verify_base.log"),
            (f"{py} scripts/verify_round5.py --cases {cases} --tag main --adapter {adapters['main/adapter']} --out {OUT}/verify/main {extra}", g1, "verify_main.log"),
        ])
    elif stage == "eval-jobs":  # round 3: any list of system:split[:weight] jobs
        m = re.search(r"--jobs\s+(\S+)", extra)
        if not m:
            raise SystemExit("eval-jobs needs --jobs system:split[:weight],...")
        rest = extra.replace(m.group(0), "").strip() or "--parse lenient --extra-questions 0"
        rc = eval_jobs(py, m.group(1), rest, adapters)
    elif stage == "brightpro":
        m = re.search(r"--systems\s+(\S+)", extra)
        if not m:
            raise SystemExit("brightpro needs --systems name,name")
        if not (REPO / "data/brightpro/robotics/manifest.json").exists():
            raise SystemExit("the evaluation bundle must include the frozen Bright-Pro corpus")
        wanted = m.group(1).split(",")
        if len(wanted) > 2:
            raise SystemExit("the fixed Bright-Pro pilot permits at most two systems")
        rc = run_parallel([
            (f"{py} scripts/run_brightpro_pilot.py --adapter {adapters[f'{name}/adapter']} --out {OUT}/brightpro/{name}",
             {"CUDA_VISIBLE_DEVICES": str(i)}, f"brightpro_{name}.log") for i, name in enumerate(wanted)
        ])
    elif stage == "eval-bench":  # external benchmarks (plan section 22): each system runs every benchmark in turn on one GPU
        m = re.search(r"--benches\s+(\S+)", extra)
        benches = m.group(1).split(",") if m else ["squad2", "rgb", "rgb_int", "2wiki", "musique"]
        rest = extra.replace(m.group(0), "") if m else extra
        m = re.search(r"--systems\s+(\S+)", rest)
        systems = m.group(1).split(",") if m else ["base", "main"]
        rest = rest.replace(m.group(0), "") if m else rest

        def cmd(name: str, b: str) -> str:
            ev = f"{py} scripts/run_eval.py --split bench_{b} {rest}"
            if name == "base":
                return f"{ev} --system base --out {OUT}/eval/base_bench_{b}"
            if name == "closed":
                return f"{ev} --system closed_book --out {OUT}/eval/closed_bench_{b}"
            key = f"{name}/adapter"
            if key not in adapters:
                raise SystemExit(f"adapter {key!r} not found under /kaggle/input (have {sorted(adapters)})")
            return f"{ev} --system adapter --adapter {adapters[key]} --out {OUT}/eval/{name}_bench_{b}"

        load = {0: 0.0, 1: 0.0}
        chains: dict[int, list[str]] = {0: [], 1: []}
        for name in systems:  # greedy balance; closed book is short
            g = min(load, key=lambda k: load[k])
            chains[g] += [cmd(name, b) for b in benches]
            load[g] += 0.1 if name == "closed" else 1.0
        rc = run_parallel([(" ; ".join(c), {"CUDA_VISIBLE_DEVICES": str(g)}, f"eval_bench_gpu{g}.log") for g, c in chains.items() if c])
    elif stage in ("eval-dev", "eval-test", "eval-test2", "eval-product"):
        split = stage.split("-")[1]
        ckpt, wanted, rest = "adapter", ["base", "main", "control"] + ([] if split == "product" else ["closed"]), extra
        m = re.search(r"--ckpt\s+(\S+)", rest)
        if m:  # which saved adapter of each training run to evaluate: adapter (final) or adapter_ep1, ...
            ckpt, rest = m.group(1), rest.replace(m.group(0), "")
        m_systems = re.search(r"--systems\s+(\S+)", rest)
        if m_systems:
            wanted, rest = m_systems.group(1).split(","), rest.replace(m_systems.group(0), "")
        tag = "" if ckpt == "adapter" else f"_{ckpt}"
        ev = f"{py} scripts/run_eval.py --split {split} {rest}"

        def adapter_for(name: str) -> str:
            key = f"{name}/{ckpt}"
            if key not in adapters:
                raise SystemExit(f"adapter {key!r} not found under /kaggle/input (have {sorted(adapters)}); attach the training kernel")
            return adapters[key]

        m = re.search(r"--bias\s+(\S+)", rest)
        if m:  # accuracy-coverage sweep: the main adapter at several abstain biases, split over both GPUs
            biases = m.group(1).split(",")
            ev = f"{py} scripts/run_eval.py --split {split} {rest.replace(m.group(0), '')}"
            chains: dict[int, list[str]] = {0: [], 1: []}
            for i, b in enumerate(biases):
                chains[i % 2].append(f"{ev} --system adapter --adapter {adapter_for('main')} --abstain-bias {b} --out {OUT}/eval/main{tag}_b{b}_{split}")
            rc = run_parallel([(" && ".join(c), {"CUDA_VISIBLE_DEVICES": str(g)}, f"eval_main{tag}_bias_gpu{g}_{split}.log") for g, c in chains.items() if c])
            wanted = []

        # Systems run two at a time (GPU 0, GPU 1) in the order given; default order keeps the first round's pairing.
        def command(name: str) -> str:
            if name == "base":
                return f"{ev} --system base --out {OUT}/eval/base_{split}"
            if name == "closed":
                return f"{ev} --system closed_book --out {OUT}/eval/closed_{split}"
            return f"{ev} --system adapter --adapter {adapter_for(name)} --out {OUT}/eval/{name}{tag}_{split}"

        order = [n for n in ("base", "main", "closed", "control") if n in wanted] if not m_systems else wanted
        jobs_all = [(command(n), f"eval_{n}{tag}_{split}.log") for n in order]
        for i in range(0, len(jobs_all), 2):
            wave = [(cmd, env, log) for (cmd, log), env in zip(jobs_all[i:i + 2], (g0, g1))]
            rc = max(rc, run_parallel(wave))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    print("done; outputs in", OUT, "exit", rc)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
