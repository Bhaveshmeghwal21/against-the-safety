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
    source = os.environ.get("ATS_KAGGLE_CREDENTIALS_ENV")
    config_dir = ROOT / ".cache" / "kaggle"
    if source:
        value = os.environ.get(source)
        if not value and os.name == "nt":
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value = winreg.QueryValueEx(key, source)[0]
        if not value:
            raise SystemExit(f"credential variable {source!r} is empty")
        value = value.strip().strip('"')
        if value.startswith("{"):
            kj = json.loads(value)
        else:
            path = Path(value).expanduser()
            if not path.is_file():
                raise SystemExit("the credential variable must contain a kaggle.json path or credential JSON")
            kj = json.loads(path.read_text(encoding="utf-8-sig"))
        config_dir = config_dir / kj["username"]
    else:
        kj = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text(encoding="utf-8"))
    return {"KAGGLE_USERNAME": kj["username"], "KAGGLE_KEY": kj["key"], "KAGGLE_CONFIG_DIR": str(config_dir)}


def kaggle(*args: str, check: bool = True) -> str:
    import shutil

    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle"))
    env = {**os.environ, **creds(), "PYTHONUTF8": "1"}
    temp = ROOT / ".cache" / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env.update(TMP=str(temp), TEMP=str(temp), TMPDIR=str(temp))
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_FILE"):
        env.pop(key, None)  # use the explicitly selected legacy account, not a different inherited login
    p = subprocess.run([exe, *args], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        useful = [line for line in out.splitlines() if not ("%|" in line and "B/s" in line)]
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n" + "\n".join(useful)[-6000:])
    return out


def username() -> str:
    return creds()["KAGGLE_USERNAME"]


def make_dataset(existing_bundle: bool = False) -> None:
    if not existing_bundle:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "bundle_for_kaggle.py")], check=True, env={**os.environ, "PYTHONUTF8": "1"})
    elif not (BUNDLE / "against-the-safety.zip").is_file():
        raise SystemExit("the prepared bundle is missing")
    meta = {"title": "against-the-safety-bundle", "id": f"{username()}/{SLUG}", "licenses": [{"name": "CC-BY-SA-4.0"}]}
    (BUNDLE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    status = kaggle("datasets", "status", f"{username()}/{SLUG}", check=False).lower()  # `datasets list -s` is a search index and can lag
    exists = any(w in status for w in ("ready", "pending", "processing"))
    if exists:
        print(kaggle("datasets", "version", "-p", str(BUNDLE), "-m", f"bundle {time.strftime('%Y-%m-%d %H:%M')}", "--dir-mode", "zip", "--quiet"))
    else:
        print(kaggle("datasets", "create", "-p", str(BUNDLE), "--dir-mode", "zip", "--quiet"))  # private by default
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
    if not tmp.resolve().is_relative_to(ROOT.resolve()):
        raise SystemExit("probe directory resolves outside the workspace")
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


def run_stage(stage: str, adapters_from: str | None, extra: str, tag: str = "", timeout: int | None = None) -> str:
    if served_bundle_id() != bundle_id():
        raise SystemExit("the dataset on Kaggle is not the local bundle; run `kaggle_launch.py dataset` first (it waits until the new version is served)")
    slug = f"ats-{stage}" + (f"-{tag}" if tag else "")
    kdir = ROOT / "runs" / "kaggle" / f"push-{slug}"
    kdir.mkdir(parents=True, exist_ok=True)
    code = (
        "import pathlib, subprocess, sys, time\n"
        "p = next(pathlib.Path('/kaggle/input').rglob('bootstrap.py'))\n"
        f"expect = {bundle_id()!r}\n"
        "idf = p.parent.parent / 'BUNDLE_ID'\n"
        "got = idf.read_text().strip() if idf.exists() else 'missing'\n"
        "if got != expect:\n"
        "    raise SystemExit(f'stale dataset mount: this kernel was launched for bundle {expect}, the mounted one is {got}')\n"
        "job = subprocess.Popen([sys.executable, str(p), " + repr(stage) + (", " + repr(extra) if extra else "") + "])\n"
        "sizes = {}\n"
        "while job.poll() is None:\n"
        "    time.sleep(30)\n"
        "    for log in pathlib.Path('/kaggle/working/out').glob('*.log'):\n"
        "        size = log.stat().st_size\n"
        "        if sizes.get(log.name) == size:\n"
        "            continue\n"
        "        sizes[log.name] = size\n"
        "        with log.open('rb') as stream:\n"
        "            stream.seek(max(0, size - 2048))\n"
        "            tail = stream.read().decode('utf-8', errors='replace').splitlines()[-3:]\n"
        "        print('[progress]', log.name, '\\n'.join(tail), flush=True)\n"
        "raise SystemExit(job.returncode)\n"
    )
    (kdir / "run.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{username()}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{username()}/{SLUG}"], "competition_sources": [],
        "kernel_sources": [f"{username()}/{s}" for s in adapters_from.split(",")] if adapters_from else [],
        "model_sources": [MODEL_SOURCE],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    push_args = ["kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4"]
    if timeout is not None:
        push_args += ["--timeout", str(timeout)]
    print(kaggle(*push_args))
    return slug


def main() -> None:
    global BUNDLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--credentials-env", help="process/Windows user variable containing a kaggle.json path or JSON")
    ap.add_argument("--bundle-dir", help="use an already prepared, isolated bundle directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dataset_parser = sub.add_parser("dataset")
    dataset_parser.add_argument("--existing", action="store_true", help="upload the already prepared bundle, including explicit adapters")
    r = sub.add_parser("run")
    r.add_argument("stage")
    r.add_argument("--adapters-from", default=None, help="training kernel slug(s) whose outputs are attached, comma separated")
    r.add_argument("--extra", default="")
    r.add_argument("--tag", default="", help="suffix for the kernel slug, so two runs of one stage do not overwrite each other")
    r.add_argument("--timeout", type=int, default=None, help="maximum remote runtime in seconds")
    s = sub.add_parser("status")
    s.add_argument("slug")
    o = sub.add_parser("output")
    o.add_argument("slug")
    a = ap.parse_args()
    if a.credentials_env:
        os.environ["ATS_KAGGLE_CREDENTIALS_ENV"] = a.credentials_env
    if a.bundle_dir:
        BUNDLE = Path(a.bundle_dir).resolve()
    if a.cmd == "dataset":
        make_dataset(a.existing)
    elif a.cmd == "run":
        print("kernel slug:", run_stage(a.stage, a.adapters_from, a.extra, a.tag, a.timeout))
    elif a.cmd == "status":
        print(kaggle("kernels", "status", f"{username()}/{a.slug}"))
    elif a.cmd == "output":
        dest = ROOT / "runs" / "kaggle" / a.slug
        dest.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", f"{username()}/{a.slug}", "-p", str(dest)))


if __name__ == "__main__":
    main()
