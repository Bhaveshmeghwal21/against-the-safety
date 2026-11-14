"""Zip the code and the exact built data so Kaggle runs on identical inputs without cloning the private repo.

Run:  python scripts/bundle_for_kaggle.py
Then upload runs/kaggle_bundle/ as a private Kaggle Dataset (kaggle datasets create -p runs/kaggle_bundle, or the web UI)
and attach it to the notebook. The bundle holds derived HotpotQA data (CC BY-SA 4.0): keep the dataset private
until the project is published.
"""

import hashlib
import json
import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "kaggle_bundle"
INCLUDE_DIRS = ["src", "scripts", "tests", "kaggle", "data/manifests", "data/processed", "data/index", "data/product", "data/bench", "reports", "examples"]
INCLUDE_FILES = ["pyproject.toml", "requirements-gpu.lock.txt", "requirements-cpu.lock.txt", "README.md"]
SKIP = ("__pycache__", ".pyc", "tests/fixtures", "trajectories.sqlite")
SEALED_REPORTS = {"reports/round6_test_manifest.json", "reports/round6_allocation.json"}


def permitted_path(rel, development_only):
    """Exclude actual Round 6 test inputs and their allocation from a dev bundle."""
    rel = Path(rel)
    if not development_only:
        return True
    return not (len(rel.parts) >= 3 and rel.parts[:2] == ('data', 'bench') and rel.parts[2].startswith('r6t_')) \
        and rel.as_posix() not in SEALED_REPORTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--adapter", action="append", default=[], help="NAME=local adapter directory; credentials are never included")
    ap.add_argument("--brightpro", action="store_true", help="include the optional evaluation-only full robotics corpus")
    ap.add_argument('--development-only', action='store_true', help='exclude Round 6 sealed-test inputs and restrict permitted jobs')
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError('Bundle output must stay within the workspace')
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "against-the-safety.zip"
    files = [ROOT / f for f in INCLUDE_FILES]
    if args.development_only:
        files.append(ROOT / 'research/2026-11-14-round6-preregistration.md')
    for d in INCLUDE_DIRS + (["data/brightpro"] if args.brightpro else []):
        base = ROOT / d
        if args.development_only and d == 'data/bench':
            # Do not descend into a sealed-test directory at all.
            for child in base.iterdir():
                if child.name.startswith('r6t_'):
                    continue
                files.extend(p for p in (child.rglob('*') if child.is_dir() else [child]) if p.is_file())
        else:
            files += [p for p in base.rglob("*") if p.is_file()]
    files = [p for p in files if not any(s in p.as_posix() for s in SKIP)
             and permitted_path(p.relative_to(ROOT), args.development_only)]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        content = hashlib.sha256()
        for p in sorted(files):
            rel = p.relative_to(ROOT).as_posix()
            z.write(p, rel)
            content.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
        for spec in args.adapter:
            name, path = spec.split("=", 1)
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("invalid adapter name")
            source = Path(path)
            for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                p = source / filename
                if not p.is_file():
                    raise FileNotFoundError(f"adapter {name} is missing {filename}")
                rel = f"initial_adapters/{name}/adapter/{filename}"
                z.write(p, rel)
                content.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
        if args.development_only:
            marker = b'Only the preregistered development jobs are permitted.\n'
            z.writestr('DEVELOPMENT_ONLY', marker)
            content.update(b'DEVELOPMENT_ONLY\0' + hashlib.sha256(marker).digest())
        # the kernel compares this id with the one it was launched with, so a stale dataset mount fails fast
        z.writestr("BUNDLE_ID", content.hexdigest()[:20])
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    index = json.loads((ROOT / "data" / "manifests" / "index_manifest.json").read_text(encoding="utf-8"))
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "against-the-safety-bundle", "id": "REPLACE_WITH_YOUR_KAGGLE_USERNAME/against-the-safety-bundle",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }, indent=2), encoding="utf-8")
    print(f"{zpath}  {zpath.stat().st_size / 2**20:.1f} MB  files={len(files)}  sha256={digest[:16]}  index_hash={index['index_hash'][:12]}")


if __name__ == "__main__":
    main()
