"""Where the Tiny Aya Global weights come from, and a check that a local copy matches the pinned Hugging Face revision.

Order of preference: $TINY_AYA_PATH, the Kaggle Models copy attached to a notebook, then the gated Hub repo at the
pinned revision. The adapter is always published against the Hub repo; the Kaggle copy is only a download shortcut,
so `verify_local` compares file hashes with the Hub's published LFS hashes before any result is trusted.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

HF_ID = "CohereLabs/tiny-aya-global"
HF_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
KAGGLE_PATHS = ("/kaggle/input/tiny-aya/transformers/global/1", "/kaggle/input/models/cohere-labs/tiny-aya/transformers/global/1")


def resolve() -> tuple[str, str | None, str]:
    """(model path or id, revision for from_pretrained, human-readable source label)."""
    override = os.environ.get("TINY_AYA_PATH")
    if override and Path(override).exists():
        return override, None, f"local:{override}"
    for p in KAGGLE_PATHS:
        if Path(p).exists():
            return p, None, f"kaggle-models:{p}"
    return HF_ID, HF_REVISION, f"hf:{HF_ID}@{HF_REVISION}"


def hub_file_hashes(token: str | None = None) -> dict[str, str]:
    """Published sha256 of every LFS file at the pinned revision (needs no gate for metadata)."""
    req = urllib.request.Request(f"https://huggingface.co/api/models/{HF_ID}/revision/{HF_REVISION}?blobs=true")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.load(r)
    return {s["rfilename"]: s["lfs"]["sha256"] for s in info.get("siblings", []) if s.get("lfs")}


def sha256_file(path: Path, chunk: int = 16 * 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_local(path: str, token: str | None = None) -> dict:
    """Compare local weight files with the Hub's hashes. Returns {matched, mismatched, missing_on_hub}."""
    published = hub_file_hashes(token)
    out: dict = {"matched": [], "mismatched": [], "missing_on_hub": [], "published_files": sorted(published)}
    for f in sorted(Path(path).rglob("*")):
        if not f.is_file() or f.suffix not in {".safetensors", ".bin", ".model"}:
            continue
        rel = f.relative_to(path).as_posix()
        if rel not in published:
            out["missing_on_hub"].append(rel)
        elif sha256_file(f) == published[rel]:
            out["matched"].append(rel)
        else:
            out["mismatched"].append(rel)
    out["identical_to_pinned_revision"] = bool(out["matched"]) and not out["mismatched"]
    return out
