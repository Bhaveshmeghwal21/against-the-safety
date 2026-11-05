"""Upload release/ to Hugging Face as public repositories and group them into two collections.

  python scripts/publish_release.py --user BMEscanor            # create/update everything
  python scripts/publish_release.py --user BMEscanor --check    # only verify, anonymously, that everything is public and un-gated

The token comes from the HuggingFace_TOKEN environment variable (never printed). Safe to re-run: repositories are created with
exist_ok, uploads are idempotent, and collections are looked up by title before they are created.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

REPOS = [  # (kind, name, local folder)
    ("model", "against-the-safety-tiny-aya-rag-main", "model_main"),
    ("model", "against-the-safety-tiny-aya-rag-control", "model_control"),
    ("dataset", "against-the-safety-rag-episodes", "dataset_episodes"),
    ("dataset", "nimbus-home-support-suite", "dataset_nimbus"),
    ("space", "against-the-safety-trace-viewer", "space"),
    ("model", "against-the-safety-tiny-aya-rag-cand-a", "model_cand_a"),  # round 2, not the selected model (plan section 21)
]

COLLECTIONS = {
    "models": {
        "title": "Against-the-safety: Tiny Aya RAG adapters",
        "description": "LoRA adapters for Tiny Aya Global trained with AgentTune on search-cite-abstain episodes (PS 4).",
        "items": [("model", 0), ("model", 1), ("model", 5)],
    },
    "data": {
        "title": "Against-the-safety: RAG episodes, traces and demo",
        "description": "Training episodes, AuditKIT evaluation traces, the Nimbus Home suite and a static trace viewer (PS 4).",
        "items": [("dataset", 2), ("dataset", 3), ("space", 4)],
    },
}


def repo_id(user: str, i: int) -> str:
    return f"{user}/{REPOS[i][1]}"


def token() -> str:
    t = os.environ.get("HuggingFace_TOKEN") or os.environ.get("HF_TOKEN")
    if not t:
        sys.exit("HuggingFace_TOKEN is not set")
    return t


def publish(user: str, release: Path) -> None:
    api = HfApi(token=token())
    who = api.whoami()["name"]
    if who != user:
        sys.exit(f"token belongs to {who!r}, not {user!r}")
    for kind, name, folder in REPOS:
        rid = f"{user}/{name}"
        src = release / folder
        if not src.exists():
            sys.exit(f"missing {src}; run scripts/build_release.py first")
        kwargs = {"space_sdk": "static"} if kind == "space" else {}
        api.create_repo(rid, repo_type=kind, private=False, exist_ok=True, **kwargs)
        t0 = time.time()
        api.upload_folder(repo_id=rid, repo_type=kind, folder_path=str(src), commit_message="Upload release files")
        print(f"uploaded {kind} {rid} in {time.time() - t0:.0f}s", flush=True)

    existing = {c.title: c for c in api.list_collections(owner=user)}
    for spec in COLLECTIONS.values():
        col = existing.get(spec["title"]) or api.create_collection(
            title=spec["title"], namespace=user, description=spec["description"], private=False
        )
        have = {(i.item_type, i.item_id) for i in api.get_collection(col.slug).items}
        for kind, idx in spec["items"]:
            if (kind, repo_id(user, idx)) not in have:
                api.add_collection_item(col.slug, item_id=repo_id(user, idx), item_type=kind)
        print("collection", f"https://huggingface.co/collections/{col.slug}", flush=True)


def check(user: str) -> bool:
    """Anonymous check: every repository answers 200 without a token, and none is gated."""
    ok = True
    api = HfApi()  # no token on purpose
    for kind, name, _ in REPOS:
        rid = f"{user}/{name}"
        base = {"model": "", "dataset": "datasets/", "space": "spaces/"}[kind]
        r = requests.get(f"https://huggingface.co/{base}{rid}", timeout=30)
        info = None
        try:
            info = {"model": api.model_info, "dataset": api.dataset_info, "space": api.space_info}[kind](rid)
        except Exception as e:  # noqa: BLE001
            print("  api error:", type(e).__name__, e)
        gated = getattr(info, "gated", None)
        private = getattr(info, "private", None)
        good = r.status_code == 200 and not gated and not private
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {kind:7s} {rid}  http={r.status_code} gated={gated} private={private}")
    for spec in COLLECTIONS.values():
        found = [c for c in HfApi(token=token()).list_collections(owner=user) if c.title == spec["title"]]
        for c in found:
            r = requests.get(f"https://huggingface.co/collections/{c.slug}", timeout=30)
            good = r.status_code == 200 and not c.private
            ok &= good
            print(f"{'OK ' if good else 'BAD'} collection {c.slug} http={r.status_code} private={c.private} items={len(c.items)}")
        if not found:
            ok = False
            print("BAD collection missing:", spec["title"])
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="BMEscanor")
    ap.add_argument("--release", default="release")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        publish(args.user, ROOT / args.release)
    sys.exit(0 if check(args.user) else 1)


if __name__ == "__main__":
    main()
