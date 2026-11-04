"""Check a local Tiny Aya Global copy against the pinned Hugging Face revision's published SHA-256 hashes.

  python scripts/verify_model.py .cache/tiny-aya-global
The HF token (env HuggingFace_TOKEN or HF_TOKEN) is optional; file metadata is public. Writes reports/model_verification.json.
"""

import json
import os
import sys
from pathlib import Path

from ragharness import modelref

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / ".cache" / "tiny-aya-global")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HuggingFace_TOKEN")
    res = modelref.verify_local(path, token)
    res["revision"] = modelref.HF_REVISION
    print(json.dumps({k: v for k, v in res.items() if k != "published_files"}, indent=1))
    (ROOT / "reports" / "model_verification.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    sys.exit(0 if res["identical_to_pinned_revision"] else 1)


if __name__ == "__main__":
    main()
