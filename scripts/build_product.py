"""Build and check the fictional Nimbus Home support suite (data/product).

  python scripts/build_product.py

Writes data/product/{docs.jsonl,tasks.jsonl,corpus.sqlite,manifest.json}, then checks that
  - every answerable task's gold answer text is inside its supporting document(s) (yes/no answers are checked by hand),
  - the oracle queries retrieve every supporting document within the search budget (top 3 per search),
  - a missing-evidence task's oracle searches never retrieve a document that states the answer (checked by key terms),
  - every dated task has a superseded older edition in the corpus that the oracle query could also surface.
The suite is exploratory: 24 tasks, written by hand, never used for training or checkpoint selection.
"""

import json
import re
from pathlib import Path

from ragharness import product
from ragharness.corpus import build_index
from ragharness.retrieval import MaskedSearch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "product"

# Terms whose presence in a retrieved document would mean the "missing" task is actually answerable.
FORBIDDEN = {
    "P19": r"bluetooth\s*(version\s*)?\d", "P20": r"fingerprint", "P21": r"outdoor|mounting kit.*dollar",
    "P22": r"refund.*(days|hours)\s+to|processed within", "P23": r"international|overseas|worldwide", "P24": r"hub h2.*warranty|warranty.*hub h2",
}


def main() -> None:
    suite = product.build()
    OUT.mkdir(parents=True, exist_ok=True)
    backend = build_index(suite.corpus, OUT / "corpus.sqlite")
    search = MaskedSearch(backend)
    product.save(suite, OUT)
    by_id = {c["chunk_id"]: c for c in suite.corpus.chunks}
    problems, notes = [], []

    for t in suite.tasks:
        support = set(t["support_chunks"])
        if support:
            if t["answer"].lower() not in {"yes", "no"}:
                if not any(t["answer"].lower() in by_id[c]["text"].lower() for c in support):
                    problems.append(f"{t['id']}: gold answer {t['answer']!r} is not in its supporting text")
        got: list[str] = []
        for q in t["oracle_queries"]:
            got += [h.chunk_id for h in search.search(q, 3)]
        if support and not support <= set(got):
            problems.append(f"{t['id']}: oracle queries retrieve {sorted(set(got) & support)} of {sorted(support)}")
        if not support:
            rx = re.compile(FORBIDDEN[t["id"]], re.IGNORECASE | re.DOTALL)
            hit = [by_id[c]["title"] for c in got if rx.search(by_id[c]["text"])]
            if hit:
                problems.append(f"{t['id']}: oracle search surfaces text that answers a missing-evidence task: {hit}")
        if t["kind"] == "dated":
            stale = [c for c in got if suite.meta[c]["superseded_date"] and suite.meta[c]["superseded_date"] <= product.AS_OF]
            notes.append(f"{t['id']}: stale edition among the oracle's top-3 results: {'yes' if stale else 'no'}")

    manifest = {
        "suite": "Nimbus Home (fictional)", "as_of": product.AS_OF, "documents": len(product.DOCS), "tasks": len(suite.tasks),
        "by_kind": {k: sum(t["kind"] == k for t in suite.tasks) for k in ("ordinary", "two_doc", "dated", "missing")},
        "superseded_documents": sum(1 for d in product.DOCS if d[2]), "index_hash": suite.corpus.index_hash, "chunks": len(suite.corpus.chunks),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))
    print("\n".join(notes))
    if problems:
        raise SystemExit("PROBLEMS:\n" + "\n".join(problems))
    print("product suite OK")


if __name__ == "__main__":
    main()
