"""Ask the submitted system a question over your own documents: main adapter + fail-closed + self-check.

  python scripts/ask.py --docs my_docs.jsonl --question "What is the refund window?" --adapter BMEscanor/against-the-safety-tiny-aya-rag-main

`--docs` is a JSONL file with one {"title": ..., "text": ...} per line (each line is one passage; split long documents yourself, the model sees the
first 800 characters of each passage it retrieves). `--adapter` is a Hugging Face repository id or a local adapter folder. Needs a CUDA GPU and access
to the gated base model (CohereLabs/tiny-aya-global); set TINY_AYA_PATH to use a local copy. Prints one JSON object: the decision the user sees, whether a
repair changed it and why, the queries the model ran, and the passages it saw.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


def load_model_for_system(adapter: str):
    """Base model in 4-bit with the adapter on top. The tokenizer comes from the base model (the published adapters carry no tokenizer files)."""
    from peft import PeftModel

    from ragharness import generate, modelref

    model_path, revision, _ = modelref.resolve()
    model, tok = generate.load_model(model_path, revision, adapter=None)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--question", action="append", required=True, help="repeat to ask several questions")
    ap.add_argument("--adapter", default="BMEscanor/against-the-safety-tiny-aya-rag-main")
    ap.add_argument("--index", default=None, help="where to write the SQLite index (default: a temporary folder)")
    args = ap.parse_args()

    from ragharness import system

    docs = [json.loads(l) for l in Path(args.docs).read_text(encoding="utf-8").splitlines() if l.strip()]
    for d in docs:
        if not {"title", "text"} <= set(d):
            sys.exit('every line of --docs needs "title" and "text"')
    db = Path(args.index) if args.index else Path(tempfile.mkdtemp()) / "docs.sqlite"
    backend, chunks = system.index_documents(docs, db)
    by_id = {c["chunk_id"]: c["title"] for c in chunks}
    model, tok = load_model_for_system(args.adapter)
    rag = system.RagSystem(model, tok, backend)
    for question in args.question:
        d = rag.ask(question)
        out = {"question": question, "decision": d.decision, "text": d.text, "answer": d.answer, "citations": [{"chunk_id": c, "title": by_id.get(c)} for c in d.citations],
               "repaired_by": d.repaired_by, "reason": d.reason, "p_supported": d.p_supported, "queries": d.queries,
               "shown": [{"chunk_id": c, "title": by_id.get(c)} for c in d.shown_chunk_ids]}
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
