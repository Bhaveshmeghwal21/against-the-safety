"""The logic behind the web demo (scripts/demo_app.py): parse pasted documents, index them, ask the packaged system, describe the decision.

Kept free of any web library so it can be tested on a CPU with scripted model replies. The system is the submitted one: the main adapter with
the fail-closed rule and the self-check (src/ragharness/system.py).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import system

MAX_DOCS = 200
MAX_PASSAGE_CHARS = 800  # the model reads only this much of each passage


class DocumentError(ValueError):
    """The pasted documents cannot be used; the message says why in one sentence."""


def parse_documents(text: str) -> tuple[list[dict], list[str]]:
    """One passage per line, either a JSON object with "title" and "text", or `Title: text`. Returns the passages and any notes for the user."""
    docs: list[dict] = []
    notes: list[str] = []
    for number, line in enumerate((l.strip() for l in text.splitlines()), start=1):
        if not line:
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DocumentError(f"Line {number} looks like JSON but does not parse ({error.msg}).") from error
            if not isinstance(row, dict) or not str(row.get("text", "")).strip():
                raise DocumentError(f'Line {number}: a JSON passage needs a non-empty "text" (and a "title").')
            title, body = str(row.get("title") or f"Passage {len(docs) + 1}").strip(), str(row["text"]).strip()
        elif ": " in line:
            title, body = (part.strip() for part in line.split(": ", 1))
            if not title or not body:
                raise DocumentError(f"Line {number}: write `Title: text` with both parts filled in.")
        else:
            title, body = f"Passage {len(docs) + 1}", line
        if len(body) > MAX_PASSAGE_CHARS:
            notes.append(f'"{title}" is {len(body)} characters long; the model reads only the first {MAX_PASSAGE_CHARS}. Split long documents into several lines.')
        docs.append({"title": title, "text": body})
    if not docs:
        raise DocumentError("Paste at least one passage, one per line.")
    if len(docs) > MAX_DOCS:
        raise DocumentError(f"{len(docs)} passages is more than the demo's limit of {MAX_DOCS}.")
    return docs, notes


@dataclass
class Prepared:
    backend: object
    chunks: list[dict]
    notes: list[str]

    @property
    def by_id(self) -> dict[str, dict]:
        return {c["chunk_id"]: c for c in self.chunks}


class DemoSession:
    """Holds the loaded model and the index of the most recently pasted documents."""

    def __init__(self, model, tokenizer, *, generate=None, support_fn=None, workdir: str | Path | None = None):
        self.model, self.tokenizer, self.generate, self.support_fn = model, tokenizer, generate, support_fn
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ats-demo-"))
        self._key: str | None = None
        self._prepared: Prepared | None = None

    def prepare(self, docs_text: str) -> Prepared:
        key = hashlib.sha256(docs_text.strip().encode("utf-8")).hexdigest()
        if key != self._key:
            docs, notes = parse_documents(docs_text)
            backend, chunks = system.index_documents(docs, self.workdir / f"docs-{key[:12]}.sqlite")
            self._prepared, self._key = Prepared(backend, chunks, notes), key
        return self._prepared

    def ask(self, docs_text: str, question: str) -> dict:
        question = (question or "").strip()
        if not question:
            raise DocumentError("Type a question first.")
        prepared = self.prepare(docs_text)
        rag = system.RagSystem(self.model, self.tokenizer, prepared.backend, generate=self.generate, support_fn=self.support_fn)
        return describe(question, rag.ask(question), prepared)


def describe(question: str, d: system.Decision, prepared: Prepared) -> dict:
    """A plain record of one decision, for the screen and for the saved smoke run."""
    by_id = prepared.by_id

    def passage(chunk_id: str) -> dict:
        c = by_id.get(chunk_id, {})
        return {"chunk_id": chunk_id, "title": c.get("title", "(unknown)"), "text": c.get("text", "")}

    if d.decision == "answer":
        why = "The model answered, and the self-check agreed the cited passages support it."
    elif d.repaired_by == "fail_closed":
        why = f"The model did not reach a valid ending ({d.reason.replace('invalid turn: ', '')}), so the fail-closed rule turned it into a refusal."
    elif d.repaired_by == "self_check":
        why = f"The model drafted an answer, and the self-check withheld it ({d.reason})."
    else:
        why = "The model searched and said the documents do not hold the answer."
    return {
        "question": question, "decision": d.decision, "answer": d.answer, "text": d.text, "why": why, "repaired_by": d.repaired_by,
        "p_supported": d.p_supported, "citations": [passage(c) for c in d.citations], "searches": list(d.queries),
        "passages_seen": [passage(c) for c in d.shown_chunk_ids], "withheld_draft": d.model_text if d.repaired_by == "self_check" else None,
        "model_calls": d.model_calls, "notes": list(prepared.notes),
    }


def render_verdict(r: dict) -> str:
    out = []
    if r["decision"] == "answer":
        out.append(f"### Answer\n\n**{r['answer']}**")
        if r["p_supported"] is not None:
            out.append(f"\nSelf-check: P(SUPPORTED) = {r['p_supported']:.2f}")
        if r["citations"]:
            out.append("\n**Cited passages**")
            out += [f"- `{c['chunk_id']}` {c['title']}: {c['text']}" for c in r["citations"]]
    else:
        out.append("### Refusal: the documents don't say\n")
    out.append(f"\n{r['why']}")
    if r["withheld_draft"]:
        out.append(f"\nWithheld draft: `{r['withheld_draft'][:300]}`")
    out += [f"\nNote: {n}" for n in r["notes"]]
    return "\n".join(out)


def render_trace(r: dict) -> str:
    lines = [f"**Searches the model ran ({len(r['searches'])})**"] + ([f"{i}. {q}" for i, q in enumerate(r["searches"], 1)] or ["(none)"])
    lines.append(f"\n**Passages it saw ({len(r['passages_seen'])})**")
    lines += [f"- `{p['chunk_id']}` {p['title']}" for p in r["passages_seen"]] or ["(none)"]
    lines.append(f"\nModel calls: {r['model_calls']}")
    return "\n".join(lines)
