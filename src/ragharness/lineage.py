"""CPU-only, content-based guards for training corpora and assembled prompts."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from . import corpus as C, data
from .evidence import normalize


class PromptGuard:
    def __init__(self, holdouts):
        self.questions = {}
        self.fragments = defaultdict(list)
        self.titles = set()
        for group, questions in holdouts.items():
            for q in questions:
                self.questions[normalize(q.question)] = (group, q.qid)
                for title, facts in q.support_sentences().items():
                    self.titles.add(normalize(title))
                    facts = [*facts, q.paragraph(title).text]
                    for fact in facts:
                        words = normalize(fact).split()
                        if len(words) >= 6:
                            self.fragments[tuple(words[:5])].append((" ".join(words), group, q.qid))

    def evidence_matches(self, text):
        words = normalize(text).split()
        padded = " " + " ".join(words) + " "
        matches = set()
        for i in range(max(0, len(words) - 4)):
            for fact, group, qid in self.fragments.get(tuple(words[i:i + 5]), ()):
                if f" {fact} " in padded:
                    matches.add((group, qid))
        return sorted(matches)

    def matches(self, prompt):
        users = [m["content"] for m in prompt if m["role"] == "user"]
        question = users[0].split("\n\nEvidence already provided:", 1)[0] if users else ""
        result = self.evidence_matches("\n".join(users))
        if normalize(question) in self.questions:
            result.append(self.questions[normalize(question)])
        return sorted(set(result))

    def assert_clean(self, rows):
        conflicts = conflicting_prompts(rows)
        leaks = [(i, self.matches(r["prompt"])) for i, r in enumerate(rows) if self.matches(r["prompt"])]
        if leaks or conflicts:
            raise ValueError(f"training validation failed: {len(leaks)} leaking prompts, "
                             f"{len(conflicts)} conflicting prompts; clean data before training")


def prompt_key(row):
    text = json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def conflicting_prompts(rows):
    targets = defaultdict(set)
    for row in rows:
        target = row.get("chosen", row.get("completion", []))
        targets[prompt_key(row)].add(json.dumps(target, sort_keys=True, ensure_ascii=False))
    return {key for key, values in targets.items() if len(values) > 1}


def clean_rows(rows, guard):
    conflicts, seen, clean, removed = conflicting_prompts(rows), set(), [], []
    for i, row in enumerate(rows):
        key, matches = prompt_key(row), guard.matches(row["prompt"])
        reason = "heldout_evidence_or_question" if matches else "conflicting_target" if key in conflicts else "duplicate_prompt" if key in seen else None
        if reason:
            removed.append({"row": i, "prompt_sha256": key, "reason": reason, "matches": matches})
        else:
            seen.add(key)
            clean.append(row)
    guard.assert_clean(clean)
    return clean, {"input_rows": len(rows), "clean_rows": len(clean), "removed": removed,
                   "conflicting_prompts": len(conflicts)}


def load_holdouts(root: Path):
    paths = {f"hotpot_{name}": root / "data/processed" / f"questions_{name}.jsonl" for name in ("dev", "test", "test2")}
    paths.update({p.parent.name: p for p in (root / "data/bench").glob("*/questions.jsonl") if p.parent.name != "r5train"})
    required = ("hotpot_dev", "hotpot_test", "r4dev", "r4test", "r5t_squad", "r5t_hotpot", "r5t_rgb",
                "r6t_squad", "r6t_hotpot", "r6t_rgb")
    missing = [name for name in required if name not in paths or not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(f"held-out manifests required for training validation: {missing}")
    return {name: data.read_questions(path) for name, path in paths.items() if path.is_file()}


def training_corpus(shared, train_questions, guard):
    """Retain original IDs while indexing only unexposed training paragraphs."""
    owned = {cid for q in train_questions for cid in shared.by_question[q.qid].values()}
    chunks = [c for c in shared.chunks if c["chunk_id"] in owned
              and normalize(c["title"]) not in guard.titles and not guard.evidence_matches(c["text"])]
    retained = {c["chunk_id"] for c in chunks}
    usable = [replace(q, paragraphs=[p for p in q.paragraphs if shared.by_question[q.qid][p.title] in retained])
              for q in train_questions if set(shared.support_chunks(q).values()) <= retained]
    mapping = {q.qid: {title: cid for title, cid in shared.by_question[q.qid].items() if cid in retained} for q in usable}
    digest = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c["chunk_id"]):
        digest.update(f"{c['chunk_id']}\t{c['title']}\t{c['text']}\n".encode("utf-8"))
    return C.Corpus(chunks, mapping, digest.hexdigest()), usable
