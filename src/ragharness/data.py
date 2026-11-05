"""HotpotQA (distractor setting) loading, seeded stratified sampling and split manifests.

Splits follow plan section 5.2: one seeded, type-stratified order per split, and no
question is accepted into a later split if one of its supporting titles is already a
supporting title of an earlier split (the "evidence family" rule, applied at title level).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATASET = "hotpotqa/hotpot_qa"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
TRAIN_FILE = "distractor/train-00000-of-00002.parquet"
VAL_FILE = "distractor/validation-00000-of-00001.parquet"
SELECTION_RULE = (
    "v2: type-stratified seeded order; later splits skip questions sharing a support title with earlier splits; "
    "dev drawn from level=hard only to match the all-hard official dev partition"
)


@dataclass
class Paragraph:
    title: str
    sentences: list[str]

    @property
    def text(self) -> str:
        return " ".join(s.strip() for s in self.sentences if s.strip())


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    qtype: str
    level: str
    paragraphs: list[Paragraph]
    support: dict[str, list[int]] = field(default_factory=dict)

    @property
    def support_titles(self) -> list[str]:
        return list(self.support)

    def paragraph(self, title: str) -> Paragraph:
        for p in self.paragraphs:
            if p.title == title:
                return p
        raise KeyError(title)

    def support_sentences(self) -> dict[str, list[str]]:
        out = {}
        for title, ids in self.support.items():
            sents = self.paragraph(title).sentences
            out[title] = [sents[i].strip() for i in ids if i < len(sents)]
        return out


def download(cache_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {
        name: Path(
            hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION, cache_dir=str(cache_dir))
        )
        for name in (TRAIN_FILE, VAL_FILE)
    }


def load_questions(parquet_path: Path) -> tuple[list[Question], Counter]:
    """Read one parquet shard. Questions whose supporting titles are missing from the context are dropped."""
    import pyarrow.parquet as pq

    rejected: Counter = Counter()
    out: list[Question] = []
    for row in pq.read_table(parquet_path).to_pylist():
        ctx = row["context"]
        paragraphs = [Paragraph(t, list(s)) for t, s in zip(ctx["title"], ctx["sentences"])]
        titles = {p.title for p in paragraphs}
        sf = row["supporting_facts"]
        support: dict[str, list[int]] = {}
        for title, sid in zip(sf["title"], sf["sent_id"]):
            support.setdefault(title, []).append(int(sid))
        if not support:
            rejected["no_support_facts"] += 1
            continue
        if any(t not in titles for t in support):
            rejected["support_title_missing_from_context"] += 1
            continue
        if len(titles) != len(paragraphs):
            rejected["duplicate_titles_in_context"] += 1
            continue
        out.append(
            Question(
                qid=row["id"],
                question=row["question"].strip(),
                answer=row["answer"].strip(),
                qtype=row["type"],
                level=row["level"],
                paragraphs=paragraphs,
                support={t: sorted(set(ids)) for t, ids in support.items()},
            )
        )
    return out, rejected


def stratified_order(rows: list[Question], seed: int) -> list[Question]:
    """Seeded order whose every prefix keeps the natural bridge/comparison ratio."""
    rng = random.Random(seed)
    groups: dict[str, list[Question]] = {}
    for q in sorted(rows, key=lambda r: r.qid):
        groups.setdefault(q.qtype, []).append(q)
    for g in groups.values():
        rng.shuffle(g)
    total = sum(len(g) for g in groups.values())
    share = {k: len(g) / total for k, g in groups.items()}
    taken = {k: 0 for k in groups}
    order: list[Question] = []
    for i in range(total):
        # pick the group furthest below its target share that still has rows
        k = max((k for k in groups if taken[k] < len(groups[k])), key=lambda k: share[k] * (i + 1) - taken[k])
        order.append(groups[k][taken[k]])
        taken[k] += 1
    return order


def take_disjoint(order: list[Question], n: int, blocked_titles: set[str], skip_ids: set[str]) -> list[Question]:
    chosen: list[Question] = []
    for q in order:
        if len(chosen) == n:
            break
        if q.qid in skip_ids or blocked_titles.intersection(q.support_titles):
            continue
        chosen.append(q)
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} of {n} questions available after the evidence-family filter")
    return chosen


def sample_splits(
    train_rows: list[Question], val_rows: list[Question], n_train: int, n_dev: int, n_test: int, seed: int
) -> dict[str, list[Question]]:
    train = take_disjoint(stratified_order(train_rows, seed), n_train, set(), set())
    blocked = {t for q in train for t in q.support_titles}
    # The official dev partition (our test pool) is all "hard", so dev uses only hard questions to match it.
    hard_pool = [q for q in train_rows if q.level == "hard"]
    dev = take_disjoint(stratified_order(hard_pool, seed + 1), n_dev, blocked, {q.qid for q in train})
    blocked |= {t for q in dev for t in q.support_titles}
    test = take_disjoint(stratified_order(val_rows, seed + 2), n_test, blocked, set())
    return {"train": train, "dev": dev, "test": test}


def sample_test2(val_rows: list[Question], existing: dict[str, list[Question]], n: int, seed: int) -> list[Question]:
    """A second, fresh test split from the same pool as test (plan section 20).

    Stricter than the first round's rule: a question is skipped if any of its support titles is the title of ANY paragraph
    already in the corpus (support or distractor of train, dev or test), so no support passage of test2 can have been shown
    in a training row. Questions already in test are skipped by id."""
    blocked = {p.title for qs in existing.values() for q in qs for p in q.paragraphs}
    used = {q.qid for qs in existing.values() for q in qs}
    return take_disjoint(stratified_order(val_rows, seed + 4), n, blocked, used)


def manifest(splits: dict[str, list[Question]], seed: int, rejected: dict[str, Counter]) -> dict:
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "files": {"train_pool": TRAIN_FILE, "test_pool": VAL_FILE},
        "seed": seed,
        "selection_rule": SELECTION_RULE,
        "load_rejections": {k: dict(v) for k, v in rejected.items()},
        "splits": {
            name: {
                "n": len(qs),
                "by_type": dict(Counter(q.qtype for q in qs)),
                "by_level": dict(Counter(q.level for q in qs)),
                "ids": [q.qid for q in qs],
            }
            for name, qs in splits.items()
        },
    }


def write_questions(path: Path, questions: list[Question]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def read_questions(path: Path) -> list[Question]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["paragraphs"] = [Paragraph(**p) for p in d["paragraphs"]]
            out.append(Question(**d))
    return out
