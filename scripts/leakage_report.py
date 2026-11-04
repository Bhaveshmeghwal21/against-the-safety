"""Document- and question-level overlap between SFT rows and the dev/test questions (plan section 5.2).

Claims of document-level transfer apply only to dev/test questions whose support passages never appear in an SFT
row, so those flags are saved for the evaluation report.

Run (after make_training_set.py):  python scripts/leakage_report.py
"""

import json
import re
from pathlib import Path

from ragharness import corpus as corpus_mod, data

ROOT = Path(__file__).resolve().parents[1]
CHUNK = re.compile(r"\[chunk_id=(c\d{5})\]")


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def rows_of(name: str) -> list[dict]:
    p = ROOT / "data" / "processed" / "training" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def seen_chunks(rows: list[dict]) -> tuple[set[str], str]:
    ids: set[str] = set()
    text = []
    for r in rows:
        blob = " ".join(m["content"] for m in r["prompt"][1:]) + " " + r["completion"][0]["content"]  # skip the system prompt
        ids.update(CHUNK.findall(blob))
        ids.update(re.findall(r"<citations>(.*?)</citations>", blob))
        text.append(blob)
    return ids, norm(" ".join(text))


def main() -> None:
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])

    sets = {"main": rows_of("main"), "control": rows_of("control")}
    all_train = [json.loads(l) for l in (ROOT / "data" / "processed" / "episodes" / "rows_train.jsonl").read_text(encoding="utf-8").splitlines()]
    sets["all_train_rows"] = all_train
    train_question_norms = {norm(q.question) for q in splits["train"]}

    report: dict = {}
    flags: dict = {}
    for set_name, rows in sets.items():
        seen, blob = seen_chunks(rows)
        report[set_name] = {"rows": len(rows), "distinct_chunks_seen": len(seen)}
        for split in ("dev", "test"):
            n_all = n_any = n_none = 0
            unseen_ids, partly_ids = [], []
            same_question = 0
            for q in splits[split]:
                sup = set(corp.support_chunks(q).values())
                hit = sup & seen
                if hit == sup:
                    n_all += 1
                elif hit:
                    n_any += 1
                    partly_ids.append(q.qid)
                else:
                    n_none += 1
                    unseen_ids.append(q.qid)
                same_question += norm(q.question) in train_question_norms
            report[set_name][split] = {
                "questions": len(splits[split]),
                "all_support_seen_in_sft": n_all,
                "some_support_seen_in_sft": n_any,
                "no_support_seen_in_sft": n_none,
                "question_text_also_in_train_questions": same_question,
            }
            if set_name == "main":
                flags[split] = {"unseen": unseen_ids, "partly_seen": partly_ids}

    # corpus-level: how many paragraphs of dev/test questions exist in the train questions' own paragraph pools
    train_chunks = {c for q in splits["train"] for c in corp.by_question[q.qid].values()}
    for split in ("dev", "test"):
        total = shared = 0
        for q in splits[split]:
            ids = set(corp.by_question[q.qid].values())
            total += len(ids)
            shared += len(ids & train_chunks)
        report[f"{split}_paragraphs_also_in_train_pools"] = {"paragraphs": total, "shared_with_train_pools": shared}

    (ROOT / "reports" / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    (ROOT / "data" / "manifests" / "support_seen_in_sft.json").write_text(json.dumps(flags, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
