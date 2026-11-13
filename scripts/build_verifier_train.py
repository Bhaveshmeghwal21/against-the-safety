"""Build question-disjoint checker training cases from saved training artifacts."""

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ragharness import checker, corpus, data, evidence, lineage, metrics, protocol, rescore, round3
from ragharness.retrieval import Hit, MaskedSearch, open_verified_backend


def as_prompt(case):
    return [{"role": "user", "content": case["question"] + "\n\nEvidence already provided:\n" + case["passages"]}]


def filter_cases(cases):
    return [case for case in cases if not case["deterministic"]["cites_unshown"]
            and not case["deterministic"]["answer_part_absent_from_cited"]]


def swap_answer(answer, passage, question=None):
    if re.fullmatch(r"[\d,]+(?:\.\d+)?", answer.strip()):
        candidates = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", passage)
    elif answer.strip().lower() in ("yes", "no"):
        return None
    else:
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", passage)
    gold = evidence.normalize(answer)
    return next((candidate for candidate in candidates if evidence.normalize(candidate) != gold
                 and (question is None or not metrics.judge_answer(candidate, question)[2])), None)


def make_case(run, world, qid, variant, question, answer, citations, hits, label, source):
    flags = checker.deterministic_flags(answer, citations, hits, 800)
    return {"run": run, "world": world, "qid": qid, "variant": variant, "question": question,
            "answer": answer, "citations": citations, "passages": flags.pop("passages"),
            "deterministic": flags, "label": label, "source": source}


def hit(corp, cid):
    chunk = corp.chunk(cid)
    return Hit(cid, chunk["title"], chunk["text"], 0.0)


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sampled_case(row, corp, question):
    if row["final"]["kind"] != "answer":
        return None
    shown = list(dict.fromkeys(row["prefill"] + [cid for step in row["steps"] for cid in step["hits"]]))
    hits = [hit(corp, cid) for cid in shown]
    final = row["final"]
    transcript = {"qid": row["qid"], "variant": row["variant"], "answerable": row["answerable"],
                  "masked": row["masked"], "max_chars": row["max_chars"], "final": final,
                  "provided_evidence": [{"chunk_id": cid} for cid in row["prefill"]],
                  "steps": [{"retrieved": [{"chunk_id": cid} for cid in step["hits"]],
                             "observation": step["observation"]} for step in row["steps"]]}
    score = rescore.score(transcript, question, corp)
    if score["excluded_v3"]:
        return None
    return make_case(row["system"], row["set"], row["qid"], row["variant"], question.question,
                     final["answer"], final["citations"], hits, int(score["outcome_v3"] == "supported_correct"), "sampled")


def context_hits(context, corp):
    seen = set()
    for message in context["prompt"]:
        if message["role"] == "user":
            seen.update(re.findall(r"\[chunk_id=(c\d+)\]", message["content"]))
    return [hit(corp, cid) for cid in sorted(seen) if cid in corp._by_id]


def build_gold_and_swaps(world, questions, corp, db, candidates):
    search = MaskedSearch(open_verified_backend(db, corp))
    by_article = defaultdict(set)
    for chunk in corp.chunks:
        by_article[round3.article_of(chunk["title"])].add(chunk["chunk_id"])
    for question in questions:
        support = list(corp.support_chunks(question).values())
        if not support:
            continue
        hits = [hit(corp, cid) for cid in support]
        candidates.append(make_case("gold", world, question.qid, "gold", question.question, question.answer,
                                    support, hits, 1, "gold"))
        related = set(corp.by_question[question.qid].values())
        for title in question.support:
            related.update(by_article[round3.article_of(title)])
        for candidate in search.search(question.question, top_k=60, masked=frozenset(support)):
            if candidate.chunk_id not in related:
                continue
            answer = swap_answer(question.answer, candidate.text[:800], question)
            if answer:
                candidates.append(make_case("swap", world, question.qid, "swap", question.question, answer,
                                            [candidate.chunk_id], [candidate], 0, "swap"))
                break


def main():
    import build_pairs_round5 as pairs
    import prepare_round3 as r3
    import prepare_round4 as r4

    guard = lineage.PromptGuard(lineage.load_holdouts(ROOT))
    worlds = {name: pairs.load_world(name) for name in ("r5train", "hotpot_train")}
    paths = sorted((ROOT / "runs/account2_round5/ats-sample-r5-pairs/out/samples").glob("*/samples_*.jsonl"))
    if len(paths) != 4:
        raise FileNotFoundError("runs/account2_round5/ats-sample-r5-pairs/out/samples/*/samples_*.jsonl")
    candidates, sampled_ids = [], defaultdict(set)
    for path in paths:
        for row in read(path):
            corp, questions = worlds[row["set"]]
            sampled_ids[row["set"]].add(row["qid"])
            case = sampled_case(row, corp, questions[row["qid"]])
            if case:
                candidates.append(case)

    contexts = {row["id"]: row for row in read(ROOT / "data/processed/training/force_contexts_r5.jsonl")}
    forced_root = ROOT / "runs/account2_round5/ats-force-r5-r5b/out/out/forced"
    for name in ("main", "candA"):
        for row in read(forced_root / f"forced_{name}.jsonl"):
            context = contexts[row["id"]]
            corp, questions = worlds[context["set"]]
            parsed, _ = protocol.parse_turn_lenient(row["text"])
            if parsed.kind != "answer" or not parsed.citations:
                continue
            q = questions[context["qid"]]
            candidates.append(make_case(name, context["set"], q.qid, context["variant"], q.question,
                                        parsed.answer, parsed.citations, context_hits(context, corp), 0, "forced"))

    for name in ("r5train", "hotpot_train"):
        corp, questions = worlds[name]
        ids = set(questions) if name == "r5train" else sampled_ids[name]
        db = ROOT / ("data/bench/r5train/corpus.sqlite" if name == "r5train" else "data/index/corpus.sqlite")
        build_gold_and_swaps(name, [questions[qid] for qid in sorted(ids)], corp, db, candidates)

    source_questions, source_contexts, _ = r3.squad_train()
    for q in source_questions:
        q.qtype = "single"
    report = json.loads((ROOT / "reports/round4_data.json").read_text(encoding="utf-8"))
    articles = report["groups"]["train"]["articles"]
    selected = r4.pick(source_questions, articles, 240, 100)
    context = [q for q in source_contexts if round3.article_of(q.paragraphs[0].title) in set(articles)]
    r4corp = corpus.build_corpus(context + selected, seed=r4.SEED, id_start=40000)
    build_gold_and_swaps("train4", selected, r4corp, ROOT / "data/index/corpus_train4.sqlite", candidates)

    removed = Counter()
    clean = []
    for case in candidates:
        if not filter_cases([case]):
            removed["deterministic"] += 1
        elif guard.matches(as_prompt(case)):
            removed["heldout_match"] += 1
        else:
            clean.append(case)
    rng = random.Random(61)
    # A checker sees only these three values. Opposing labels for identical
    # inputs are irresolvable even when they came from different episodes.
    labels = defaultdict(set)
    for case in clean:
        labels[(case["question"], case["answer"], case["passages"])].add(case["label"])
    consistent = []
    for case in clean:
        if len(labels[(case["question"], case["answer"], case["passages"])]) > 1:
            removed["conflicting_labels"] += 1
        else:
            consistent.append(case)
    clean = consistent
    rng.shuffle(clean)
    per_question = Counter()
    capped = []
    seen = set()
    for case in clean:
        key = (case["question"], case["answer"], case["passages"])
        if key in seen:
            removed["duplicate"] += 1
        elif per_question[case["qid"]] >= 6:
            removed["question_cap"] += 1
        else:
            seen.add(key)
            per_question[case["qid"]] += 1
            capped.append(case)
    split = {"train": [], "val": []}
    for case in capped:
        bucket = int(hashlib.sha256(case["qid"].encode("utf-8")).hexdigest(), 16) % 10
        split["val" if bucket == 0 else "train"].append(case)
    for name, cases in split.items():
        groups = {label: [case for case in cases if case["label"] == label] for label in (0, 1)}
        each = min(len(groups[0]), len(groups[1]), 1000 if name == "train" else len(cases))
        if not each:
            raise ValueError(f"Both labels are required in {name} data")
        split[name] = groups[0][:each] + groups[1][:each]
        rng.shuffle(split[name])
        removed[f"{name}_balance_or_cap"] += len(cases) - 2 * each
    output = ROOT / "data/processed/verifier"
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, cases in split.items():
        path = output / f"{name}_v2.jsonl"
        path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
        reports[name] = {"rows": len(cases), "questions": len({c['qid'] for c in cases}),
                         "by_source_and_label": dict(Counter(f"{c['source']}:{c['label']}" for c in cases)),
                         "yes_no": sum(c["answer"].strip().lower() in ("yes", "no") for c in cases),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {"seed": 61, "candidate_rows": len(candidates), "removed": dict(removed), "splits": reports}
    (ROOT / "reports/round6_verifier_data.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required saved artifact is missing: {exc.filename or exc}") from exc
