"""Build the multilingual check sets (research/2026-11-15-multilingual-plan.md, sections 2 and 3): the same questions in Hindi, English and Spanish.

Run:  python scripts/prepare_xquad.py
Writes data/bench/xq_hi, xq_en, xq_es ({questions.jsonl, corpus.json, corpus.sqlite, manifest.json}) and reports/xquad_selection.json.
Evaluation only. The sample size and seed are fixed in the plan, before any model runs on these questions.

A question is kept only if, in all three languages, its official answer offset is valid, its answer sentence fits inside one displayed
passage, and its id is not in our earlier SQuAD suites (squad2, r3t_squad2). 300 of the remaining questions are drawn with seed 47,
round-robin across the 48 Wikipedia articles (article labels come from SQuAD v1.1 dev and are never shown to a model).
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from ragharness import benchmarks as B, corpus as corpus_mod, data
from ragharness.retrieval import open_verified_backend

import prepare_benchmarks as PB

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("hi", "en", "es")
N_QUESTIONS, SEED = 300, 47
TOKENIZER = "unicode61 categories 'L* N* Co M*'"  # keeps Devanagari words whole; the same for all three languages
EARLIER_SQUAD_SUITES = ("squad2", "r3t_squad2")
OUT = ROOT / "data" / "bench"
REPORT = ROOT / "reports" / "xquad_selection.json"


def article_labels() -> dict[str, str]:
    path = PB.hub_file("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet")
    return {r["id"]: r["title"] for r in pq.read_table(path).select(["id", "title"]).to_pylist()}


def earlier_ids() -> set[str]:
    ids: set[str] = set()
    for name in EARLIER_SQUAD_SUITES:
        ids |= {json.loads(line)["qid"] for line in (OUT / name / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    return ids


def select(valid: list[str], articles: dict[str, str], n: int, seed: int) -> list[str]:
    """`n` ids spread across articles: shuffle the articles and the ids inside each (seeded), then take one from each article in turn."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for qid in sorted(valid):
        groups[articles[qid]].append(qid)
    names = sorted(groups)
    rng.shuffle(names)
    for g in groups.values():
        rng.shuffle(g)
    chosen: list[str] = []
    while len(chosen) < n and any(groups.values()):
        for name in names:
            if groups[name] and len(chosen) < n:
                chosen.append(groups[name].pop())
    return chosen


def main() -> None:
    loaded, rejections = {}, {}
    for lang in LANGS:
        path = PB.hub_file("google/xquad", f"xquad.{lang}/validation-00000-of-00001.parquet")
        questions, holders, rej = B.load_xquad(path, lang)
        loaded[lang] = ({q.qid: q for q in questions}, holders)
        rejections[lang] = dict(rej)
        print(lang, "questions", len(questions), "passages", len(holders), dict(rej), flush=True)
    total = len(pq.read_table(PB.hub_file("google/xquad", "xquad.en/validation-00000-of-00001.parquet")))
    in_all = set.intersection(*(set(loaded[lang][0]) for lang in LANGS))
    earlier = earlier_ids()
    valid = sorted(in_all - earlier)
    articles = article_labels()
    chosen = select(valid, articles, N_QUESTIONS, SEED)
    print(f"{total} questions, {len(in_all)} valid in all three languages, {len(in_all & earlier)} already in earlier SQuAD suites, "
          f"{len(valid)} available, {len(chosen)} chosen", flush=True)

    report = {"source": "google/xquad (CC BY-SA 4.0)", "seed": SEED, "tokenizer": TOKENIZER, "total_questions_per_language": total,
              "valid_in_all_three_languages": len(in_all), "removed_for_earlier_squad_suites": len(in_all & earlier),
              "available": len(valid), "chosen": len(chosen), "articles_in_sample": len({articles[i] for i in chosen}),
              "load_rejections": rejections, "ids": chosen, "languages": {}}
    for lang in LANGS:
        qmap, holders = loaded[lang]
        evalq = [qmap[i] for i in chosen]
        corp = corpus_mod.build_corpus(holders + evalq, seed=SEED)
        out = OUT / f"xq_{lang}"
        out.mkdir(parents=True, exist_ok=True)
        db = out / "corpus.sqlite"
        corpus_mod.build_index(corp, db, tokenizer=TOKENIZER)
        open_verified_backend(db, corp)  # recomputes the hash from the stored rows and runs a probe search
        data.write_questions(out / "questions.jsonl", evalq)
        corpus_mod.save_corpus(out / "corpus.json", corp)
        lengths = sorted(len(c["text"]) for c in corp.chunks)
        titles = Counter(c["title"].casefold() for c in corp.chunks)
        stats = {"passages": len(corp.chunks),
                 "passage_chars_median": lengths[len(lengths) // 2], "passage_chars_max": lengths[-1],
                 "passages_over_800_chars": sum(n > B.MAX_CHARS for n in lengths), "duplicate_titles": sum(n > 1 for n in titles.values()),
                 "answer_chars_median": sorted(len(q.answer) for q in evalq)[len(evalq) // 2], "index_hash": corp.index_hash}
        report["languages"][lang] = stats
        man = {"benchmark": f"xq_{lang}", "repo": "google/xquad", "file": f"xquad.{lang}/validation-00000-of-00001.parquet", "license": "CC BY-SA 4.0",
               "cite": "Artetxe et al., ACL 2020", "seed": SEED, "lang": lang, "tokenizer": TOKENIZER, "n_questions": len(evalq),
               "n_chunks": len(corp.chunks), "index_hash": corp.index_hash, "load_rejections": rejections[lang], "ids": chosen,
               "article_of": {i: articles[i] for i in chosen}}
        (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
        print(lang, json.dumps(stats), flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
