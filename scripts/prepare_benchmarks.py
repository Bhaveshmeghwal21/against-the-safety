"""Build the external benchmark sets (plan section 22): questions, corpus file, search index and manifest per benchmark.

Run:  python scripts/prepare_benchmarks.py [--only 2wiki,musique,squad2,rgb,rgb_int]
Writes data/bench/<name>/{questions.jsonl, corpus.json, corpus.sqlite, manifest.json} and, for RGB, prompt_variants.json.
Evaluation only. Sizes and seeds are fixed here before any model is run on these sets.
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from ragharness import benchmarks as B, corpus as corpus_mod, data
from ragharness.retrieval import MaskedSearch, open_verified_backend

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / ".cache" / "huggingface" / "hub"
RAW = ROOT / ".cache" / "bench_raw"
SEED = 22

SOURCES = {
    "2wiki": {"repo": "xanhho/2WikiMultihopQA", "file": "dev.parquet", "license": "Apache-2.0", "cite": "Ho et al., COLING 2020"},
    "musique": {"repo": "bdsaglam/musique", "file": "musique_ans_v1.0_dev.jsonl", "license": "CC BY 4.0", "cite": "Trivedi et al., TACL 2022"},
    "squad2": {"repo": "rajpurkar/squad_v2", "file": "squad_v2/validation-00000-of-00001.parquet", "license": "CC BY-SA 4.0", "cite": "Rajpurkar et al., ACL 2018"},
    "rgb": {"url": "https://raw.githubusercontent.com/chen700564/RGB/master/data/en_refine.json", "license": "CC BY-NC-SA 4.0", "cite": "Chen et al., AAAI 2024"},
    "rgb_int": {"url": "https://raw.githubusercontent.com/chen700564/RGB/master/data/en_int.json", "license": "CC BY-NC-SA 4.0", "cite": "Chen et al., AAAI 2024"},
}
SIZES = {"2wiki": (100, 600), "musique": (200, 600), "squad2": (160, None), "rgb": (100, None), "rgb_int": (60, None)}  # (evaluated, corpus questions)


def hub_file(repo: str, file: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, file, repo_type="dataset", cache_dir=str(HUB)))


def raw_file(url: str) -> Path:
    import urllib.request

    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / url.rsplit("/", 1)[1]
    if not p.exists():
        urllib.request.urlretrieve(url, p)
    return p


def build(name: str) -> dict:
    src = SOURCES[name]
    n_eval, n_corpus = SIZES[name]
    variants: dict = {}
    if name == "2wiki":
        qs, rej = B.load_2wiki(hub_file(src["repo"], src["file"]))
        order = data.stratified_order(qs, SEED)
        corpus_qs, evalq = order[:n_corpus], order[:n_eval]
    elif name == "musique":
        qs, rej = B.load_musique(hub_file(src["repo"], src["file"]))
        order = data.stratified_order(qs, SEED)
        corpus_qs, evalq = order[:n_corpus], order[:n_eval]
    elif name == "squad2":
        qs, ctx_qs, rej = B.load_squad2(hub_file(src["repo"], src["file"]))
        evalq = data.stratified_order(qs, SEED)[:n_eval]  # stratified by qtype: half answerable, half natural unanswerable
        corpus_qs = ctx_qs
    else:
        qs, rej = B.load_rgb(raw_file(src["url"]), integration=(name == "rgb_int"))
        corpus_qs = qs  # every passage of every RGB query is in the index, so negatives of other queries are distractors too
        evalq = B.sample(qs, len(qs), SEED, stratify=False)

    out = ROOT / "data" / "bench" / name
    out.mkdir(parents=True, exist_ok=True)
    seen_ids = {q.qid for q in corpus_qs}
    corp = corpus_mod.build_corpus(list(corpus_qs) + [q for q in evalq if q.qid not in seen_ids], seed=SEED)
    db = out / "corpus.sqlite"
    corpus_mod.build_index(corp, db)
    dropped = Counter()
    if name.startswith("rgb"):
        backend = open_verified_backend(db, corp)
        search = MaskedSearch(backend)
        kept = []
        for q in evalq:
            by_title = corp.by_question[q.qid]
            chosen = B.choose_rgb_support(q, search, by_title)
            if chosen is None:
                dropped["no_positive_shows_the_answer_within_the_display_limit"] += 1
                continue
            q.support, extra = chosen
            corp.extra_copies[q.qid] = extra
            v = B.rgb_prefill_variants(q, by_title, SEED)
            if len(v) < (3 if name == "rgb" else 2):
                dropped["too_few_usable_positives_or_negatives_for_the_prompt_settings"] += 1
                continue
            variants[q.qid] = v
            kept.append(q)
            if len(kept) == n_eval:
                break
        evalq = kept
    data.write_questions(out / "questions.jsonl", evalq)
    corpus_mod.save_corpus(out / "corpus.json", corp)
    if variants:
        (out / "prompt_variants.json").write_text(json.dumps(variants, indent=1), encoding="utf-8", newline="\n")
    man = {
        "benchmark": name, **src, "seed": SEED, "n_questions": len(evalq), "by_type": dict(Counter(q.qtype for q in evalq)),
        "natural_unanswerable": sum(q.natural_unanswerable for q in evalq), "corpus_questions": len(corpus_qs),
        "n_chunks": len(corp.chunks), "index_hash": corp.index_hash, "load_rejections": dict(rej), "dropped_after_index": dict(dropped),
        "ids": [q.qid for q in evalq],
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    return {k: v for k, v in man.items() if k != "ids"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SOURCES))
    args = ap.parse_args()
    for name in args.only.split(","):
        print(json.dumps(build(name), indent=1), flush=True)


if __name__ == "__main__":
    main()
