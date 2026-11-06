"""Round 5 data (plan section 27, priorities 0d and 2): article allocation, the pair-sampling set and the sealed test set.

Nothing here reads a model output. Run:  python scripts/prepare_round5.py [--only alloc,train,test_squad]

  alloc         allocate the SQuAD training-split articles that no earlier round used: r5train (sampling for preference pairs),
                r5test (sealed final test), and a reserve. Seeded; written to reports/round5_allocation.json.
  train         data/bench/r5train       questions, corpus, index and prompt variants for pair sampling (training articles only)
  test_squad    data/bench/r5t_squad     sealed final-test SQuAD part (single + compound questions, round-4 variants)
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_round3 as R3  # noqa: E402
import prepare_round4 as R4  # noqa: E402
from ragharness import round3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 53
N_TRAIN_ARTICLES, N_TEST_ARTICLES = 50, 25
N_HOTPOT_TEST = 60
SIZES = {"r5train": (60, 30), "r5t_squad": (80, 40)}


def allocation(qs, ctx):
    articles = sorted({round3.article_of(q.paragraphs[0].title) for q in ctx})
    old_dev, old_train = round3.split_articles(articles, R3.SEED, R3.N_DEV_ARTICLES, R3.N_TRAIN_ARTICLES)
    remaining = sorted(set(articles) - set(old_dev) - set(old_train))
    random.Random(R4.SEED).shuffle(remaining)
    r4 = {"train": remaining[:60], "dev": remaining[60:75], "test": remaining[75:95]}  # exactly as scripts/prepare_round4.py
    pool = sorted(remaining[95:])
    random.Random(SEED).shuffle(pool)
    alloc = {"r5train": pool[:N_TRAIN_ARTICLES], "r5test": pool[N_TRAIN_ARTICLES:N_TRAIN_ARTICLES + N_TEST_ARTICLES],
             "reserve": pool[N_TRAIN_ARTICLES + N_TEST_ARTICLES:]}
    groups = {"r3_dev": old_dev, "r3_train": old_train, **{f"r4_{k}": v for k, v in r4.items()}, **alloc}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(groups[a]) & set(groups[b]), f"article overlap between {a} and {b}"
    assert sum(len(v) for v in groups.values()) == len(articles), "every article must be in exactly one group"
    return alloc, {k: len(v) for k, v in groups.items()}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_hotpot_test() -> dict:
    """r5t_hotpot: ~60 HotpotQA validation questions no earlier set used, with no support title already in any earlier corpus."""
    import prepare_round3  # noqa: F401  (module import keeps the path setup identical to the other builders)
    from ragharness import corpus as corpus_mod, data

    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    splits["test2"] = data.read_questions(ROOT / "data" / "processed" / "questions_test2.jsonl")
    splits["r3t_hotpot"] = data.read_questions(ROOT / "data" / "bench" / "r3t_hotpot" / "questions.jsonl")
    files = data.download(ROOT / ".cache" / "huggingface" / "hub")
    val_rows, rejected = data.load_questions(files[data.VAL_FILE])
    blocked = {p.title for qs in splits.values() for q in qs for p in q.paragraphs}
    used = {q.qid for qs in splits.values() for q in qs}
    order = data.stratified_order(val_rows, man["seed"] + 9)
    picked = data.take_disjoint(order, N_HOTPOT_TEST, blocked, used)
    prior = corpus_mod.load_corpus(ROOT / "data" / "bench" / "r3t_hotpot" / "corpus.json")  # base + test2 + r3t_hotpot passages
    corp = corpus_mod.extend_corpus(prior, picked)
    ids = {c["chunk_id"]: (c["title"], c["text"]) for c in prior.chunks}
    assert all(ids[c["chunk_id"]] == (c["title"], c["text"]) for c in corp.chunks[: len(prior.chunks)])
    assert not {c for q in picked for c in corp.support_copies(q)} & set(ids), "an r5t_hotpot support passage is already in an earlier corpus"
    out = ROOT / "data" / "bench" / "r5t_hotpot"
    out.mkdir(parents=True, exist_ok=True)
    data.write_questions(out / "questions.jsonl", picked)
    corpus_mod.save_corpus(out / "corpus.json", corp)
    corpus_mod.build_index(corp, out / "corpus.sqlite")
    manifest = {"benchmark": "r5t_hotpot", "dataset": data.DATASET, "revision": data.REVISION, "file": data.VAL_FILE, "license": "CC BY-SA 4.0",
                "order_seed": man["seed"] + 9, "n_questions": len(picked), "n_chunks": len(corp.chunks), "index_hash": corp.index_hash,
                "load_rejections": dict(rejected), "ids": [q.qid for q in picked]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return {k: v for k, v in manifest.items() if k != "ids"}


def build_rgb_test() -> dict:
    """r5t_rgb: the RGB questions neither the first benchmark set nor the round-3 fresh set used."""
    import prepare_benchmarks as PB

    skip = frozenset()
    for name in ("rgb", "r3t_rgb"):
        skip |= frozenset(json.loads((ROOT / "data" / "bench" / name / "manifest.json").read_text(encoding="utf-8"))["ids"])
    manifest = PB.build("rgb", out_name="r5t_rgb", n_eval=100, fresh=True, skip=skip)
    ids = json.loads((ROOT / "data" / "bench" / "r5t_rgb" / "manifest.json").read_text(encoding="utf-8"))["ids"]
    assert not set(ids) & skip, "r5t_rgb shares a question with an earlier RGB set"
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="alloc,train,test_squad")
    only = set(ap.parse_args().only.split(","))
    qs, ctx, _ = R3.squad_train()
    for q in qs:
        q.qtype = "single"
    alloc, counts = allocation(qs, ctx)
    path = ROOT / "reports" / "round5_allocation.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report.update(seed=SEED, article_groups=counts, allocation=alloc)
    print("article groups:", counts, flush=True)
    existing = set()
    for name in ("r3dev", "r4dev", "r4test"):
        existing |= set(json.loads((ROOT / "data" / "bench" / name / "manifest.json").read_text(encoding="utf-8"))["ids"])
    for key, name, articles in (("train", "r5train", alloc["r5train"]), ("test_squad", "r5t_squad", alloc["r5test"])):
        if key not in only:
            continue
        ns, nc = SIZES[name]
        manifest, _ = R4.build_group(name, qs, ctx, articles, ns, nc)
        assert not set(manifest["ids"]) & existing, f"{name} shares a question with an earlier set"
        report.setdefault("sets", {})[name] = {k: v for k, v in manifest.items() if k not in ("articles", "ids")}
        report["sets"][name]["questions_sha256"] = sha(ROOT / "data" / "bench" / name / "questions.jsonl")
        report["sets"][name]["corpus_sha256"] = sha(ROOT / "data" / "bench" / name / "corpus.json")
        print(name, json.dumps(report["sets"][name]), flush=True)
    for key, name, build in (("test_hotpot", "r5t_hotpot", build_hotpot_test), ("test_rgb", "r5t_rgb", build_rgb_test)):
        if key not in only:
            continue
        report.setdefault("sets", {})[name] = build()
        report["sets"][name]["questions_sha256"] = sha(ROOT / "data" / "bench" / name / "questions.jsonl")
        report["sets"][name]["corpus_sha256"] = sha(ROOT / "data" / "bench" / name / "corpus.json")
        print(name, json.dumps(report["sets"][name]), flush=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
