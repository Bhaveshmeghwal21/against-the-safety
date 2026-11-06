"""Round 3 data (plan section 24): training episodes for the patch, the development set for choosing between candidates,
and a fresh test set. Nothing here reads a model output.

Run:  python scripts/prepare_round3.py [--only train,dev,test]
Writes
  data/processed/episodes/{episodes,eventlogs,rows}_train3.jsonl     new training episodes (git-ignored, rebuild with this script)
  data/index/corpus_train3.sqlite                                    index for the SQuAD training episodes (git-ignored)
  data/bench/r3dev/                                                   development set: held-out SQuAD training-split articles
  data/bench/r3t_{hotpot,2wiki,musique,squad2,rgb,rgb_int}/          fresh test set
  reports/round3_data.json                                            what was drawn and how many episodes came out (committed)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

import prepare_benchmarks as PB
from ragharness import benchmarks as B, builder, corpus as corpus_mod, data, episodes, mixture, protocol, round3, trainer
from ragharness.retrieval import MaskedSearch, open_verified_backend
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
SEED = 23
N_DEV_ARTICLES, N_TRAIN_ARTICLES = 25, 80
N_SINGLE_TRAIN, N_COMPOUND_TRAIN = 240, 120
N_SINGLE_DEV, N_COMPOUND_DEV = 80, 60
TRAIN_ID_START = 20000  # chunk ids of the SQuAD training index start here, so none equals a HotpotQA chunk id
TEST_SIZES = {"2wiki": 60, "musique": 100, "squad2": 120, "rgb": 60, "rgb_int": 40}
N_HOTPOT_TEST = 60
EP = ROOT / "data" / "processed" / "episodes"


def squad_train() -> tuple[list, list, Counter]:
    qs, ctx, rej = B.load_squad2(PB.hub_file("rajpurkar/squad_v2", "squad_v2/train-00000-of-00001.parquet"))
    return [q for q in qs if not q.natural_unanswerable], ctx, rej  # natural unanswerables are never used for training


def draw(qs: list, ctx: list):
    articles = sorted({round3.article_of(q.paragraphs[0].title) for q in ctx})
    dev_art, train_art = round3.split_articles(articles, SEED, N_DEV_ARTICLES, N_TRAIN_ARTICLES)
    assert not set(dev_art) & set(train_art)
    for q in qs:
        q.qtype = "single"
    singles_tr = round3.pick_singles(qs, train_art, N_SINGLE_TRAIN, SEED)
    singles_dv = round3.pick_singles(qs, dev_art, N_SINGLE_DEV, SEED + 1)
    used = lambda ss: {s.qid for s in ss} | {s.paragraphs[0].title for s in ss}
    pool = lambda arts, ss: [q for q in qs if round3.article_of(q.paragraphs[0].title) in set(arts) and q.qid not in used(ss) and q.paragraphs[0].title not in used(ss)]
    comp_tr = round3.pick_compounds(pool(train_art, singles_tr), train_art, N_COMPOUND_TRAIN, SEED + 2)
    comp_dv = round3.pick_compounds(pool(dev_art, singles_dv), dev_art, N_COMPOUND_DEV, SEED + 3)
    return dev_art, train_art, singles_tr, comp_tr, singles_dv, comp_dv


def write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_train(qs, ctx, train_art, singles, compounds) -> dict:
    """Episodes of the three families, rows with the same decision structure as the first round."""
    base_man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    base = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=base_man["seed"])
    assert base.index_hash == json.loads((ROOT / "data" / "manifests" / "index_manifest.json").read_text(encoding="utf-8"))["index_hash"]
    base_backend = open_verified_backend(ROOT / "data" / "index" / "corpus.sqlite", base)
    base_tool, base_search = CorpusSearchTool(base_backend, top_k=3), MaskedSearch(base_backend)

    ctx_tr = [q for q in ctx if round3.article_of(q.paragraphs[0].title) in set(train_art)]
    corp3 = corpus_mod.build_corpus(ctx_tr + singles + compounds, seed=SEED, id_start=TRAIN_ID_START)
    db = ROOT / "data" / "index" / "corpus_train3.sqlite"
    corpus_mod.build_index(corp3, db)
    backend3 = open_verified_backend(db, corp3)
    tool3, search3 = CorpusSearchTool(backend3, top_k=3), MaskedSearch(backend3)
    system = protocol.system_prompt(tool3)
    main_system = json.loads((ROOT / "data" / "processed" / "training" / "main.jsonl").read_text(encoding="utf-8").splitlines()[0])["prompt"][0]["content"]
    assert system == main_system, "the system prompt must be the one main was trained with"

    built: list[tuple[str, builder.Built, object]] = []  # (family, built, question)
    for fam, qs_, corp, tool, search, full in (("single", singles, corp3, tool3, search3, True), ("compound", compounds, corp3, tool3, search3, True),
                                              ("hotpot", splits["train"], base, base_tool, base_search, False)):
        for q in qs_:
            bs = (builder.build_question(q, "train3", corp, tool, search) if full else []) + round3.build_extras(q, "train3", corp, tool, search)
            built += [(fam, b, q) for b in bs]
    accepted = [(f, b, q) for f, b, q in built if b.episode is not None]

    # no training row may show a HotpotQA dev, test or test2 support passage (the first-round rule); SQuAD dev articles are not in this index
    logs, metas, rows_out = [], [], []
    with (EP / "episodes_train3.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for fam, b, q in accepted:
            ep = b.episode
            meta = {"parent_qid": b.qid, "split": "train3", "state": b.state, "variant": b.variant, "answerable": b.answerable, "qtype": q.qtype,
                    "n_searches": ep.n_searches, "final": ep.final.turn.kind, "family": fam}
            logs.append(episodes.to_eventlog(ep))
            metas.append(meta)
            f.write(json.dumps({"meta": meta, "user": ep.user_text, "steps": [{"assistant": s.text, "observation": s.observation} for s in ep.steps],
                                "support_masked": sorted(ep.masked), "first_search_masked": sorted(ep.first_search_masked), "gold_answer": q.answer}, ensure_ascii=False) + "\n")
    episodes.write_eventlogs(EP / "eventlogs_train3.jsonl", logs)
    collector = trainer.RowCollector(system, metas)
    result = trainer.build_trainer(logs, collector).train()
    protected = {c for n in ("dev", "test") for q in splits[n] for c in base.support_copies(q)}
    kept, dropped = mixture.drop_overlap(collector.rows, protected)
    write_jsonl(EP / "rows_train3.jsonl", kept)
    qids = [m["parent_qid"] for m in metas]
    return {
        "squad_train_articles_for_questions": len(train_art), "squad_train_paragraphs_in_index": len(ctx_tr), "index_hash_train3": corp3.index_hash,
        "singles": len(singles), "compounds": len(compounds), "hotpot_train_questions": len(splits["train"]),
        "episodes_accepted": len(accepted), "rows": result["n_rows"], "rows_by_kind": result["by_kind"],
        "rows_dropped_for_showing_hotpot_dev_or_test_support": dropped, "rows_kept": len(kept),
        "episodes_by_family_variant_state": dict(sorted(Counter(f"{f}:{b.variant}:{b.state}" for f, b, _ in accepted).items())),
        "rejects_or_unreachable": dict(sorted(Counter(f"{f}:{b.variant}:{b.reject_reason}" for f, b, _ in built if b.episode is None).items())),
        "questions_with_an_open_episode": len({q for q, m in zip(qids, metas) if m["variant"] == "open"}),
    }


def build_dev(ctx, dev_art, singles, compounds) -> dict:
    """Development set: held-out articles of the SQuAD training split, single and compound questions, with the prompt variants."""
    ctx_dv = [q for q in ctx if round3.article_of(q.paragraphs[0].title) in set(dev_art)]
    out = ROOT / "data" / "bench" / "r3dev"
    out.mkdir(parents=True, exist_ok=True)
    corp = corpus_mod.build_corpus(ctx_dv + singles + compounds, seed=SEED)
    db = out / "corpus.sqlite"
    corpus_mod.build_index(corp, db)
    backend = open_verified_backend(db, corp)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    variants, reach, rejected = {}, Counter(), Counter()
    kept = []
    for q in singles + compounds:
        bs = {b.variant: b for b in builder.build_question(q, "r3dev", corp, tool, search)}
        if "open" not in bs or bs["open"].episode is None:
            reach[f"{q.qtype}:unreachable_or_rejected"] += 1
            continue
        reach[f"{q.qtype}:reachable"] += 1
        kept.append(q)
        v = {}
        for b in round3.build_extras(q, "r3dev", corp, tool, search):
            if b.episode is not None:
                v[b.variant] = round3.prompt_variants(q, corp, search)[b.variant]
            else:
                rejected[f"{b.variant}:{b.reject_reason}"] += 1
        if v:
            variants[q.qid] = v
    data.write_questions(out / "questions.jsonl", kept)
    corpus_mod.save_corpus(out / "corpus.json", corp)
    (out / "prompt_variants.json").write_text(json.dumps(variants, indent=1), encoding="utf-8", newline="\n")
    man = {"benchmark": "r3dev", "source": "rajpurkar/squad_v2 train split (CC BY-SA 4.0), held-out articles", "seed": SEED, "articles": dev_art,
           "n_questions": len(kept), "by_type": dict(Counter(q.qtype for q in kept)), "reach": dict(reach), "prompt_variants": dict(Counter(k for v in variants.values() for k in v)),
           "extras_rejected": dict(rejected), "n_chunks": len(corp.chunks), "index_hash": corp.index_hash, "ids": [q.qid for q in kept]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    return {k: v for k, v in man.items() if k not in ("ids", "articles")}


def build_test_hotpot() -> dict:
    """Fresh HotpotQA questions (test3): no support title may be the title of any paragraph already in the first-round or test2 corpus."""
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    base = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    t2 = data.read_questions(ROOT / "data" / "processed" / "questions_test2.jsonl")
    corp2 = corpus_mod.extend_corpus(base, t2)
    files = data.download(ROOT / ".cache" / "huggingface" / "hub")
    val_rows, rejected = data.load_questions(files[data.VAL_FILE])
    existing = {**splits, "test2": t2}
    blocked = {p.title for qs in existing.values() for q in qs for p in q.paragraphs}
    used = {q.qid for qs in existing.values() for q in qs}
    test3 = data.take_disjoint(data.stratified_order(val_rows, man["seed"] + 5), N_HOTPOT_TEST, blocked, used)
    corp = corpus_mod.extend_corpus(corp2, test3)
    base_ids = {c["chunk_id"]: (c["title"], c["text"]) for c in corp2.chunks}
    assert all(base_ids[c["chunk_id"]] == (c["title"], c["text"]) for c in corp.chunks[: len(corp2.chunks)])
    assert not {c for q in test3 for c in corp.support_copies(q)} & set(base_ids), "a test3 support passage is already in an earlier corpus"
    out = ROOT / "data" / "bench" / "r3t_hotpot"
    out.mkdir(parents=True, exist_ok=True)
    data.write_questions(out / "questions.jsonl", test3)
    corpus_mod.save_corpus(out / "corpus.json", corp)
    corpus_mod.build_index(corp, out / "corpus.sqlite")
    m = {"benchmark": "r3t_hotpot", "dataset": data.DATASET, "revision": data.REVISION, "file": data.VAL_FILE, "license": "CC BY-SA 4.0", "seed": man["seed"], "order_seed": man["seed"] + 5,
         "n_questions": len(test3), "by_type": dict(Counter(q.qtype for q in test3)), "n_chunks": len(corp.chunks), "new_chunks": len(corp.chunks) - len(corp2.chunks),
         "index_hash": corp.index_hash, "load_rejections": dict(rejected), "ids": [q.qid for q in test3]}
    (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8", newline="\n")
    return {k: v for k, v in m.items() if k != "ids"}


def build_test() -> dict:
    res = {"r3t_hotpot": build_test_hotpot()}
    for name, n in TEST_SIZES.items():
        skip = frozenset(json.loads((ROOT / "data" / "bench" / name / "manifest.json").read_text(encoding="utf-8"))["ids"])
        res[f"r3t_{name}"] = PB.build(name, out_name=f"r3t_{name}", n_eval=n, fresh=True, skip=skip)
    # the fresh sets must not share a question or a support title with the first-round sets
    for name in TEST_SIZES:
        old = json.loads((ROOT / "data" / "bench" / name / "manifest.json").read_text(encoding="utf-8"))["ids"]
        new = json.loads((ROOT / "data" / "bench" / f"r3t_{name}" / "manifest.json").read_text(encoding="utf-8"))["ids"]
        assert not set(old) & set(new), f"{name}: the fresh test set shares questions with the first benchmark set"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="train,dev,test")
    args = ap.parse_args()
    only = set(args.only.split(","))
    report: dict = {"seed": SEED}
    if only & {"train", "dev"}:
        qs, ctx, rej = squad_train()
        dev_art, train_art, singles_tr, comp_tr, singles_dv, comp_dv = draw(qs, ctx)
        report["squad_train"] = {"answerable_questions": len(qs), "paragraphs": len(ctx), "load_rejections": dict(rej), "dev_articles": len(dev_art), "train_articles": len(train_art),
                                 "singles_train": len(singles_tr), "compounds_train": len(comp_tr), "singles_dev": len(singles_dv), "compounds_dev": len(comp_dv)}
        ids_tr = {q.qid for q in singles_tr + comp_tr}
        assert not ids_tr & {q.qid for q in singles_dv + comp_dv}
        if "train" in only:
            report["train"] = build_train(qs, ctx, train_art, singles_tr, comp_tr)
            print(json.dumps(report["train"], indent=1), flush=True)
        if "dev" in only:
            report["dev"] = build_dev(ctx, dev_art, singles_dv, comp_dv)
            print(json.dumps(report["dev"], indent=1), flush=True)
    if "test" in only:
        report["test"] = build_test()
        print(json.dumps(report["test"], indent=1), flush=True)
    path = ROOT / "reports" / "round3_data.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(report)
    path.write_text(json.dumps(old, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
