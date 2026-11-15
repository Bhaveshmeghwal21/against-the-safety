"""Hindi training, development and fresh-test data (research/2026-11-15-hindi-training-preregistration.md).

Run (CPU only, no model output is read):
  python scripts/prepare_hindi.py dev      # data/bench/hi_dev: Hindi development questions (IndicSQuAD, held-out articles)
  python scripts/prepare_hindi.py xq2      # data/bench/xq2_hi, xq2_en: 300 XQuAD questions no model has run on (same 48 articles as xq_*)
  python scripts/prepare_hindi.py train    # data/processed/training/hiMain.jsonl, hiF.jsonl (+ episodes, report)

Training and development questions come from IndicSQuAD Hindi (L3Cube, CC BY 4.0), a machine translation of SQuAD 2.0 whose ids are the SQuAD ids, so every
question maps to its English article. Only articles that are in the "reserve" of the round-5 allocation and not in the round-6 test are used: no English
model was trained or evaluated on them, and none is a SQuAD v1.1 development article (the source of XQuAD, our Hindi test). Hindi answers whose
translation looks broken (very short, a function word, or inside the question) are dropped. Contexts keep their machine-translated Devanagari digits.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 53
N_TRAIN_ARTICLES, N_DEV_ARTICLES = 70, 15
N_DEV_QUESTIONS = 120
N_XQ2 = 300
TRAIN_ID_START = 60000  # chunk ids of the Hindi training index; the English training indexes use 1.., 20000.. and 40000..
TOKENIZER = "unicode61 categories 'L* N* Co M*'"
MAX_ROW_TOKENS = 5800  # one training row; train.py gets --max-length 6144
DANDA = "\u0964\u0965"
EP = ROOT / "data" / "processed" / "episodes"
TRAIN = ROOT / "data" / "processed" / "training"
BENCH = ROOT / "data" / "bench"
PATHS = ROOT / ".cache" / "indic" / "paths.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------------------- source data


def space_after_danda(context: str, offset: int) -> tuple[str, int]:
    """IndicSQuAD writes the next sentence right after the danda ("...हैं।१९५० के ..."), so sentence splitting needs a space there.
    Insert one and move the answer offset by the number of spaces inserted before it."""
    out, shift = [], 0
    for i, ch in enumerate(context):
        out.append(ch)
        if ch in DANDA and i + 1 < len(context) and not context[i + 1].isspace():
            out.append(" ")
            if i < offset:
                shift += 1
    return "".join(out), offset + shift


def english_articles() -> dict[str, str]:
    import pyarrow.parquet as pq

    import prepare_benchmarks as PB

    table = pq.read_table(PB.hub_file("rajpurkar/squad_v2", "squad_v2/train-00000-of-00001.parquet")).select(["id", "title"]).to_pylist()
    return {r["id"]: r["title"].replace("_", " ") for r in table}


def article_pool() -> list[str]:
    """Reserve articles of the round-5 allocation minus the round-6 test: untouched by every English model, as training data or as test."""
    alloc = json.loads((ROOT / "reports" / "round5_allocation.json").read_text(encoding="utf-8"))["allocation"]
    r6 = set(json.loads((ROOT / "reports" / "round6_allocation.json").read_text(encoding="utf-8"))["r6test"])
    return sorted(set(alloc["reserve"]) - r6)


def split_articles(pool: list[str]) -> tuple[list[str], list[str]]:
    names = list(pool)
    random.Random(SEED).shuffle(names)
    return names[:N_TRAIN_ARTICLES], names[N_TRAIN_ARTICLES:N_TRAIN_ARTICLES + N_DEV_ARTICLES]


def load_rows(split_file: str, articles: set[str], art_of: dict[str, str]) -> list[dict]:
    """Rows of the chosen articles with a space after each danda. Rows without an answer are kept so that their contexts become distractor passages."""
    rows = json.load(open(json.loads(PATHS.read_text())[split_file], encoding="utf-8"))
    out = []
    for r in rows:
        if art_of.get(r["id"]) not in articles:
            continue
        answers = r["answers"]
        text, offset = (answers["text"][0], int(answers["answer_start"][0])) if answers["text"] else ("", -1)
        context, new_offset = space_after_danda(r["context"], offset if offset >= 0 else 0)
        if text and context[new_offset:new_offset + len(text)] != text:
            continue  # the answer spans a danda: the text would change
        out.append({"id": r["id"], "context": context, "question": r["question"].strip(),
                    "answers": {"text": [text] if text else [], "answer_start": [new_offset] if text else []}})
    return out


def good_question(q, stop: frozenset[str]) -> str | None:
    """None when the machine-translated pair looks usable, else the reason it is dropped."""
    words = q.question.split()
    if len(words) < 4 or len(words) > 40:
        return "question_length"
    answer = q.answer.strip()
    if len(answer) < 3 or len(answer) > 60:
        return "answer_length"
    tokens = answer.split()
    if all(t.strip(".,;:()\"'") in stop for t in tokens):
        return "answer_function_word"
    if answer in q.question:
        return "answer_inside_question"
    return None


def usable_questions(questions, art_of) -> tuple[list, Counter]:
    from ragharness.policy import _STOP_ML

    stop = _STOP_ML["hi"]
    kept, rejected = [], Counter()
    for q in questions:
        reason = good_question(q, stop)
        if reason:
            rejected[reason] += 1
        else:
            kept.append(q)
    return kept, rejected


def pick(questions, art_of, articles: list[str], n: int, seed: int) -> list:
    """`n` questions spread round-robin over the articles (seeded)."""
    by_art: dict[str, list] = {a: [] for a in articles}
    for q in sorted(questions, key=lambda q: q.qid):
        by_art.setdefault(art_of[q.qid], []).append(q)
    rng = random.Random(seed)
    for v in by_art.values():
        rng.shuffle(v)
    order = [a for a in by_art if by_art[a]]
    rng.shuffle(order)
    out = []
    while len(out) < n and any(by_art[a] for a in order):
        for a in order:
            if by_art[a] and len(out) < n:
                out.append(by_art[a].pop())
    return out


def hindi_world(articles: list[str], split_file: str = "train"):
    """(questions, holders, rejections, article of each question id) for the Hindi rows of these articles."""
    from ragharness import benchmarks as B

    art_of = english_articles()
    rows = load_rows(split_file, set(articles), art_of)
    questions, holders, rej = B.load_squad_rows(rows, "hi")
    for q in questions:
        q.evidence_scope = "corpus"
    return questions, holders, rej, art_of


# ---------------------------------------------------------------------------------------------------------- bench folders


def write_bench(name: str, lang: str, holders, evalq, manifest: dict) -> dict:
    from ragharness import corpus as corpus_mod, data
    from ragharness.retrieval import open_verified_backend

    corp = corpus_mod.build_corpus(holders + evalq, seed=SEED)
    out = BENCH / name
    out.mkdir(parents=True, exist_ok=True)
    db = out / "corpus.sqlite"
    corpus_mod.build_index(corp, db, tokenizer=TOKENIZER)
    open_verified_backend(db, corp)
    data.write_questions(out / "questions.jsonl", evalq)
    corpus_mod.save_corpus(out / "corpus.json", corp)
    lengths = sorted(len(c["text"]) for c in corp.chunks)
    man = {"benchmark": name, "lang": lang, "tokenizer": TOKENIZER, "seed": SEED, "n_questions": len(evalq), "n_chunks": len(corp.chunks),
           "index_hash": corp.index_hash, "passage_chars_median": lengths[len(lengths) // 2], "passages_over_800_chars": sum(n > 800 for n in lengths),
           "ids": [q.qid for q in evalq], **manifest}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    man["file_sha256"] = {n: sha256(out / n) for n in ("questions.jsonl", "corpus.json", "manifest.json")}
    return man


def stage_dev() -> None:
    train_art, dev_art = split_articles(article_pool())
    questions, holders, rej, art_of = hindi_world(dev_art)
    usable, bad = usable_questions(questions, art_of)
    evalq = pick(usable, art_of, dev_art, N_DEV_QUESTIONS, SEED + 1)
    man = write_bench("hi_dev", "hi", holders, evalq, {
        "source": "l3cube-pune/indic-squad Hindi/train.json (CC BY 4.0)", "articles": dev_art, "load_rejections": dict(rej),
        "quality_rejections": dict(bad), "article_of": {q.qid: art_of[q.qid] for q in evalq}})
    print("hi_dev", {k: v for k, v in man.items() if k not in ("ids", "article_of", "articles")})
    dump(ROOT / "reports" / "hindi_dev_selection.json", {k: v for k, v in man.items() if k != "ids"} | {"ids": man["ids"]})


def stage_xq2() -> None:
    import pyarrow.parquet as pq

    import prepare_benchmarks as PB
    import prepare_xquad as PX
    from ragharness import benchmarks as B

    loaded = {}
    for lang in ("hi", "en"):
        path = PB.hub_file("google/xquad", f"xquad.{lang}/validation-00000-of-00001.parquet")
        questions, holders, rej = B.load_xquad(path, lang)
        loaded[lang] = ({q.qid: q for q in questions}, holders, dict(rej))
    in_all = set(loaded["hi"][0]) & set(loaded["en"][0])
    earlier = PX.earlier_ids()
    first = json.loads((ROOT / "reports" / "xquad_selection.json").read_text(encoding="utf-8"))["ids"]
    valid = sorted(in_all - earlier - set(first))
    articles = PX.article_labels()
    chosen = PX.select(valid, articles, N_XQ2, SEED)
    print(f"{len(in_all)} valid in hi and en, {len(first)} used by xq_*, {len(valid)} available, {len(chosen)} chosen", flush=True)
    report = {"seed": SEED, "source": "google/xquad (CC BY-SA 4.0)", "excluded_first_selection": len(first), "available": len(valid),
              "chosen": len(chosen), "articles_in_sample": len({articles[i] for i in chosen}), "ids": chosen, "benches": {}}
    for lang in ("hi", "en"):
        qmap, holders, rej = loaded[lang]
        man = write_bench(f"xq2_{lang}", lang, holders, [qmap[i] for i in chosen], {
            "source": "google/xquad", "license": "CC BY-SA 4.0", "load_rejections": rej, "article_of": {i: articles[i] for i in chosen},
            "first_selection": "reports/xquad_selection.json"})
        report["benches"][f"xq2_{lang}"] = {k: v for k, v in man.items() if k not in ("ids", "article_of")}
        print(f"xq2_{lang}", report["benches"][f"xq2_{lang}"], flush=True)
    dump(ROOT / "reports" / "hindi_test_selection.json", report)


# ---------------------------------------------------------------------------------------------------------- training set


def stage_train(n_questions: int, replay_share: float) -> None:
    from transformers import AutoTokenizer

    from ragharness import builder, corpus as corpus_mod, episodes, mixture, protocol, trainer
    from ragharness.retrieval import MaskedSearch, open_verified_backend
    from ragharness.tools import CorpusSearchTool

    train_art, dev_art = split_articles(article_pool())
    assert not set(train_art) & set(dev_art)
    questions, holders, rej, art_of = hindi_world(train_art)
    usable, bad = usable_questions(questions, art_of)
    selected = pick(usable, art_of, train_art, n_questions, SEED)
    print(f"{len(questions)} questions loaded, {len(usable)} usable, {len(selected)} selected from {len({art_of[q.qid] for q in selected})} articles", flush=True)

    corp = corpus_mod.build_corpus(holders + selected, seed=SEED, id_start=TRAIN_ID_START)
    db = ROOT / "data" / "index" / "corpus_trainhi.sqlite"
    corpus_mod.build_index(corp, db, tokenizer=TOKENIZER)
    backend = open_verified_backend(db, corp)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    system = protocol.system_prompt(tool)
    main_rows = [json.loads(l) for l in (TRAIN / "main.jsonl").read_text(encoding="utf-8").splitlines()]
    assert system == main_rows[0]["prompt"][0]["content"], "the system prompt must be the one main was trained with"

    built = [b for q in selected for b in builder.build_question(q, "trainhi", corp, tool, search)]
    accepted = [b for b in built if b.episode is not None]
    logs, metas = [], []
    for b in accepted:
        ep = b.episode
        metas.append({"parent_qid": b.qid, "split": "trainhi", "state": b.state, "variant": b.variant, "answerable": b.answerable, "qtype": "single",
                      "n_searches": ep.n_searches, "final": ep.final.turn.kind, "family": "hindi"})
        logs.append(episodes.to_eventlog(ep))
    episodes.write_eventlogs(EP / "eventlogs_trainhi.jsonl", logs)
    jsonl(EP / "episodes_trainhi.jsonl", [{"meta": m} for m in metas])
    collector = trainer.RowCollector(system, metas)
    trainer.build_trainer(logs, collector).train()
    rows = collector.rows
    tok = AutoTokenizer.from_pretrained(str(ROOT / "runs/kaggle/ats-train-both/out/main/adapter"), local_files_only=True)
    mixture.annotate(rows, tok)
    for r in rows:
        r["meta"]["real_tokens"] = r["meta"]["est_tokens"]
    jsonl(EP / "rows_trainhi.jsonl", rows)
    hindi = mixture.build_main(rows, SEED, MAX_ROW_TOKENS, max_ratio=1.0)
    hindi_tokens = sum(r["meta"]["est_tokens"] for r in hindi)
    longest = max(r["meta"]["est_tokens"] for r in hindi)
    print(f"{len(accepted)} episodes, {len(rows)} rows, {len(hindi)} Hindi rows chosen, {hindi_tokens} tokens, longest {longest}", flush=True)

    rng = random.Random(SEED)
    sets = {}
    for name, source in (("hiMain", "main.jsonl"), ("hiF", "candF.jsonl")):
        replay_pool = [json.loads(l) for l in (TRAIN / source).read_text(encoding="utf-8").splitlines()]
        by_kind = {k: [r for r in replay_pool if r["meta"]["kind"] == k] for k in ("search", "answer", "abstain")}
        for group in by_kind.values():
            rng.shuffle(group)
        target, replay, spent, counts = replay_share * hindi_tokens, [], 0, Counter()
        while spent < target and any(by_kind.values()):
            kind = min((k for k in by_kind if by_kind[k]), key=lambda k: counts[k])
            row = by_kind[kind].pop()
            n = mixture.rowmod.real_tokens(row, tok)[0]
            row["meta"]["real_tokens"] = n
            replay.append(row)
            spent += n
            counts[kind] += 1
        chosen = hindi + replay
        rng.shuffle(chosen)
        keys = {(r["meta"]["parent_qid"], r["meta"]["variant"], r["meta"]["step"]) for r in chosen}
        assert len(keys) == len(chosen), "duplicate training keys"
        jsonl(TRAIN / f"{name}.jsonl", chosen)
        sets[name] = {"rows": len(chosen), "hindi_rows": len(hindi), "replay_rows": len(replay), "replay_source": source, "hindi_tokens": hindi_tokens,
                      "replay_tokens": spent, "tokens_per_epoch": hindi_tokens + spent, "by_kind": dict(Counter(r["meta"]["kind"] for r in chosen)),
                      "sha256": sha256(TRAIN / f"{name}.jsonl")}
    report = {"seed": SEED, "source": "l3cube-pune/indic-squad Hindi/train.json (CC BY 4.0)", "train_articles": train_art, "dev_articles": dev_art,
              "english_pool": "reserve of reports/round5_allocation.json minus reports/round6_allocation.json r6test",
              "questions_loaded": len(questions), "usable": len(usable), "selected": len(selected), "load_rejections": dict(rej), "quality_rejections": dict(bad),
              "episodes_accepted": len(accepted), "episodes_by_variant_state": dict(sorted(Counter(f"{b.variant}:{b.state}" for b in accepted).items())),
              "rejected_or_unreachable": dict(sorted(Counter(f"{b.variant}:{b.reject_reason}" for b in built if b.episode is None).items())),
              "rows": len(rows), "hindi_rows_chosen": len(hindi), "hindi_tokens_per_epoch": hindi_tokens, "longest_row_tokens": longest,
              "max_row_tokens": MAX_ROW_TOKENS, "replay_share": replay_share, "index_hash": corp.index_hash, "sets": sets,
              "parent_questions": len({r["meta"]["parent_qid"] for r in hindi})}
    dump(ROOT / "reports" / "hindi_training_data.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("train_articles", "dev_articles")}, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "xq2", "train"])
    ap.add_argument("--questions", type=int, default=600)
    ap.add_argument("--replay-share", type=float, default=0.5)
    args = ap.parse_args()
    {"dev": stage_dev, "xq2": stage_xq2, "train": lambda: stage_train(args.questions, args.replay_share)}[args.stage]()


if __name__ == "__main__":
    main()
