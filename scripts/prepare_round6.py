"""Build fresh Round 6 benchmark inputs without reading model outputs."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import prepare_benchmarks as benchmarks
import prepare_round3 as round3_builder
import prepare_round4 as round4_builder
from ragharness import corpus, data


ROOT = Path(__file__).resolve().parents[1]
PARTS = ("r6t_squad", "r6t_hotpot", "r6t_rgb")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_oracles():
    from ragharness import rescore

    path = ROOT / "reports/round6_test_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for part in PARTS:
        folder = ROOT / "data/bench" / part
        for name, expected in manifest["sets"][part]["file_sha256"].items():
            if sha(folder / name) != expected:
                raise ValueError(f"Round 6 sealed file changed: {folder / name}")
        corp = corpus.load_corpus(folder / "corpus.json")
        questions = {q.qid: q for q in data.read_questions(folder / "questions.jsonl")}
        trace_path = ROOT / "runs/oracle_r6" / part / "strict/transcripts.jsonl"
        with trace_path.open(encoding="utf-8") as f:
            traces = [json.loads(line) for line in f if line.strip()]
        scores = [rescore.score(trace, questions[trace["qid"]], corp) for trace in traces]
        eligible = [score for score in scores if not score["excluded_v3"]]
        failed = [score for score in eligible if not score["strong_v3"]]
        if not eligible or failed:
            raise ValueError(f"Round 6 oracle failed {len(failed)} of {len(eligible)} eligible episodes: {part}")
        manifest["sets"][part].update(episodes=len(scores), eligible=len(eligible), oracle_success=len(eligible))
        manifest["sets"][part]["file_sha256"]["corpus.sqlite"] = sha(folder / "corpus.sqlite")
        print(f"{part}: oracle passed {len(eligible)}/{len(eligible)} eligible episodes", flush=True)
    write_json(path, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="validate saved scripted-oracle traces")
    if parser.parse_args().verify:
        verify_oracles()
        return
    allocation_path = ROOT / "reports/round5_allocation.json"
    if not allocation_path.exists():
        raise FileNotFoundError(allocation_path)
    if (ROOT / "reports/round6_test_manifest.json").exists():
        raise SystemExit("Round 6 is already sealed: reports/round6_test_manifest.json")
    for part in PARTS:
        if (ROOT / "data/bench" / part / "questions.jsonl").exists():
            raise SystemExit(f"Round 6 data already exists: data/bench/{part}/questions.jsonl")

    previous = json.loads(allocation_path.read_text(encoding="utf-8"))["allocation"]
    articles = list(previous["reserve"])
    random.Random(61).shuffle(articles)
    articles = articles[:25]
    write_json(ROOT / "reports/round6_allocation.json", {"seed": 61, "r6test": articles})
    questions, contexts, _ = round3_builder.squad_train()
    for question in questions:
        question.qtype = "single"
    squad, _ = round4_builder.build_group("r6t_squad", questions, contexts, articles, 80, 40)
    print("SQuAD questions", squad["n_questions"], flush=True)

    man = json.loads((ROOT / "data/manifests/split_manifest.json").read_text(encoding="utf-8"))
    previous_sets = [data.read_questions(ROOT / "data/processed" / f"questions_{name}.jsonl")
                     for name in ("train", "dev", "test", "test2")]
    previous_sets += [data.read_questions(ROOT / "data/bench" / name / "questions.jsonl")
                      for name in ("r3t_hotpot", "r5t_hotpot")]
    prior = corpus.load_corpus(ROOT / "data/bench/r5t_hotpot/corpus.json")
    blocked_titles = {chunk["title"] for chunk in prior.chunks}
    used = {question.qid for group in previous_sets for question in group}
    files = data.download(ROOT / ".cache/huggingface/hub")
    validation, rejected = data.load_questions(files[data.VAL_FILE])
    selected = data.take_disjoint(data.stratified_order(validation, man["seed"] + 11), 60, blocked_titles, used)
    extended = corpus.extend_corpus(prior, selected)
    prior_ids = {chunk["chunk_id"] for chunk in prior.chunks}
    assert not {cid for question in selected for cid in extended.support_copies(question)} & prior_ids
    from ragharness.evidence import normalize
    prior_texts = {normalize(chunk["text"]) for chunk in prior.chunks}
    assert not any(normalize(extended.chunk(cid)["text"]) in prior_texts
                   for question in selected for cid in extended.support_chunks(question).values())
    out = ROOT / "data/bench/r6t_hotpot"
    out.mkdir(parents=True, exist_ok=True)
    data.write_questions(out / "questions.jsonl", selected)
    corpus.save_corpus(out / "corpus.json", extended)
    corpus.build_index(extended, out / "corpus.sqlite")
    write_json(out / "manifest.json", {"benchmark": "r6t_hotpot", "dataset": data.DATASET,
               "revision": data.REVISION, "file": data.VAL_FILE, "license": "CC BY-SA 4.0",
               "order_seed": man["seed"] + 11, "n_questions": len(selected), "n_chunks": len(extended.chunks),
               "index_hash": extended.index_hash, "load_rejections": dict(rejected), "ids": [q.qid for q in selected]})
    print("HotpotQA questions", len(selected), flush=True)

    skip = set()
    for name in ("rgb", "r3t_rgb", "r5t_rgb"):
        old_manifest = json.loads((ROOT / "data/bench" / name / "manifest.json").read_text(encoding="utf-8"))
        skip.update(old_manifest["ids"])
    remaining = 300 - len(skip)
    if remaining <= 0:
        raise ValueError("No unused RGB questions remain in the source.")
    rgb = benchmarks.build("rgb", out_name="r6t_rgb", n_eval=remaining, fresh=True, skip=frozenset(skip))
    if not rgb["n_questions"]:
        raise ValueError("No unused RGB questions passed the benchmark preparation filters.")
    print("RGB questions", rgb["n_questions"], flush=True)

    sets = {}
    for part in PARTS:
        folder = ROOT / "data/bench" / part
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        names = ["questions.jsonl", "corpus.json", "manifest.json"]
        if (folder / "prompt_variants.json").exists():
            names.append("prompt_variants.json")
        sets[part] = {"questions": manifest["n_questions"], "index_hash": manifest["index_hash"],
                      "file_sha256": {name: sha(folder / name) for name in names}}
    write_json(ROOT / "reports/round6_test_manifest.json", {"seed": 61, "sets": sets})
    print("Round 6 inputs sealed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required saved artifact is missing: {exc.filename or exc}") from exc
