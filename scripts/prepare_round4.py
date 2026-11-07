"""Prepare article-disjoint evidence decisions and paired training rows, before any GPU run."""

import hashlib
import json
import random
import argparse
from collections import Counter
from pathlib import Path

import prepare_round3 as R3
from ragharness import builder, corpus as C, data, episodes, protocol, round3, round4, rows, trainer
from ragharness.retrieval import MaskedSearch, open_verified_backend
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
EP = ROOT / "data/processed/episodes"
SEED = 47


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def pick(qs, articles, ns, nc):
    singles = round3.pick_singles(qs, articles, ns, SEED)
    used = {q.qid for q in singles}
    paragraphs = {q.paragraphs[0].title for q in singles}
    pool = [q for q in qs if q.qid not in used and q.paragraphs[0].title not in paragraphs]
    compounds = round3.pick_compounds(pool, articles, nc, SEED + 1)
    return singles + compounds


def hidden_sets(q):
    titles = list(q.support)
    return [[t] for t in titles] + ([titles] if len(titles) > 1 else [])


def build_group(name, qs, ctx, articles, n_single, n_compound, training=False):
    selected = pick(qs, articles, n_single, n_compound)
    context = [q for q in ctx if round3.article_of(q.paragraphs[0].title) in set(articles)]
    corp = C.build_corpus(context + selected, seed=SEED, id_start=40000 if training else 0)
    out = ROOT / ("data/index" if training else f"data/bench/{name}")
    out.mkdir(parents=True, exist_ok=True)
    db = out / ("corpus_train4.sqlite" if training else "corpus.sqlite")
    C.build_index(corp, db)
    backend = open_verified_backend(db, corp)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    system = protocol.system_prompt(tool)
    counts = Counter()
    manifest = {"seed": SEED, "articles": articles, "index_hash": corp.index_hash, "n_chunks": len(corp.chunks),
                "source": "rajpurkar/squad_v2 train; CC BY-SA 4.0; article-disjoint from round 3"}
    if training:
        logs, metas, pair_decisions = [], [], []
        for k, q in enumerate(selected):
            for hs in hidden_sets(q):
                for depth in range(4):
                    pair = round4.build_pair(q, corp, search, tool, depth, hs)
                    if pair is None:
                        counts[f"rejected:{q.qtype}:d{depth}"] += 1
                        continue
                    for ep in pair:
                        meta = {"parent_qid": q.qid, "split": "train4", "variant": ep.condition,
                                "family": q.qtype, "qtype": q.qtype, "depth": depth, "selected_step": depth,
                                "pair_id": q.qid + ":" + ep.condition.rsplit("_", 1)[0],
                                "answerable": ep.final.turn.kind == "answer", "state": "R4",
                                "n_searches": ep.n_searches, "final": ep.final.turn.kind}
                        logs.append(episodes.to_eventlog(ep))
                        metas.append(meta)
                    counts[f"paired:{q.qtype}:d{depth}"] += 1
            if k % 40 == 0:
                print(f"training parents {k}/{len(selected)}, episodes {len(logs)}", flush=True)
        episodes.write_eventlogs(EP / "eventlogs_train4.jsonl", logs)
        jsonl(EP / "episodes_train4.jsonl", [{"meta": m} for m in metas])
        collector = trainer.RowCollector(system, metas)
        trainer.build_trainer(logs, collector).train()
        for row in collector.rows:
            if row["meta"]["step"] == row["meta"]["selected_step"]:
                pair_decisions.append(row)
        jsonl(EP / "rows_train4.jsonl", pair_decisions)
        manifest.update(episodes=len(logs), paired_decisions=len(pair_decisions), counts=dict(counts), ids=[q.qid for q in selected])
        return manifest, pair_decisions

    variants, kept = {}, []
    for q in selected:
        specs = {b.variant: b for b in builder.build_question(q, name, corp, tool, search)}
        if "open" not in specs or specs["open"].episode is None:
            counts["unreachable"] += 1
            continue
        full = round4.build_pair(q, corp, search, tool, 0, hidden_sets(q)[0])
        if full is None:
            counts["no_valid_pair"] += 1
            continue
        kept.append(q)
        vs = {"r4_full": {"prefill": [h.chunk_id for h in full[0].prefill], "masked": [], "answerable": True}}
        for i, hs in enumerate(hidden_sets(q)):
            pair = round4.build_pair(q, corp, search, tool, 0, hs)
            if pair is None:
                continue
            neg = pair[1]
            ids = [h.chunk_id for h in neg.prefill]
            key = "partial" if len(hs) < len(q.support) else "missing"
            vs[f"r4_{key}_{i}"] = {"prefill": ids, "masked": sorted(neg.masked), "answerable": False}
            # Same initial input as the hidden case, but a normal search can recover the missing fact.
            recover = episodes.run_scripted(q, corp, tool, "recoverable", prefill=neg.prefill, finish="answer_if_complete")
            if recover.steps and recover.complete:
                vs[f"r4_recoverable_{i}"] = {"prefill": ids, "masked": [], "answerable": True, "oracle_search": True}
        variants[q.qid] = vs
    data.write_questions(out / "questions.jsonl", kept)
    C.save_corpus(out / "corpus.json", corp)
    dump(out / "prompt_variants.json", variants)
    manifest.update(benchmark=name, n_questions=len(kept), ids=[q.qid for q in kept],
                    counts=dict(counts), by_type=dict(Counter(q.qtype for q in kept)),
                    prompt_variants=dict(Counter(v for vs in variants.values() for v in vs)))
    dump(out / "manifest.json", manifest)
    return manifest, []


def choose_training(paired, tokenizer):
    rng = random.Random(SEED)
    pairs = {}
    for row in paired:
        pairs.setdefault(row["meta"]["pair_id"], []).append(row)
    groups = {d: [] for d in range(4)}
    rejected = Counter()
    for pid, block in pairs.items():
        assert len(block) == 2
        positive = next(r for r in block if r['meta']['kind'] == 'answer')
        answer = protocol.parse_turn(positive['completion'][0]['content']).answer
        if any(builder.answer_in_text(part.strip(), '') is None for part in answer.split(';')):
            rejected['ambiguous_short_answer_pairs'] += 1
            continue
        lengths = [rows.real_tokens(r, tokenizer)[0] for r in block]
        if max(lengths) > 2560:
            rejected["overlength_pairs"] += 1
            continue
        for row, length in zip(block, lengths):
            row["meta"]["real_tokens"] = length
        groups[block[0]["meta"]["depth"]].append((block, sum(lengths)))
    for group in groups.values():
        rng.shuffle(group)
    weights = {0: .15, 1: .15, 2: .10, 3: .60}
    selected, tokens = [], 0
    spent = {d: 0 for d in groups}
    while tokens < 600_000 and any(groups.values()):
        d = min((d for d in groups if groups[d]), key=lambda d: spent[d] / weights[d])
        block, size = groups[d].pop()
        selected.extend(block)
        tokens += size
        spent[d] += size
    target_tokens = tokens
    replay = [json.loads(l) for l in (ROOT / "data/processed/training/main.jsonl").read_text(encoding="utf-8").splitlines()]
    by_kind = {k: [r for r in replay if r["meta"]["kind"] == k] for k in ("search", "answer", "abstain")}
    for group in by_kind.values():
        rng.shuffle(group)
    replay_tokens, kind_counts = 0, Counter()
    while replay_tokens < target_tokens and any(by_kind.values()):
        kind = min((k for k in by_kind if by_kind[k]), key=lambda k: kind_counts[k])
        row = by_kind[kind].pop()
        n = rows.real_tokens(row, tokenizer)[0]
        if n > 2560:
            continue
        row["meta"]["real_tokens"] = n
        selected.append(row)
        replay_tokens += n
        kind_counts[kind] += 1
    rng.shuffle(selected)
    assert len({(r['meta']['parent_qid'], r['meta']['variant'], r['meta']['step']) for r in selected}) == len(selected)
    for name in ("candE", "candF"):
        jsonl(ROOT / f"data/processed/training/{name}.jsonl", selected)
    return {"rows": len(selected), "new_tokens": target_tokens, "replay_tokens": replay_tokens,
            "total_tokens_per_epoch": target_tokens + replay_tokens, "new_tokens_by_depth": spent,
            "new_rows": sum(r['meta']['split'] == 'train4' for r in selected),
            "by_kind": dict(Counter(r['meta']['kind'] for r in selected)), "rejected": dict(rejected),
            "sha256": hashlib.sha256((ROOT / 'data/processed/training/candE.jsonl').read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--select-only', action='store_true', help='reapply deterministic row validation/selection to the saved generated pairs')
    args = ap.parse_args()
    if args.select_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ROOT / 'runs/kaggle/ats-train-both/out/main/adapter'), local_files_only=True)
        report = json.loads((ROOT / 'reports/round4_data.json').read_text(encoding='utf-8'))
        pairs = [json.loads(l) for l in (EP / 'rows_train4.jsonl').read_text(encoding='utf-8').splitlines()]
        report['training'] = choose_training(pairs, tok)
        dump(ROOT / 'reports/round4_data.json', report)
        print(json.dumps(report['training'], indent=2), flush=True)
        return
    qs, ctx, _ = R3.squad_train()
    articles = sorted({round3.article_of(q.paragraphs[0].title) for q in ctx})
    old_dev, old_train = round3.split_articles(articles, R3.SEED, R3.N_DEV_ARTICLES, R3.N_TRAIN_ARTICLES)
    remaining = sorted(set(articles) - set(old_dev) - set(old_train))
    random.Random(SEED).shuffle(remaining)
    groups = {'train': remaining[:60], 'dev': remaining[60:75], 'test': remaining[75:95]}
    assert all(not set(groups[a]) & set(groups[b]) for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')])
    report = {"seed": SEED, "excluded_old_articles": old_dev + old_train, "groups": {}}
    pairs = []
    for name, ns, nc in [('train', 240, 100), ('dev', 60, 30), ('test', 80, 40)]:
        for q in qs:
            q.qtype = 'single'
        manifest, rr = build_group('r4' + name, qs, ctx, groups[name], ns, nc, training=name == 'train')
        report['groups'][name] = manifest
        pairs += rr
        print(name, {k: v for k, v in manifest.items() if k not in ('articles', 'ids')}, flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'runs/kaggle/ats-train-both/out/main/adapter'), local_files_only=True)
    report['training'] = choose_training(pairs, tok)
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not set(report['groups'][a]['ids']) & set(report['groups'][b]['ids'])
    dump(ROOT / 'reports/round4_data.json', report)
    print(json.dumps(report['training'], indent=2), flush=True)


if __name__ == '__main__':
    main()
