"""Reproducible November 13 audit remediation from local artifacts; no model calls.

python scripts/remediate_audit.py
Preserves sealed data, original transcripts, checkpoints and historical report tables.
"""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import report_azure_compare as A
import report_deepseek_compare as D
from ragharness import audit, benchmarks, corpus as C, data, evidence, lineage, metrics, product, protocol, rescore
from ragharness.episodes import Episode, Step
from ragharness.retrieval import Hit

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-11-13"
OUT = ROOT / "reports"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def squad_sources():
    sources = {}
    for path in (ROOT / ".cache/huggingface/hub/datasets--rajpurkar--squad_v2/snapshots").glob("*/squad_v2/*.parquet"):
        for row in pq.read_table(path, columns=["id", "context", "answers", "title"]).to_pylist():
            sources[row["id"]] = row
    if not sources:
        raise FileNotFoundError("cached official SQuAD sources required to validate offsets")
    return sources


def corrected_question(q, sources):
    q = copy.deepcopy(q)
    components = q.qid.removeprefix("cmp-").split("-") if q.qid.startswith("cmp-") else [q.qid]
    for component, paragraph in zip(components, q.paragraphs):
        row = sources.get(component)
        if not row or not row["answers"]["text"]:
            continue
        span = benchmarks.annotated_span(row["context"], row["answers"]["text"][0], int(row["answers"]["answer_start"][0]))
        if span is None:
            raise ValueError(f"invalid official SQuAD offset: {component}")
        sid, offsets = span
        q.support[paragraph.title] = [sid]
        q.support_spans[paragraph.title] = [offsets]
    return q


def mask_for(q, corpus, variant, variants):
    if variant in variants.get(q.qid, {}):
        return variants[q.qid][variant]["masked"]
    return list(corpus.support_copies(q)) if variant == "support_masked" else []


def diagnostic(row, transcript, q, corpus):
    """Exactly the two original audit rescues, with denominator/wrappers fixed."""
    if row["answerable"] and row["final"] == "answer":
        cited = set(transcript["final"]["citations"] or [])
        texts = evidence.shown_texts(transcript, corpus)
        exact = row["strong_v1"] and rescore.precision_ok(transcript["final"]["answer"], q) == (True, 1.0)
        accepted = set(corpus.support_chunks(q).values()) | set(corpus.extra_copies.get(q.qid, ()))
        alternative = q.qid.startswith("rgb-") and row["good_v2"] and rescore.fully_visible(q, corpus, cited & accepted & set(texts))
        if exact or alternative:
            return {**row, "strong_v2": True, "good_v2": True, "outcome_v2": "supported_correct"}
    return row


def as_v2(row):
    return {**row, **{f"{key}_v2": row[f"{key}_v3"] for key in ("outcome", "strong", "good", "excluded")}}


def load(folder, part, sources):
    corpus, questions = A.RS.world("bench:" + part)
    variant_path = ROOT / "data/bench" / part / "prompt_variants.json"
    variants = json.loads(variant_path.read_text(encoding="utf-8")) if variant_path.exists() else {}
    original, corrected, diagnosed = [], [], []
    for t, stored in zip(A.RS.read(folder / "transcripts.jsonl"), A.RS.read(folder / "scores.jsonl"), strict=True):
        q = questions[t["qid"]]
        row = rescore.score(t, q, corpus)
        if row["outcome_v1"] != stored["outcome"] or row["strong_v1"] != stored.get("evidence_supported_good", row["strong_v1"]):
            raise ValueError(f"historical reproduction mismatch: {folder}/{q.qid}/{t['variant']}")
        row.update(part=part, comp=A.R.component(part, q.qtype), stop=t.get("stop"))
        original.append(row)
        diagnosed.append(diagnostic(row, t, q, corpus))
        enriched = {**t, "masked": mask_for(q, corpus, t["variant"], variants), "max_chars": 800}
        updated_q, updated_corpus = corrected_question(q, sources), corpus
        # The user-supplied audit establishes this grounded answer in the legacy
        # corpus-wide condition; the future SQuAD condition stays paragraph-bounded.
        if part == "squad2" and q.qid == "5ad27473d7d075001a4294e8":
            updated_corpus = copy.copy(corpus)
            updated_corpus.by_question = {**corpus.by_question, q.qid: {**corpus.by_question[q.qid], "French and Indian War (part 2)": "c00917"}}
            ch = corpus.chunk("c00917")
            updated_q.paragraphs = [data.Paragraph(ch["title"], benchmarks.sentences(ch["text"]))]
            updated_q.support = {ch["title"]: [2]}
            updated_q.answer, updated_q.aliases = "1754", []
            updated_q.natural_unanswerable, updated_q.evidence_scope = False, "corpus"
            enriched["answerable"] = True
        fresh = rescore.score(enriched, updated_q, updated_corpus)
        if part == "musique":
            same = [other.qid for other in questions.values() if (evidence.normalize(other.question), evidence.normalize(other.answer))
                    == (evidence.normalize(q.question), evidence.normalize(q.answer))]
            if q.qid != min(same):
                fresh["excluded_v3"] = True
                fresh["eligibility_v3"] = {**fresh["eligibility_v3"], "eligible": False, "reason": "duplicate_question_answer"}
        corrected.append(as_v2({**row, "answerable": fresh["answerable"], **{k: v for k, v in fresh.items() if k.endswith("_v3")}}))
    return original, diagnosed, corrected


def wrappers(rows, cases, verdicts):
    result = dict(rows)
    for name, rr in rows.items():
        result[name + "+FB"] = A.S.fail_closed(rr)
        answer_keys = {(f"{name}_bench_{r['part']}", r["qid"], r["variant"]) for r in rr if r["final"] == "answer"}
        if name in ("base", "main", "candA", "candF1") and answer_keys <= set(cases):
            result[name + "+FB+V"] = sum((A.S.apply_v(A.S.fail_closed([r for r in rr if r["part"] == part]),
                f"{name}_bench_{part}", cases, verdicts) for part in A.PARTS), [])
    return result


def fixed_common(systems):
    indexed = {name: A.by_key(rows) for name, rows in systems.items()}
    present = set.intersection(*(set(rows) for rows in indexed.values()))
    reference = indexed["main"]
    for key in present:
        if any(rr[key]["excluded_v2"] != reference[key]["excluded_v2"] for rr in indexed.values()):
            raise ValueError(f"model-dependent v3 eligibility: {key}")
    keys = sorted(k for k in present if not reference[k]["excluded_v2"])
    excluded = [{"part": k[0], "qid": k[1], "variant": k[2], "reason": reference[k]["eligibility_v3"]["reason"]}
                for k in sorted(present - set(keys))]
    return indexed, keys, {"expected": len(reference), "present": len(present), "common": len(keys),
                           "unavailable": len(reference) - len(present), "excluded": excluded}


def clustered(indexed, keys, pairs, reps=3000):
    # SQuAD variants share an article, so resample articles rather than just QIDs.
    _, qs = A.RS.world("bench:r5t_squad")
    clusters = {}
    for i, (part, qid, _) in enumerate(keys):
        group = qs[qid].paragraphs[0].title.rsplit(" (part ", 1)[0] if part == "r5t_squad" else qid
        clusters.setdefault(part, {}).setdefault(group, []).append(i)
    rng = np.random.default_rng(47)
    comps, vectors = A.R.vectors([indexed[next(iter(indexed))][k] for k in keys], keys)
    sums = {}
    for name, rows in indexed.items():
        _, vec = A.R.vectors([rows[k] for k in keys], keys)
        sums[name] = {part: np.array([vec[idx].sum(axis=0) for idx in groups.values()]) for part, groups in clusters.items()}
    totals = {name: np.zeros((reps, vectors.shape[1])) for name in indexed}
    for part, groups in sorted(clusters.items()):
        picks = rng.integers(0, len(groups), (reps, len(groups)))
        for name in indexed:
            totals[name] += sums[name][part][picks].sum(axis=1)
    sampled = {name: [A.R.stats_from(total, comps) for total in rr] for name, rr in totals.items()}
    table = {name: A.r5_stats(rr, keys) for name, rr in indexed.items()}
    return {f"{a}/{b}": {m: {"difference": table[a][m] - table[b][m],
             "ci95": np.quantile([sa[m] - sb[m] for sa, sb in zip(sampled[a], sampled[b])], [.025, .975]).tolist()}
             for m in ("macro", "unsupported_rate", "supported_correct")} for a, b in pairs}


def table_report(name, title, indexed, keys, coverage, pairs):
    names = ("main+FB+V", "candF1+FB+V", "azure_native+FB", "azure_deepseek_native+FB")
    selected = {n: indexed[n] for n in names if n in indexed}
    comparisons = clustered(selected, keys, pairs)
    result = {"scorer": "v3", "coverage": coverage, "systems": {n: A.r5_stats(rr, keys) for n, rr in indexed.items()},
              "comparisons": comparisons, "bootstrap": "3000 article/question cluster draws, seed 47"}
    write(name, result)
    lines = ["# " + title, "", f"Corrected replay dated {DATE}; {len(keys)} shared episodes. Saved prompts, outputs, fallback and verifier are unchanged.", "",
             "Eligibility is fixed from annotated evidence and masks before scoring. Unsupported counts are annotation-based flags; unannotated alternative entailment still needs adjudication. The denominator differs from scorer v2.", "",
             "| System | Macro success | Supported correct | Flagged answers / hidden |", "| --- | ---: | ---: | ---: |"]
    for n in names:
        if n in result["systems"]:
            s = result["systems"][n]
            lines.append(f"| {n} | {s['macro']:.4f} | {s['supported_correct']:.4f} | {s['unsupported']}/{s['hidden_n']} |")
    lines += ["", "Paired differences use SQuAD article clusters and HotpotQA/RGB question clusters:", ""]
    for pair, values in comparisons.items():
        v = values["macro"]
        lines.append(f"- {pair}: {100*v['difference']:+.2f} percentage points [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}].")
    lines += ["", "These are exploratory corrections on an opened holdout. Main remains submitted; a new model-selection claim needs a fresh holdout."]
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def auditkit_recall():
    from auditkit.metrics.rag_stress import EvidenceSetRecall
    metric = EvidenceSetRecall()
    result = {}
    for part, root in (("rgb", A.OLDMAIN), ("r5t_rgb", A.R5REF)):
        corpus, questions = A.RS.world("bench:" + part)
        lookup = audit.ChunkLookup(corpus)
        statuses, values = Counter(), []
        for t in A.RS.read(root / f"main_bench_{part}/lenient/transcripts.jsonl"):
            if not t["answerable"]:
                continue
            texts = evidence.shown_texts(t, corpus)
            hits = [Hit(cid, corpus.chunk(cid)["title"], corpus.chunk(cid)["text"], 0) for cid in texts]
            f = t["final"]
            text = protocol.format_answer(f["answer"], f["citations"]) if f["kind"] == "answer" else protocol.format_abstain("no evidence") if f["kind"] == "abstain" else "invalid"
            ep = Episode(t["qid"], t["question"], t["variant"], frozenset(), prefill=hits,
                         steps=[Step(text, protocol.parse_turn(text))], max_chars=800)
            sample = audit.to_sample(ep, questions[t["qid"]], corpus, lookup, True)
            score = metric.score(sample, text)[0]
            status = "scored" if not score.metadata.get("unknown") else score.label
            statuses[status] += 1
            if status == "scored":
                values.append(score.value)
        result[part] = {"answerable": sum(statuses.values()), "statuses": dict(statuses),
                        "mean_on_scored": sum(values) / len(values) if values else None}
    return result


def external_report(sources):
    result = {}
    for part in A.OLD:
        runs = {"main": (A.OLDMAIN, "main"), "candA": (A.OLDALT, "candA"), **{n: (p, "azure") for n, p in D.REMOTE.items()}}
        systems = {n: load(root / f"{label}_bench_{part}/lenient", part, sources)[2] for n, (root, label) in runs.items()}
        systems.update({n + "+FB": A.S.fail_closed(rr) for n, rr in list(systems.items())})
        ix, keys, cov = fixed_common(systems)
        result[part] = {"coverage": cov, "systems": {n: A.old_stats(rr, keys) for n, rr in ix.items()}}
        print("external", part, cov["common"], flush=True)
    write("external_benchmarks_corrected", result)
    lines = ["# Earlier external suites: corrected replay", "", "Scorer v3, fixed evidence eligibility, official SQuAD answer offsets and duplicate MuSiQue questions removed.",
             "The French and Indian War question is adjudicated as answerable in the legacy corpus-wide condition (1754); future natural SQuAD episodes restrict evidence to their source paragraph.", "",
             "| Suite | System | Shared cases | Success | Flagged / hidden |", "| --- | --- | ---: | ---: | ---: |"]
    for part, r in result.items():
        for n in ("main", "candA", "azure_native+FB", "azure_deepseek_native+FB"):
            s = r["systems"][n]
            lines.append(f"| {part} | {n} | {s['n']} | {s['success']:.4f} | {s['unsupported']}/{s['hidden_n']} |")
    lines += ["", "No final Tiny Aya verifier was saved on these older suites. Flagged counts remain annotation-based; they are not human-adjudicated hallucination rates."]
    (OUT / "external_benchmarks_corrected.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def nimbus_report():
    suite = product.build()
    by_id = {q.qid: q for q in suite.questions}
    result = {}
    for name in ("base", "main", "control"):
        rows = []
        for t in A.RS.read(ROOT / f"runs/kaggle/ats-eval-product-v2/out/eval/{name}_product/lenient/transcripts.jsonl"):
            rows.append(rescore.score(t, by_id[t["qid"]], suite.corpus, chunk_metadata=suite.meta, decision_date=product.AS_OF))
        result[name] = {kind: {"correct": sum(r["strong_v3"] for r in rows if r["qtype"] == kind), "n": sum(r["qtype"] == kind for r in rows)}
                        for kind in ("ordinary", "two_doc", "dated", "missing")}
    write("nimbus_corrected", result)
    lines = ["# Nimbus Home: corrected citation and freshness replay", "", "Both required documents must be cited for two-document tasks; cited editions must be current at the original 2026-10-15 decision date.", "",
             "| Task kind | Base | Main | Control |", "| --- | ---: | ---: | ---: |"]
    for kind in ("ordinary", "two_doc", "dated", "missing"):
        lines.append("| " + kind + " | " + " | ".join(f"{result[n][kind]['correct']}/{result[n][kind]['n']}" for n in ("base", "main", "control")) + " |")
    (OUT / "nimbus_corrected.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def dated_markdown(report):
    diag, revised = report["two_bug_replay"], report["corrected_comparison"]
    safety_upper = report["local_replay"]["comparisons"]["candF1+FB+V/main+FB+V"]["unsupported_rate"]["ci95"][1]
    names = ("main+FB+V", "candF1+FB+V", "azure_native+FB", "azure_deepseek_native+FB")
    lines = ["# Evaluation and training-data remediation — 13 November 2026", "",
             "**Main with fail-closed fallback and the saved verifier remains the submitted system.** The evaluation bugs are corrected, later preference data is quarantined and cleaned, and affected documentation is updated. This work uses saved transcripts and local CPU computation. No checkpoint was retrained or promoted, and no GPU or hosted-model call was made.", "",
             "## The same 863 cases: isolate the two scoring bugs", "",
             "This diagnostic preserves the original comparison intersection, prompts, masks, exclusions, fallback and verifier decisions. It restores exact accepted answers and independently sufficient, displayed RGB alternatives. It does not replace the eligibility or source annotations in the historical test.", "",
             "| System | Published macro | Two-bug replay | Original flagged / hidden |", "| --- | ---: | ---: | ---: |"]
    for n in names:
        old, fixed = diag["published"][n], diag["corrected_two_bugs"][n]
        lines.append(f"| {n} | {100*old['macro']:.2f}% | {100*fixed['macro']:.2f}% | {old['unsupported']}/{old['hidden_n']} |")
    for pair, values in diag["comparisons"].items():
        v = values["macro"]
        lines += ["", f"{pair}: **{100*v['difference']:+.2f} percentage points**, 95% interval [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]."]
    lines += ["", "Intervals use 3,000 draws of SQuAD article clusters and HotpotQA/RGB question clusters, seed 47. They do not establish a clear DeepSeek–candF1 winner. These corrections do not authorize a new selection claim on an opened holdout.", "",
              "## Corrected evidence scoring and labels", "",
              "Scorer v3 preserves frozen v1/v2 outputs for historical reproduction. Citation validity, displayed annotated support and eligibility are separate fields. Each required fact has interchangeable evidence passages; a complete sufficient set must be cited. Same-title text qualifies only when it contains the annotated facts. Exact accepted answers are checked before the precision stoplist, preserving yes/no answers.", "",
              "Eligibility is fixed from the permitted corpus and evidence mask before model scoring. An incidental gold answer string such as ‘Anglican Diocese of Sydney’ cannot remove a case. Open episodes whose annotated evidence is truncated are excluded with a recorded reason. Hidden-case answer flags remain annotation-based; an unannotated alternative claim is reported as support not established, rather than a semantic hallucination verdict.", "",
              f"The full corrected hosted comparison has **{revised['coverage']['common']} shared cases**, with {len(revised['coverage']['excluded'])} fixed evidence exclusions after {revised['coverage']['unavailable']} unavailable cases. See [DeepSeek](deepseek_comparison_corrected.md), [GPT-6 Luna](azure_round5_comparison_corrected.md), [local round 5](round5_corrected.md) and [older suites](external_benchmarks_corrected.md). Their denominator differs from the 863-case diagnostic.", "",
              "Official SQuAD answer offsets now select the annotated sentence, and the supporting sentence must survive the display cutoff. Existing sealed files stay unchanged; the replay overlays source corrections in memory. Future natural-unanswerable SQuAD jobs restrict retrieval to the original paragraph. The older corpus-wide French and Indian War question is corrected to answerable with 1754, using the displayed May-1754 evidence in `c00917`; this adjudication follows the user-supplied audit and local source excerpt. Other unannotated entailments remain outside this automatic metric.", "",
              "RGB passage identity now includes the full-text hash, preserving distinct texts with the same first nine words. Zero-noise generation requires five passages and skips undersupplied variants; historical round-5 zero-noise prompts had fewer for 31/100 questions, and 57 distinct source texts had been lost to prefix collisions. MuSiQue duplicate question/answer pairs are removed in future loading and in the corrected replay. These generator corrections need a new evaluation dataset to measure their model effect.", "",
              "Nimbus scoring now requires both current documents for two-document questions and rejects superseded citations. Main's old single two-document success is removed; see [Nimbus replay](nimbus_corrected.md).", "",
              "## Oracle and AuditKIT checks", "",
              "The README's old oracle 1.0 referred to the weaker success metric. Strict scorer-v2 success was 372/471 on RGB and 78/82 on HotpotQA.", "",
              "| Oracle suite | Historical strict | Corrected strict / eligible |", "| --- | ---: | ---: |"]
    for part, s in report["oracle"].items():
        lines.append(f"| {part} | {s['historical_strict']}/{s['episodes']} | {s['corrected_strict']}/{s['corrected_eligible']} |")
    lines += ["", "Oracle checks apply to eligible displayed annotations. Recorded cutoff/annotation exclusions and automatic citation matching cannot certify semantic solvability of every constructed episode. The sealed test is not rewritten to force an oracle ceiling.", "",
              "AuditKIT now receives minimal independently sufficient evidence sets and every referenced passage in its snapshot. The old main RGB recall 0.960 was based on only 25/286 answerable episodes, with 261 unknowns. Corrected coverage is:", ""]
    for part, s in report["auditkit_rgb_recall"].items():
        lines.append(f"- {part}: mean recall {s['mean_on_scored']:.4f}; {s['statuses']} across {s['answerable']} answerable cases.")
    lines += ["", "## Preference training: quarantine and prevention", "",
              "The original round-5 pair file exposed annotated support for 1 held-out HotpotQA dev and 3 test questions; round 5b exposed 2 dev and 5 test questions. The audit found zero exact held-out question matches. Round 5b also contained three identical prompts with competing chosen search/abstain targets. These are later DPO experiments; main and the earlier round-4 candF1 checkpoint precede them.", "",
              "The HotpotQA sampler now indexes only retained training paragraphs and excludes held-out support by title and normalized text fingerprints, including copies under other titles. Pair assembly validates actual user-visible prompts, removes every member of conflicting-target groups and deduplicates identical prompts. The trainer validates the entire pair file before loading weights or applying a row limit. Original artifacts remain historical; separate clean files pass the same guard:", ""]
    for name, s in report["training_cleanup"].items():
        reasons = Counter(r["reason"] for r in s["removed"])
        lines.append(f"- `{name}_clean.jsonl`: {s['clean_rows']}/{s['input_rows']} retained; removals {dict(reasons)}. SHA256 `{s['clean_sha256']}`.")
    lines += ["", "The guard checks exact normalized question and support overlap, using at least six words for support fingerprints; it does not detect paraphrases or certify zero semantic exposure. Cleaning does not undo exposure in already trained DPO checkpoints. They remain unselected. The original safety rule and main submission are retained; a new model requires clean training and a fresh registered holdout.", "",
              f"In the corrected local replay, candF1's upper 95% bound on the difference in flagged-answer rate is **+{100*safety_upper:.2f} percentage points**, above the registered **+2.00-point** safety limit. The old diagnostic bound was +2.97 points. Neither replay qualifies candF1 for promotion under that rule.", "",
              "## Native-tool fallback and release documentation", "",
              "The native prompt now states the three-search budget and the required final action after it. Original saved outputs are not changed. On the historical 305-case hidden subset, DeepSeek had no model abstentions and fallback converted 269 invalid endings to refusals. Across the 863 shared cases, 316/320 invalid endings exceeded the budget. Its reported score describes the full fallback pipeline.", "",
              "The article now calls the 1,611 training rows assistant decisions from 488 selected parent questions, drawn from a 600-question sample. Historical Markdown tables are labeled and linked to corrected replays. The local episodes dataset card and card generator now list both CC BY-SA 4.0 and RGB's CC BY-NC-SA 4.0; 1,096 released verifier rows contain RGB-derived evidence. Source licenses and attribution apply per file. Existing remote uploads were not inspected or republished by this remediation.", "",
              "## Verification and reproduction", "",
              f"All **{sum(report['sealed_hashes'].values())}/{len(report['sealed_hashes'])} sealed hashes match**. Original v1 outcomes and strict evidence flags are checked against saved score files for every replayed transcript. Regression tests cover binary answers, RGB alternatives and the saved rgb-177 oracle, every compound component, the Sydney denominator, displayed truncation, SQuAD offsets/scope, training leakage/conflicts and current policy citations.", "",
              "```powershell", "python -m pytest tests -q", "python scripts/remediate_audit.py", "```", "",
              "Use the Python 3.12 workspace environment and workspace cache variables from `scripts/env.ps1`; local Hugging Face cache files make the replay offline. JSON files alongside the reports contain statistics, confidence intervals, fixed exclusions and cleanup hashes. GPU compute is not needed for these corrections; the authorized second Kaggle account remains available for any later, separately registered experiment."]
    return "\n".join(lines) + "\n"


def main():
    sources = squad_sources()
    cases, verdicts = D.cases_and_verdicts()
    specs = {"base": (A.R5REF, "base"), "main": (A.R5REF, "main"), "candA": (A.R5REF, "candA"),
             "candF1": (A.R5ALT, "candF1"), **{n: (p, "azure") for n, p in D.REMOTE.items()}}
    versions = [{}, {}, {}]
    for name, (path, label) in specs.items():
        rows = [load(path / f"{label}_bench_{part}/lenient", part, sources) for part in A.PARTS]
        for i in range(3):
            versions[i][name] = sum((r[i] for r in rows), [])
        print("loaded", name, flush=True)
    wrapped = [wrappers(rr, cases, verdicts) for rr in versions]
    historic = {n: rr for n, rr in wrapped[0].items() if not n.startswith("base")}
    indexed, keys, coverage = A.common_data(historic)
    diagnosed = {n: A.by_key(wrapped[1][n]) for n in indexed}
    pairs = [("azure_deepseek_native+FB", "candF1+FB+V"), ("azure_deepseek_native+FB", "azure_native+FB")]
    diag = {"episodes": len(keys), "historical_coverage": coverage,
            "published": {n: A.r5_stats(rr, keys) for n, rr in indexed.items()},
            "corrected_two_bugs": {n: A.r5_stats(rr, keys) for n, rr in diagnosed.items()},
            "comparisons": clustered({n: diagnosed[n] for n in ("candF1+FB+V", "azure_native+FB", "azure_deepseek_native+FB")}, keys, pairs)}
    write("round5_two_bug_replay", diag)
    fixed_ix, fixed_keys, fixed_cov = fixed_common({n: rr for n, rr in wrapped[2].items() if not n.startswith("base")})
    revised = table_report("deepseek_comparison_corrected", "DeepSeek comparison: corrected evidence scorer", fixed_ix, fixed_keys, fixed_cov, pairs)
    azure = {n: rr for n, rr in wrapped[2].items() if not n.startswith("base") and not n.startswith("azure_deepseek")}
    ix, kk, cov = fixed_common(azure)
    table_report("azure_round5_comparison_corrected", "GPT-6 Luna comparison: corrected evidence scorer", ix, kk, cov,
                 [("azure_native+FB", "main+FB+V"), ("azure_native+FB", "candF1+FB+V")])
    local = {n: rr for n, rr in wrapped[2].items() if not n.startswith("azure")}
    ix, kk, cov = fixed_common(local)
    local_result = table_report("round5_corrected", "Round 5: corrected local replay", ix, kk, cov,
                               [("candF1+FB+V", "main+FB+V")])
    seal = json.loads((OUT / "round5_test_manifest.json").read_text(encoding="utf-8"))
    hashes = {f"{part}/{name}": sha(ROOT / "data/bench" / part / name) == value
              for part, info in seal["sets"].items() for name, value in info["file_sha256"].items()}
    if not all(hashes.values()):
        raise ValueError("a sealed input changed")
    guard = lineage.PromptGuard(lineage.load_holdouts(ROOT))
    clean_report = {}
    for name in ("dpo_r5", "dpo_r5b"):
        path = ROOT / "data/processed/training" / (name + ".jsonl")
        clean, info = lineage.clean_rows(A.RS.read(path), guard)
        dest = path.with_name(name + "_clean.jsonl")
        dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean), encoding="utf-8")
        clean_report[name] = {**info, "source_sha256": sha(path), "clean_sha256": sha(dest), "clean_file": str(dest.relative_to(ROOT))}
    oracle = {}
    for part in A.PARTS:
        original, diagnosed_oracle, corrected_oracle = load(ROOT / "runs/oracle_r5" / part / "strict", part, sources)
        oracle[part] = {"episodes": len(original), "historical_strict": sum(r["strong_v2"] for r in original),
                        "corrected_strict": sum(r["strong_v2"] for r in corrected_oracle if not r["excluded_v2"]),
                        "corrected_eligible": sum(not r["excluded_v2"] for r in corrected_oracle)}
    recall = auditkit_recall()
    write("auditkit_evidence_recall_corrected", recall)
    external = external_report(sources)
    nimbus = nimbus_report()
    ch = A.RS.world("bench:squad2")[0].chunk("c00917")
    write("evidence_adjudications_2026-11-13", [{"suite": "squad2", "qid": "5ad27473d7d075001a4294e8",
        "scope": "legacy corpus-wide retrieval", "answerable": True, "answer": "1754", "chunk_id": ch["chunk_id"],
        "title": ch["title"], "support_sentence": benchmarks.sentences(ch["text"])[2],
        "chunk_text_sha256": hashlib.sha256(ch["text"].encode("utf-8")).hexdigest(),
        "basis": "user-supplied audit finding and local source excerpt; applied identically to every system"}])
    report = {"date": DATE, "submission": "main+FB+V", "sealed_hashes": hashes, "oracle": oracle,
              "auditkit_rgb_recall": recall, "external": external, "nimbus": nimbus,
              "two_bug_replay": diag, "corrected_comparison": revised, "local_replay": local_result, "training_cleanup": clean_report}
    write(DATE + "-evaluation-remediation", report)
    markdown = dated_markdown(report)
    tests_path = ROOT / ".cache/audit-remediation-tests.xml"
    if tests_path.exists():
        suite = ET.parse(tests_path).getroot().find("testsuite")
        if suite is not None:
            checked = {k: int(suite.get(k, "0")) for k in ("tests", "failures", "errors", "skipped")}
            report["last_recorded_tests"] = checked
            markdown = markdown.replace("## Verification and reproduction\n", "## Verification and reproduction\n\n"
                f"Latest recorded CPU test run: {checked['tests']} tests, {checked['failures']} failures, {checked['errors']} errors, {checked['skipped']} skipped.\n")
            write(DATE + "-evaluation-remediation", report)
    (OUT / (DATE + "-evaluation-remediation.md")).write_text(markdown, encoding="utf-8")
    print("sealed hashes", sum(hashes.values()), "/", len(hashes), "oracle", oracle, flush=True)
    print("clean rows", {n: r["clean_rows"] for n, r in clean_report.items()}, flush=True)
    print("two-bug replay", {n: round(diag['corrected_two_bugs'][n]['macro'], 6) for n in ('main+FB+V','candF1+FB+V','azure_native+FB','azure_deepseek_native+FB')}, flush=True)


if __name__ == "__main__":
    main()
