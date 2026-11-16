# The two Lexsi tools in this project

## How we used AgentTune

| Step | AgentTune part | What we did with it |
| --- | --- | --- |
| Search | `SQLiteFTSBackend`, `SearchCorpusTool` | We built a keyword index of the passages. The `search_corpus` tool the model calls is a subclass of AgentTune's. We made three changes: it shows 800 characters of a passage instead of 350, it hides relevance scores, and it can hide chosen passages for evidence-hidden episodes. AgentTune's backend reports a broken index as "no results", so we added a check that fails loudly when the index is empty or wrong. |
| Recording | `EventLog`, `TrajectoryStore` | Every scripted training episode is saved as an EventLog. Every evaluation run is saved as a TrajectoryStore record. Both are published in the [episodes dataset](https://huggingface.co/datasets/BMEscanor/against-the-safety-rag-episodes). |
| Training | `create_distill_trainer` | It takes the EventLogs and hands the training rows to a "trainer factory". Ours trains a 4-bit LoRA adapter on one T4 and learns only from the model's own turns. AgentTune's default factory loads the model in bf16 (a T4 has no bf16) and trains on every token. |
| Answer matching | `qa_metrics` | Word-overlap (F1) between an answer and the reference answer. |
| What did not fit | DPO trainer | AgentTune's preference trainer ran out of memory on a T4, so preference training used our own short loop with the same loss. |

Code: `src/ragharness/trainer.py`, `qlora.py`, `retrieval.py`, `tools.py`, `episodes.py`.

## How we used AuditKIT

For every evaluation episode we build two small records and hand them to AuditKIT. An `ak.Sample` carries a `RAGTrace`: the searches, the passages retrieved, the citations and the final answer or refusal. A `RAGCase` holds what the right behaviour was: whether the question was answerable, which passages are enough, the reference answer. `ak.evaluate` then returns the scores below.

First-round scores on the HotpotQA test split, scored before the 13 November corrections:

| AuditKIT metric | The question it answers | Base | Main | Control |
| --- | --- | ---: | ---: | ---: |
| Evidence-set recall | Did the searches retrieve at least one complete set of passages that is enough to answer? | 0.512 | 0.727 | 0.824 |
| Citation support | Do the citations point to passages that exist, are current and support the answer? | 0.302 | 0.771 | 0.941 |
| Abstention | Did it refuse when it should and answer when it should? | 0.721 | 0.794 | 0.714 |
| Tool-call validity | Of the turns that were JSON tool calls, did they match the tool's schema? | 1.000 | 1.000 | 1.000 |
| Redundant tool calls (lower is better) | What share of searches exactly repeat an earlier one? | 3.1% | 2.9% | 8.8% |
| Loop-free episodes | What share of episodes never fell into a repetition loop? | 94.1% | 93.4% | 78.4% |

*Control* is a second adapter trained on the same questions but with no refusal examples. It shows what the refusal examples change.

What AuditKIT showed us:

- **Training taught citing.** Citation support rose from 0.302 to 0.771.
- **Refusal examples teach the model when to stop.** The control adapter looped: only 78.4% of its episodes were loop-free and 8.8% of its searches repeated an earlier one. Main had 93.4% and 2.9%.
- **It reports "unknown" instead of guessing.** When our reference evidence was incomplete, AuditKIT returned no score for those episodes. That exposed an earlier RGB recall figure of 0.960 that covered only 25 of 286 answerable episodes. With complete evidence sets all 286 were scored and recall was 1.000 ([remediation report](../reports/2026-11-13-evaluation-remediation.md)).
- **Freshness on dated manuals.** On our fictional Nimbus manuals, which hold old and new editions of the same policy, AuditKIT's freshness score was 0.813 for main and 0.783 for the base model. About one retrieved dated passage in five was an out-of-date edition ([report](../reports/nimbus_corrected.md)).
- **Judge-based metrics, later.** Faithfulness, hallucination, context precision and recall, answer relevancy, numeric accuracy and latency came last, on a 300-episode sample and a Kaggle T4. The submitted system's answers are more faithful than the base model's ([results](AUDITKIT_METRICS.md#more-auditkit-metrics-faithfulness-retrieval-quality-numbers-and-speed)).

AuditKIT also ships general benchmarks (ARC, GSM8K, MMLU, TruthfulQA, BFCL). None of them tests answering from supplied documents with citations, so our headline numbers do not use them. We did run ARC, MMLU and TruthfulQA, with our own scorer rather than AuditKIT's lm-eval engine, as a retention check (see [RESULTS.md](RESULTS.md#did-the-fine-tuning-hurt-general-ability)); GSM8K and BFCL were not run. The headline numbers in [the README](../README.md#results) come from our own scorer, which checks in one pass that an answer is correct and that the cited passages contain the required evidence. The full first-round scorecard is in [`reports/scorecard_test_lenient.md`](../reports/scorecard_test_lenient.md). Code: `src/ragharness/audit.py`, `scorecard.py`.
