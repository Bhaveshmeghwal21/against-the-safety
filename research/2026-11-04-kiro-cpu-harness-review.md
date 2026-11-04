# Review of Kiro's CPU harness

**Reviewed:** 4 November 2026, IST  
**Team:** Against-the-safety  
**Commit inspected:** `86bde66` — CPU harness, following research commit `74c4555`  
**Scope:** Source, scripts, tests, manifests, stored reports and installed library behavior. No model training or GPU run.

## Assessment

The CPU prototype works and meaningfully connects both required project components: AgentTune supplies the search backend/tool, trajectory schemas and SFT row handoff; AuditKIT consumes episodes and runs evaluation metrics. It is a useful foundation for PS 4. The existing tests pass and the saved development reports reproduce. The next stage still needs a real model runner, complete evidence-state construction and a QLoRA training factory.

Four issues below should be addressed before using this pipeline for the GPU experiments and final reported model results. Application source and committed reports were left unchanged during this review.

## Findings

### 1. P1 — AuditKIT writes outside the promised workspace location

**Location:** `scripts/env.ps1:8–22`, `src/ragharness/audit.py:104–105`.

The environment script redirects `XDG_CACHE_HOME` but not `XDG_DATA_HOME`. AuditKIT 1.1.3's `ExperimentDB` uses `XDG_DATA_HOME`, defaulting to `~/.local/share`. Every call to our `audit.evaluate()` supplies an experiment name, which triggers persistence there.

Following the documented environment setup, an evaluation failed with `PermissionError` trying to save:

```text
C:\Users\ASUS\.local\share\auditkit\experiments\review-correct_but_cites_unobserved_gold.json
```

Setting `XDG_DATA_HOME` to `.cache/review-kiro/data` in the review process allowed evaluation to complete. This was a process-only workaround, not a source fix. Add an explicit workspace-local data home to `env.ps1` and verify where named experiments are saved. The current claim that every output stays inside the workspace is too broad.

### 2. P2 — The supposedly strict parser accepts mixed actions as an abstention

**Location:** `src/ragharness/protocol.py:76–80` and `85–95`.

The abstention regex permits closing/reopening action tags inside its captured text. This response is accepted as `kind='abstain'`:

```text
<abstain>missing</abstain><answer>Bob</answer><citations>c00001</citations><abstain>missing</abstain>
```

On an unanswerable fixture the adapter then reports `abstention=1.0`. That is a false pass for malformed mixed output. Require exactly one complete action and reject nested/reserved action delimiters. Add a regression case that exercises the parser through the scorer.

The search parser also accepts an additional parameter such as `top_k` inside `parameters`, although the documented action schema permits only `query`. It checks the outer keys but not the parameter keys. Reject unexpected parameters or explicitly document their handling.

### 3. P2 — The scripted retrieval score is not a ceiling on answerability

**Location:** `src/ragharness/ceiling.py:3–6`, `research/2026-11-04-fine-tuning-for-rag.md`, CPU harness record, retrieval-ceiling interpretation.

The 0.87 complete-support recall is the result of one restricted gold-assisted query heuristic. It is not a demonstrated upper bound on every policy using visible text. The plan incorrectly concludes that the remaining 13% are not answerable within the budget.

Reproduced counterexample:

```text
qid: 5ae604e65542996de7b71af8
question: Thomas Dimitroff joined the patriots as a scout for the team's
          33rd season in the National Football League which was part of
          what calendar year?
existing heuristic, support paragraphs found after searches: [1, 1, 1]
alternative queries: [original question, "patriots"]
alternative result: full support retrieved within two searches, top_k=3
```

The alternate query is taken directly from the question. Thus the failure is a property of that heuristic, rather than proof of missing or unreachable evidence. Rename the measurement to a scripted oracle retrieval diagnostic, and keep heuristic retrieval failures distinct from genuinely support-absent cases. Do not generate unanswerable gold labels from this failure alone.

### 4. P2 — Report citation validity separately from actual grounding

**Location:** `scripts/audit_slice.py:66–68`, `src/ragharness/audit.py:52–68`, and the metric definitions in the research plan.

In the installed AuditKIT version, `citation_support` measures whether cited spans exist and are current. Gold-support overlap is emitted separately as `citation_precision` and `citation_recall`. The evaluation call returns these additional metrics, but the report loop copies only names from `SCORERS`, dropping them.

Independent fixtures produced:

| Fixture | citation_support | citation_precision | Answer EM | Additional observation |
| --- | ---: | ---: | ---: | --- |
| Correct answer, gold citation never observed | 1.0 | 1.0 | 1.0 | `cited_not_observed=1`, evidence recall 0.0 |
| Wrong answer, citation to an irrelevant existing document | 1.0 | 0.0 | 0.0 | The structural citation score still passes |
| Malformed plain text on an answerable case | 0.0 | 0.0 | Not scored | `abstention=1.0`, while our parser says `invalid` |

The abstention metric is a binary refusal/action diagnostic, not answer correctness. Keep that diagnostic, report invalids separately, retain citation precision/recall, and calculate a separate grounded-answer success condition requiring valid output, a correct answer, actually observed evidence and sufficient support. Use all answerable cases in that denominator, including abstentions and invalids. Correct the plan's descriptions of the metric semantics. Paraphrase entailment still requires the planned audited/judge checks.

## Checks completed

- Ran the existing suite: **24 passed**, none skipped, in 15.45 seconds.
- Verified **600 train, 100 development and 100 test questions**, with no supporting-title overlap between splits.
- Rebuilt the corpus in memory: **7,820 chunks**, matching index hash `169443742963e9a39511fceb13f37cef91540c67593a6c5db30fd90ae3f6cd96`.
- Recomputed the top-k 3, three-search development diagnostic: exact agreement with the stored report, including complete support recall **0.87**.
- Recomputed the two AuditKIT slices under the workspace-local data-home workaround: open-condition scores **0.87** and support-masked abstention **1.0**, matching the saved report.
- Re-ran AgentTune's row handoff and Tiny Aya template checks: **267 rows — 167 search, 87 answer, 13 abstention**.
- The scripted oracle copies gold answers when its support check passes. Its answered-only F1 of 1.0 is correctly labeled as a plumbing check; it is not model performance.
- Scanned development support paragraphs for same-title unmasked versions containing all annotated support sentences: none found in this sample. This does not establish that every alternate evidence path is absent.

## Data and implementation gates still open

Two development questions and two test questions have at least one supporting chunk present among the training questions' context paragraphs, despite the support-title-only split filter. This is consistent with the leakage-report item already listed as open in the CPU harness record. A shared retrieval corpus is legitimate RAG evaluation, but document-level transfer claims require checking the actual SFT observations and excluding exposure to those documents. The current split filter alone does not prove document-disjoint training.

Before a substantial GPU run, finish these parts:

1. Correct the output location, parser and metric/reachability interpretations above.
2. Build the planned evidence conditions and audit their labels and observed training context. The current slice covers only open and support-masked cases.
3. Add a real model-driven episode loop with budgets, invalid/error handling and the same serialization used for training. The current episode runner is scripted.
4. Replace `_NoTrain`/`RowCollector` with the actual QLoRA factory, add a GPU notebook/config, and verify a loss step, save/reload and inference episode.
5. Count tokens with the real accepted Tiny Aya tokenizer, measure memory/throughput on T4, and admit the experiment scope from those measurements.

## Review execution notes

Used the existing CPU virtual environment and cached data. Review-created experiment files were directed to `.cache/review-kiro/data`; no fresh dataset or model download was requested. Existing source, split manifests, index and committed reports were not overwritten. The first evaluation exposed the default AuditKIT output-location failure; subsequent checks used the process-local workaround described above.
