# Judge-based RAG metrics, numeric accuracy and single-request latency: pre-registration

Written 16 November 2026, before any judge call or latency run. Nothing here changes the submitted English system (main adapter, fail-closed rule, self-check). It fills the AuditKIT metrics our earlier runs did not compute (README, "Hackathon submission notes"). All results are descriptive. They are not used to choose or retune a system.

## 1. What is measured

| AuditKIT metric | How | Where |
| --- | --- | --- |
| faithfulness, hallucination, response groundedness, answer relevancy | AuditKIT judge classes, unmodified prompts (prompt version v2) | 300 sampled episodes of the first sealed test |
| context relevance, context precision, context recall | same | same |
| numeric accuracy | deterministic, defined in section 5 | all eligible numeric-answer episodes of the first sealed test |
| latency (p50, p95, p99) and throughput | AuditKIT `LatencyStats` and `Throughput`, batch size 1 | Kaggle T4, section 6 |

## 2. Sample

- Test: the first sealed test (r5t), scorer v3. Eligible episodes: the 934 that are eligible for base, main and candF1 alike.
- Draw: 300 episodes, seed 61, proportional to the eight strata (question group x answerable / hidden evidence), largest-remainder rounding, by `scripts/build_judge_cases.py`. The draw uses labels only. It was made before any judge call.
- Sample hash (SHA-256 of the episode list): `91aa3283ead72dbfe120ef947714cb0aa531bb59ff77cb8161ebad650216a984`. Strata: single 42 + 28, compound 29 + 23, HotpotQA 14 + 12, RGB 94 + 58 (answerable + hidden).
- Systems judged: the raw answers of base, main and candF1, as saved. `main+FB+V` (submitted) and `candF1+FB+V` are not judged separately. Their answers are the raw answers that the fail-closed rule and the saved self-check verdicts let through, so each case carries a flag for that. A raw answer that is not a valid answer turn is not judged as an answer (counted as "no valid answer").
- Passages: what the system was shown (the 800-character excerpts, in order of first sight, plus any evidence supplied up front).

## 3. Judge

- Azure deployment `gpt-6-luna` (`gpt-6-luna-2026-09-22`), the model our hosted comparison used as a competitor. It does not judge any of its own answers here. It is a different family from the Tiny Aya systems.
- Default sampling of the deployment (it fixes temperature). Reasoning effort is the deployment default (measured at about 30 reasoning tokens per call on test prompts). Maximum 3,000 completion tokens per call.
- Every call is cached by the hash of its prompt and its token usage is logged, so the cost is metered. Prices: $0.10 per million input tokens, $0.50 per million output tokens (as in `reports/azure_roi.md`; cached input is billed at $0.01 but metered at the full price here).
- Budget: expected about $1. The run stops when the metered cost reaches $2.50. Azure spend was approved by the user as an estimated "under $1"; the cap is the stated limit for the whole run.
- An unreadable judge reply or a provider block leaves the metric empty for that episode and is counted. It is never scored as 0 or 1. A network or API failure that survives the retries stops the run, which resumes from the cache.
- The judge sees the question, the answer text only (not the citation tags), and the passages. Hidden-evidence episodes have no reference answer, because nothing the system could see contains it. They get faithfulness, hallucination, groundedness, answer relevancy and context relevance, but not context precision or recall.
- Only the first 12 episodes run as a pilot to check the plumbing and the token meter. The pilot is part of the final run. No prompt, setting or sample changes after the pilot.

## 4. Aggregation and reading rules

- Per system and metric: the mean over episodes where the metric is defined. The count is always reported. Faithfulness, hallucination, groundedness and answer relevancy exist only for answered episodes with at least one claim; a refusal has no claims. Coverage (answers given out of 300) is reported next to them, because a system that answers less can look more faithful.
- Subsets: all, answerable, hidden evidence.
- Intervals: 95% bootstrap over episodes of the sample (3,000 draws, seed 47), for the means and for differences against base.
- Pre-stated readings, all descriptive:
  1. The annotation-based flag "answers when evidence was hidden" is corroborated by the judge if the mean judge hallucination of main's answers on hidden-evidence episodes is above 0.5. If it is not, the README says the judge does not corroborate the flag.
  2. The self-check improves faithfulness if the mean faithfulness of `main+FB+V` answers is above that of main's raw answers, with the difference interval above 0 (the comparison is within main's answers, so the sets are nested).
  3. Whatever the direction, the numbers go into the README and the write-up. If the judge contradicts a claim we made, the claim is changed in the same commit.
- Judge limits stated with the results: one judge, one sample of 300, claim extraction by the judge is not deterministic, and the hidden-evidence flags are still not human-checked. A 50-case blank sheet for hand labels (25 hidden-evidence and 25 answerable answers of main, seed 61, no judge verdicts shown) is prepared for the team. If labels come back before submission, the agreement with the judge is reported. Otherwise the README keeps saying the flags are not hand-checked.

## 5. Numeric accuracy

AuditKIT has no numeric metric and our datasets carry no numeric type. We define a numeric-answer episode as an answerable episode of the first sealed test (eligible, scorer v3) whose gold answer is a single number, optionally with a currency sign, thousands separators or a percent sign (regular expression in `scripts/report_numeric.py`). That is 17 questions before variants, so the result is small and is reported with its count.

- Numeric accuracy: of the numeric-answer episodes, the share where the system answered and the numbers in its answer include the gold value (commas removed, compared as numbers).
- Also reported: the share answered, and the share correct among answered.
- Systems: base, main, `main+FB+V`, `candF1+FB+V`, as in the headline table.

## 6. Latency and throughput

- Hardware and software: one Kaggle T4 (16 GB), the model in 4-bit as in all runs, Hugging Face `generate`, greedy decoding, at most 160 new tokens per turn, batch size 1, retrieval from the SQLite index on local disk. No quantisation, compilation or serving-engine tuning beyond what the earlier runs used. This is a floor for a research stack, not a tuned deployment.
- Episodes: 50 from each of the three r5t suites (150 in total), drawn with seed 61 from the jobs `scripts/run_eval.py` builds (masks and pre-supplied evidence unchanged). The same 150 requests run in both arms, in the same order. Three untimed warm-up requests precede each arm.
- Arms, in one process with one loaded model: `base` (adapter switched off, same prompt and search loop, fail-closed rule, no self-check) and `submitted` (main adapter, fail-closed rule, self-check).
- Each request is timed from the call to `RagSystem.ask` until the decision returns, with the time split into generation, search and self-check.
- Reported: p50, p95, p99, mean and max latency; requests per second at batch size 1 (AuditKIT `Throughput`); mean output tokens; peak GPU memory. Per suite as well as pooled. Also the cost per answer on a T4 at the rate in `reports/azure_roi.md` ($0.526 per hour), if the GPU is busy.
- Two requests that overlap other work on the same machine cannot be ruled out on a shared Kaggle host. We report what we measured and state this.

## 7. What this does not do

- It does not run a second judge, a second training seed, or the hosted models on a second sealed test or on Hindi.
- Freshness stays measured only on the Nimbus product suite. The SQuAD, HotpotQA and RGB suites carry no version labels.
- No new training, no change to a published adapter, no change to the pre-registered selection of the submitted system.

## 8. Files

`scripts/build_judge_cases.py`, `scripts/run_judge.py`, `scripts/report_judge.py`, `scripts/report_numeric.py`, `scripts/latency_run.py`, `scripts/report_latency.py`; `data/processed/judge/` (sample, cases, cached judge calls); results in `reports/judge_results.{json,md}`, `reports/numeric_accuracy.{json,md}` and `reports/latency.{json,md}`. Deviations from this document are labelled as deviations where they occur.
