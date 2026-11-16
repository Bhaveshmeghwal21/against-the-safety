# AuditKIT metrics

This page has the judge-based metrics, retrieval quality, numeric accuracy, speed and tool-call checks. The first-round scorecard, which uses checks that need no judge, is in [TOOLS.md](TOOLS.md).

## Tool use on the sealed test

AuditKIT's tool-call checks, as each evaluation run saved them, for the raw agents before the fail-closed rule, over all 938 episodes of the first sealed test ([`reports/tool_use_r5t.md`](../reports/tool_use_r5t.md)).

| System | Suite | Well-formed tool calls | Repeated searches (lower is better) | Loop-free episodes | Searches per episode | Invalid endings (lower is better) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tiny Aya base | SQuAD | 100.0% | 0.0% | 100.0% | 0.98 | 28.3% |
| Tiny Aya base | HotpotQA | 100.0% | 0.0% | 100.0% | 0.98 | 9.8% |
| Tiny Aya base | RGB | 100.0% | 0.5% | 98.9% | 0.83 | 16.3% |
| Main adapter | SQuAD | 100.0% | 5.7% | 87.5% | 1.62 | 12.2% |
| Main adapter | HotpotQA | 100.0% | 2.0% | 93.9% | 2.18 | 7.3% |
| Main adapter | RGB | 100.0% | 3.1% | 94.1% | 1.11 | 5.7% |
| candF1 | SQuAD | 100.0% | 7.6% | 84.2% | 1.45 | 15.6% |
| candF1 | HotpotQA | 100.0% | 3.0% | 90.2% | 2.09 | 7.3% |
| candF1 | RGB | 100.0% | 2.2% | 96.2% | 1.06 | 4.0% |

Every search call was well formed for all three systems. The trained adapters search more (1.1 to 2.2 searches per episode, against about one for the base model) and repeat a query more often: 2% to 8% of their searches repeat an earlier one, and 84% to 96% of their episodes are loop-free. The fail-closed rule turns such an episode into a refusal. The base model looks loop-free because it rarely searches twice. It ends 10% to 28% of episodes with an invalid turn instead, against 4% to 16% for the adapters. Latency per suite is in [`reports/latency.md`](../reports/latency.md).

## More AuditKIT metrics: faithfulness, retrieval quality, numbers and speed

Our scorecards so far use the AuditKIT checks that need no judge model. On 16 November we added the metrics that do need one, plus numeric accuracy and single-request speed, so that the AuditKIT list for RAG is covered. The plan was written before any of it ran ([pre-registration](../research/2026-11-16-judge-latency-preregistration.md)). Nothing here changed the submitted system.

**How.** AuditKIT's own judge classes (`faithfulness`, `hallucination`, `response_groundedness`, `answer_relevancy`, `context_relevance`, `context_precision`, `context_recall`; prompts unmodified) ran on 300 episodes of the first sealed test. The sample was drawn with a fixed seed, in proportion to the question groups and to answerable and hidden-evidence episodes. The judge was Azure's `gpt-6-luna`. Our hosted comparison used the same model as a competitor, but it judged none of its own answers here. The whole judge run cost $0.56, metered from the token counts the API returned. When we asked the judge a second time about a third of the sample, it gave the same value for 96% of faithfulness scores (mean absolute difference 0.014). Full tables: [`reports/judge_results.md`](../reports/judge_results.md).

**Faithfulness** is the share of an answer's claims that the passages the system was shown support. Hallucination is its complement. Both are averaged over the answers a system gave, so a system that answers less is not automatically better; the answer count is in the second column.

| System | Answers given (of 300 episodes) | Faithfulness (higher is better) | Hallucination (lower is better) | Response groundedness | Answer relevancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny Aya base | 231 | 0.625 (0.566 to 0.685) | 0.375 (0.315 to 0.435) | 0.641 | 0.912 |
| Main adapter alone | 166 | 0.722 (0.653 to 0.790) | 0.278 (0.210 to 0.347) | 0.982 | 0.976 |
| Main + both repairs (submitted) | 125 | 0.808 (0.736 to 0.872) | 0.192 (0.124 to 0.261) | 0.984 | 1.000 |
| candF1 + both repairs (alternative) | 128 | 0.885 (0.830 to 0.935) | 0.115 (0.066 to 0.171) | 0.977 | 0.989 |

- Faithfulness rises at each step: 0.625 for the base model, 0.722 for the adapter, 0.808 for the submitted system. The self-check adds 0.086 (95% interval 0.043 to 0.136) over the adapter's own answers. Part of that is selection, because the self-check removes answers.
- Almost every sentence the adapter writes is in the passages (groundedness 0.98 against 0.64 for the base model).
- Answer relevancy is close to 1 for every system, because the answers are short phrases. It tells us little here.

**Does the judge agree with our labels?** Our scorer labels each answer from annotations. The judge reads the passages and the answer without seeing those labels.

| Our label of the submitted system's answer | Answers | Mean faithfulness | Share fully faithful | Share with an unsupported or contradicted claim |
| --- | ---: | ---: | ---: | ---: |
| Correct and cited (our scorer) | 85 | 0.965 | 0.965 | 0.035 |
| Wrong answer on answerable question (our scorer) | 28 | 0.607 | 0.571 | 0.429 |
| Answered although the evidence was hidden (our flag) | 12 | 0.167 | 0.167 | 0.833 |

The judge agrees with our scorer on the clear cases: 96.5% of the answers we call correct and cited are fully faithful. Of the 12 answers on hidden evidence, 10 contain a claim the shown passages do not support and 2 do not. So the flag "answered although the evidence was hidden" is mostly, not always, a real hallucination. Across the clear cases of all three raw systems (answers we call correct and cited, or answers on hidden evidence), the judge's fully-faithful verdict agrees with our label on 90.2% of 357 answers (Cohen's kappa 0.80). We added this count after seeing the results. The judge is a model, not a person. The flags are still not hand-checked. A blank 50-row sheet for that, with the labels withheld, is in [`reports/hand_label_sheet.csv`](../reports/hand_label_sheet.csv).

**Retrieval quality.** Every system uses the same search tool and writes its own queries.

| System | Context relevance (share of shown passages that are relevant) | Context precision (relevant passages ranked first) | Context recall (reference answer found in the passages) |
| --- | ---: | ---: | ---: |
| Tiny Aya base | 0.358 (0.326 to 0.392) | 0.696 (0.648 to 0.743) | 0.892 (0.847 to 0.937) |
| Main adapter | 0.356 (0.321 to 0.393) | 0.692 (0.643 to 0.740) | 0.950 (0.916 to 0.978) |
| candF1 | 0.359 (0.322 to 0.395) | 0.700 (0.652 to 0.747) | 0.955 (0.922 to 0.983) |

Relevance is low for all three because the tool returns the top three passages and usually only one or two are relevant. The adapter's queries find the reference answer in the passages a little more often than the base model's (0.950 against 0.892, with overlapping intervals).

**Numeric accuracy.** AuditKIT has no numeric metric and our question sets carry no numeric type. We took the answerable episodes whose gold answer is a single number (16 questions) and counted an answer as correct when the numbers in it include the gold value. The sample is small.

| System | Numeric-answer episodes (questions) | Answered | Numerically correct | Numeric accuracy | Correct among answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 43 (16) | 34 | 24 | 55.8% | 70.6% |
| main | 43 (16) | 30 | 27 | 62.8% | 90.0% |
| main+FB+V (submitted) | 43 (16) | 28 | 26 | 60.5% | 92.9% |
| candF1+FB+V (alternative) | 43 (16) | 29 | 26 | 60.5% | 89.7% |

**Speed.** One Kaggle T4, one request at a time, 150 requests from the three suites of the first sealed test, 3 untimed warm-up requests, greedy decoding, the model in 4-bit, Hugging Face `generate` with no serving-engine tuning. Percentiles and requests per second are AuditKIT's `LatencyStats` and `Throughput`. Full report: [`reports/latency.md`](../reports/latency.md).

| System | Requests | p50 (s) | p95 (s) | p99 (s) | Requests per second | Cost per 1,000 requests, busy T4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny Aya base, same loop, no model self-check | 150 | 8.77 | 19.44 | 23.44 | 0.097 | $1.51 |
| Submitted system (main adapter, fail-closed rule, self-check) | 150 | 16.27 | 23.84 | 26.17 | 0.064 | $2.27 |

The submitted system takes a median of 16.3 s per question and a 95th percentile of 23.8 s. That is 1.9 times the base model's median, because it searches more and uses the model once more to check the answer (the check itself costs 0.28 s per request on average; nearly all the time is generation). Peak GPU memory is 3.98 GB. Our batched evaluation runs needed 4.3 GPU-seconds per episode ([`reports/azure_roi.md`](../reports/azure_roi.md)), because many episodes share a batch. One at a time, the same system needs 15.5 s on average, so the cost per question in that report is a best case, not what a single user would see. A serving engine would be faster; we did not tune one. The base arm applies the fail-closed rule and the code checks of the self-check, so it shows only 7 answers in 150, but it skips the model self-check. The Kaggle host is shared, so some noise is possible.

**Limits of these numbers.** One judge, one sample of 300 episodes, claim extraction by a model, and the hidden-evidence flags are not hand-checked.
