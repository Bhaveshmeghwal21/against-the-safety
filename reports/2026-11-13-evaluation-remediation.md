# Evaluation and training-data remediation — 13 November 2026

**Main with fail-closed fallback and the saved verifier remains the submitted system.** The evaluation bugs are corrected, later preference data is quarantined and cleaned, and affected documentation is updated. This work uses saved transcripts and local CPU computation. No checkpoint was retrained or promoted, and no GPU or hosted-model call was made.

## The same 863 cases: isolate the two scoring bugs

This diagnostic preserves the original comparison intersection, prompts, masks, exclusions, fallback and verifier decisions. It restores exact accepted answers and independently sufficient, displayed RGB alternatives. It does not replace the eligibility or source annotations in the historical test.

| System | Published macro | Two-bug replay | Original flagged / hidden |
| --- | ---: | ---: | ---: |
| main+FB+V | 50.73% | 57.30% | 22/305 |
| candF1+FB+V | 55.54% | 62.81% | 25/305 |
| azure_native+FB | 54.90% | 60.04% | 59/305 |
| azure_deepseek_native+FB | 61.10% | 64.71% | 36/305 |

azure_deepseek_native+FB/candF1+FB+V: **+1.89 percentage points**, 95% interval [-2.70, +6.42].

azure_deepseek_native+FB/azure_native+FB: **+4.67 percentage points**, 95% interval [+1.38, +8.17].

Intervals use 3,000 draws of SQuAD article clusters and HotpotQA/RGB question clusters, seed 47. They do not establish a clear DeepSeek–candF1 winner. These corrections do not authorize a new selection claim on an opened holdout.

## Corrected evidence scoring and labels

Scorer v3 preserves frozen v1/v2 outputs for historical reproduction. Citation validity, displayed annotated support and eligibility are separate fields. Each required fact has interchangeable evidence passages; a complete sufficient set must be cited. Same-title text qualifies only when it contains the annotated facts. Exact accepted answers are checked before the precision stoplist, preserving yes/no answers.

Eligibility is fixed from the permitted corpus and evidence mask before model scoring. An incidental gold answer string such as ‘Anglican Diocese of Sydney’ cannot remove a case. Open episodes whose annotated evidence is truncated are excluded with a recorded reason. Hidden-case answer flags remain annotation-based; an unannotated alternative claim is reported as support not established, rather than a semantic hallucination verdict.

The full corrected hosted comparison has **925 shared cases**, with 4 fixed evidence exclusions after 9 unavailable cases. See [DeepSeek](deepseek_comparison_corrected.md), [GPT-6 Luna](azure_round5_comparison_corrected.md), [local round 5](round5_corrected.md) and [older suites](external_benchmarks_corrected.md). Their denominator differs from the 863-case diagnostic.

Official SQuAD answer offsets now select the annotated sentence, and the supporting sentence must survive the display cutoff. Existing sealed files stay unchanged; the replay overlays source corrections in memory. Future natural-unanswerable SQuAD jobs restrict retrieval to the original paragraph. The older corpus-wide French and Indian War question is corrected to answerable with 1754, using the displayed May-1754 evidence in `c00917`; this adjudication follows the user-supplied audit and local source excerpt. Other unannotated entailments remain outside this automatic metric.

RGB passage identity now includes the full-text hash, preserving distinct texts with the same first nine words. Zero-noise generation requires five passages and skips undersupplied variants; historical round-5 zero-noise prompts had fewer for 31/100 questions, and 57 distinct source texts had been lost to prefix collisions. MuSiQue duplicate question/answer pairs are removed in future loading and in the corrected replay. These generator corrections need a new evaluation dataset to measure their model effect.

Nimbus scoring now requires both current documents for two-document questions and rejects superseded citations. Main's old single two-document success is removed; see [Nimbus replay](nimbus_corrected.md).

## Oracle and AuditKIT checks

The README's old oracle 1.0 referred to the weaker success metric. Strict scorer-v2 success was 372/471 on RGB and 78/82 on HotpotQA.

| Oracle suite | Historical strict | Corrected strict / eligible |
| --- | ---: | ---: |
| r5t_squad | 385/385 | 381/381 |
| r5t_hotpot | 78/82 | 82/82 |
| r5t_rgb | 372/471 | 471/471 |

Oracle checks apply to eligible displayed annotations. Recorded cutoff/annotation exclusions and automatic citation matching cannot certify semantic solvability of every constructed episode. The sealed test is not rewritten to force an oracle ceiling.

AuditKIT now receives minimal independently sufficient evidence sets and every referenced passage in its snapshot. The old main RGB recall 0.960 was based on only 25/286 answerable episodes, with 261 unknowns. Corrected coverage is:

- rgb: mean recall 1.0000; {'scored': 286} across 286 answerable cases.
- r5t_rgb: mean recall 0.9966; {'scored': 292} across 292 answerable cases.

## Preference training: quarantine and prevention

The original round-5 pair file exposed annotated support for 1 held-out HotpotQA dev and 3 test questions; round 5b exposed 2 dev and 5 test questions. The audit found zero exact held-out question matches. Round 5b also contained three identical prompts with competing chosen search/abstain targets. These are later DPO experiments; main and the earlier round-4 candF1 checkpoint precede them.

The HotpotQA sampler now indexes only retained training paragraphs and excludes held-out support by title and normalized text fingerprints, including copies under other titles. Pair assembly validates actual user-visible prompts, removes every member of conflicting-target groups and deduplicates identical prompts. The trainer validates the entire pair file before loading weights or applying a row limit. Original artifacts remain historical; separate clean files pass the same guard:

- `dpo_r5_clean.jsonl`: 276/335 retained; removals {'heldout_evidence_or_question': 9, 'duplicate_prompt': 50}. SHA256 `c603c183c15f4cc401448336ade8a7884e8262e905ec4c944c896f739d17bf68`.
- `dpo_r5b_clean.jsonl`: 425/498 retained; removals {'heldout_evidence_or_question': 16, 'conflicting_target': 6, 'duplicate_prompt': 51}. SHA256 `541dbba3befec4fe31f44e3ec76b416f61ac1ebffc244db04d05ddd5ea9cfca0`.

The guard checks exact normalized question and support overlap, using at least six words for support fingerprints; it does not detect paraphrases or certify zero semantic exposure. Cleaning does not undo exposure in already trained DPO checkpoints. They remain unselected. The original safety rule and main submission are retained; a new model requires clean training and a fresh registered holdout.

In the corrected local replay, candF1's upper 95% bound on the difference in flagged-answer rate is **+3.35 percentage points**, above the registered **+2.00-point** safety limit. The old diagnostic bound was +2.97 points. Neither replay qualifies candF1 for promotion under that rule.

## Native-tool fallback and release documentation

The native prompt now states the three-search budget and the required final action after it. Original saved outputs are not changed. On the historical 305-case hidden subset, DeepSeek had no model abstentions and fallback converted 269 invalid endings to refusals. Across the 863 shared cases, 316/320 invalid endings exceeded the budget. Its reported score describes the full fallback pipeline.

The article now calls the 1,611 training rows assistant decisions from 488 selected parent questions, drawn from a 600-question sample. Historical Markdown tables are labeled and linked to corrected replays. The local episodes dataset card and card generator now list both CC BY-SA 4.0 and RGB's CC BY-NC-SA 4.0; 1,096 released verifier rows contain RGB-derived evidence. Source licenses and attribution apply per file. Existing remote uploads were not inspected or republished by this remediation.

## Verification and reproduction

Latest recorded CPU test run: 93 tests, 0 failures, 0 errors, 0 skipped.

All **11/11 sealed hashes match**. Original v1 outcomes and strict evidence flags are checked against saved score files for every replayed transcript. Regression tests cover binary answers, RGB alternatives and the saved rgb-177 oracle, every compound component, the Sydney denominator, displayed truncation, SQuAD offsets/scope, training leakage/conflicts and current policy citations.

```powershell
python -m pytest tests -q
python scripts/remediate_audit.py
```

Use the Python 3.12 workspace environment and workspace cache variables from `scripts/env.ps1`; local Hugging Face cache files make the replay offline. JSON files alongside the reports contain statistics, confidence intervals, fixed exclusions and cleanup hashes. GPU compute is not needed for these corrections; the authorized second Kaggle account remains available for any later, separately registered experiment.
