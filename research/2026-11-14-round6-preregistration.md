# Round 6 preregistration

Written on 14 November 2026 before training a Round 6 checker or generating any Round 6 model output. The final-push plan dated 13 November supplied the design and prior observations. The repository remains private at the team's direction.

## Hypotheses

A checker adapter trained on labelled answers will block more unsupported and wrong answers than the existing checker at the same loss of correct answers. A second look at refusals, filtered by the trained checker, will increase correct and cited answers without increasing answers on hidden-evidence episodes.

## Systems and development data

The answering adapters are `main` and `candF1`. The existing checker `V` uses the main adapter. `V2m` continues that adapter with checker training. `V2b` trains a fresh LoRA adapter on the Tiny Aya base. Development selection uses `r4dev` and HotpotQA `dev` only. Preference-trained checkpoints are excluded from the candidate pool.

For the checker gate, pool both actors' answers that pass the deterministic citation and answer-presence checks. For each of `V2m` and `V2b`, choose the smallest threshold in `{0.50, 0.55, ..., 0.95}` at which at most 15% of answers from hidden-evidence episodes pass. Adopt a checker only if it also keeps at least 90% of correct and cited answers at that threshold. Between eligible checkers, select the one that keeps more correct answers. A tie goes to `V2m`. If neither qualifies, Round 6 ends and `V` remains in use.

For the second-look gate, choose the smallest threshold in `{0.60, 0.65, ..., 0.95}` at which at most 5% of forced answers from hidden-evidence episodes pass. Adopt the second look only if it adds at least three correct and cited answers for each answer added on a hidden-evidence episode. Otherwise leave it off.

## Fresh test and decision

Build and hash `r6test` before producing any Round 6 model output. After development selection is committed, run the fresh test once for raw base, main and candF1. Score main with fail-closed and `V` as the reference. Score both actors with fail-closed and the selected `V2`, plus the second look if it passed its gate. Run the hosted models on the same episodes with the planned explicit budget and answer-format prompt. Report scorer-v3 macro success over SQuAD single, SQuAD two-part, HotpotQA and RGB, and an always-refuse baseline.

A new system replaces main with fail-closed and `V` only if the macro difference has a 95% interval above zero and the hidden-evidence answer-rate difference is below zero as a point estimate with an upper interval bound at most +0.02. Use 3,000 clustered bootstrap draws with seed 47, clustering SQuAD by article and HotpotQA and RGB by question. Otherwise report the new system as an alternative.

## Prior observations and disclosures

The round-5 holdout has already been opened. Corrected scores and eligibility, local and hosted comparisons, reference policies and limitations are recorded in `reports/round5_corrected.json`, `reports/azure_round5_comparison_corrected.json`, `reports/deepseek_comparison_corrected.json`, `reports/reference_policies.json`, and `reports/2026-11-13-evaluation-remediation.md`. Hosted runs used prompts without an explicit search limit or the expected two-part answer format. The saved preference-training prompts include held-out HotpotQA evidence. Main and candF1 predate those preference updates.

Before this registration, the 13 November final-push plan also recorded the error breakdown and an exploratory threshold table for `V` on main's opened round-5 answers. Those observations informed this checker project and its development gate. They do not constitute a prospective test of any Round 6 system. The known observations include the answerable-episode outcomes, family-specific answer rates, hosted budget overruns, the refusal baseline, checker retention and false-pass rates at several thresholds, checker AUROCs, and the remaining available source data listed in section 2 of that plan.

The plan reported the following exploratory checker-threshold table for the opened round-5 answers.

| Threshold | Correct answers kept | Hidden-evidence answers let through | Wrong answers on answerable let through |
| ---: | ---: | ---: | ---: |
| 0.5 (in use) | 259/273 (95%) | 42/115 (37%) | 84/138 (61%) |
| 0.7 | 83% | 17% | 34% |
| 0.8 | 71% | 9% | 22% |
| 0.9 | 44% | 2% | 8% |

The checker training examples will come from saved training rollouts, forced training answers, displayed gold support, and matched swaps. Holdout fingerprints will remove cases that match prior or Round 6 evaluation questions and evidence. A question-disjoint validation split will be used. Data hashes and any implementation departures will be recorded as an amendment before the checker is trained.

## Amendment 1: verifier data

Recorded before GPU training. The builder removes held-out matches, deterministic failures, duplicate cases and conflicting labels. Swap negatives are weak labels and may contain remaining noise.

- Train: 820 cases, 373 questions; SHA256 `03e2ad66e97f85c13c287ee96d89751a6c38268fb12d671133a0709ecfb62786`.
- Validation: 82 cases, 41 questions; SHA256 `ccbef53e9c5bde178d2cca72ce1f06184b44774eb6f2bfff599949e99b619d56`.
- Yes/no cases: 4 train and 0 validation; the validation set cannot establish yes/no performance.

## Amendment 2: threshold-grid revision

Recorded before any Round 6 development-model output existed, after inspecting the final held-out training-validation results at threshold 0.50. V2m kept 85% of positive cases and passed 7% of negative cases; V2b kept 88% and passed 10%, respectively. These are internal validation examples, not the development gate; the validation set contains 0 yes/no examples. The grid change is therefore informed by validation and is not a blind preregistration relative to that validation.

Replace only the checker-gate threshold grid with `{0.20, 0.25, ..., 0.95}`, inclusive, in increments of 0.05. For each checker, choose the smallest threshold at which at most 15% of hidden-evidence answers pass; the checker is eligible only if at least 90% of correct and cited answers are kept at that threshold. The actor pool, deterministic checks, checker tie break, second-look threshold grid and second-look gate are unchanged.

The selection report will also show exact yes and no answers separately for every checker and every checker threshold, among eligible answers that passed deterministic checks. Terminal punctuation is ignored when identifying yes/no. These counts are diagnostic only and do not affect either gate.

The eventual committed development decision will be recorded separately as `## Amendment 3: development selection` along with the exact selection report, before any sealed-test output is opened.
