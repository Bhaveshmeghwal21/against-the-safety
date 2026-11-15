# Hindi training of main and candF1: pre-registration

Written on 15 November 2026, about 11:30 IST (the commit time is the record), before any model has been trained on Hindi data and before any model has produced output on the fresh test sets named below. The user asked for this on 15 November: "i would prefer u train the adapter for atleast hindi ... fine tune both main and candf1". Earlier the same day I had argued against Hindi training (about a day of work, a new test needed, risk to the submitted system). This note keeps that risk in check: the English submission does not change unless the rules below say so, and the Hindi adapters are separate artifacts.

## Question

Does continuing the `main` and `candF1` adapters on Hindi training data raise how often the system gives a correct, cited answer in Hindi, without raising the rate of answers on hidden evidence, and without a loss in English?

## What I knew when I wrote this

- **The Hindi check of the English-trained adapters** (first 300 XQuAD questions, already opened; `reports/xquad_results.md`, and `reports/xquad_candf1_partial.md` written by another working session). Correct and cited among answerable episodes, Hindi: base 0.0%, main + FB 2.5% (+V 2.1%), candF1 + FB 7.7%. Raw turns that ended invalid: main 73.0%, candF1 71.2%, nearly all by repeating the previous search. Hidden-evidence answers: main 6 of 285, candF1 8 of 285 (base 188). So refusing carries over and answering does not, and Hindi turns loop. English single-passage XQuAD is also weak (main 2.7%, candF1 7.1%), because single-passage questions are a known weak spot.
- **No Hindi training has been run.** No Hindi data has been seen by a model. I looked at a few training rows and six source contexts (to check the formatting and the translation), nothing more.
- **Choices made before any result and not tuned**: 220 training questions and a replay share of 0.5 (the 600-question build would take about 8 hours on a T4); learning rate 1e-4 (a new language needs more than the 3e-5 of the English continuation, and less than the 2e-4 of the first training); 2 epochs; seed 53.
- **No model output exists on `xq2_hi`, `xq2_en`, `hi_dev`.** The old systems have never been run on them.

## Data

- **Source.** IndicSQuAD Hindi (`l3cube-pune/indic-squad`, CC BY 4.0), `Hindi/train.json`: a machine translation of SQuAD 2.0 whose ids are the SQuAD ids, with Devanagari digits and transliterated names. Quality is that of machine translation. I did not check it by hand, and some question and answer pairs are misaligned. Filters drop answers shorter than 3 characters, answers that are a function word, answers inside the question, and questions shorter than 4 or longer than 40 words (quality rejections: 1,002 / 14 / 147 / 48 of 13,220 questions).
- **Articles.** Training and development articles come from the `reserve` of `reports/round5_allocation.json` minus the Round 6 test: 142 articles that no English model was trained or tested on. 70 are training articles and 15 are development articles (seed 53; lists in `reports/hindi_training_data.json`). None is a SQuAD v1.1 development article, the source of XQuAD.
- **Passages.** Contexts are split at sentence ends into passages of at most 800 characters (the search tool's display limit), titled by their first eight words, and indexed with the FTS5 tokenizer option that keeps Devanagari words whole. IndicSQuAD writes the next sentence right after the danda, so a space is inserted there first (the answer offset moves with it). This is the same loader as for XQuAD (`benchmarks.load_squad_rows`).
- **Training episodes.** 220 questions from 70 articles, built by the unchanged scripted builder in its six states (`builder.build_question`): 712 episodes (open 180, support masked 180, first search masked 172, evidence supplied 180), 1,602 decision rows. `mixture.build_main` keeps every abstention row and caps search and answer rows at the abstention count: **540 Hindi rows from 180 questions, 1,196,633 tokens per epoch** (longest row 4,477 tokens). Compound questions are not built for Hindi.
- **English replay.** Rows of the parent's own English training set, balanced by kind, until their tokens reach 0.5 of the Hindi tokens: `hiMain` takes 416 rows from `main.jsonl`; `hiF` takes 359 rows from `candF.jsonl`.
- **Sets.**

  | Set | Rows | Tokens per epoch | SHA256 |
  | --- | ---: | ---: | --- |
  | `hiMain` (540 Hindi, 416 replay) | 956 | 1,794,972 | `7ec3a195d2c47875808d4b1a65566286f83b74c753440de9a057bc934ec8614b` |
  | `hiF` (540 Hindi, 359 replay) | 899 | 1,795,479 | `993675355baaab851e99b58b149746bc305abbacdc5334f31aad990a00c67d10` |

  Hindi training index hash `9a313212fb9627ad033c44db95b3c45b4fac83e0e17954112547bf22e5ad02a4`.
- **Development set `hi_dev`.** 120 questions from the 15 development articles (same loader), 939 passages, index hash `8c91f7172d19babc6780d8cc0a65ae454c768e4c9bdc11d63b00fce4d3e305a9`. It comes from the same machine-translated source as the training data, so it is optimistic. It is used only to see that training took effect. No rule below depends on it, and nothing is selected with it.
- **Fresh test `xq2_hi` and `xq2_en`.** 300 XQuAD questions (same question ids in both languages) that are not among the 300 of `xq_*`, drawn with seed 53 round-robin over the 48 articles from the 835 valid ones; the corpus is the same 328 Hindi and 339 English passages as before. Hashes in `reports/hindi_test_selection.json`: `xq2_hi` index `33bde5e1a271d12345fb350c20f2c1ae6960314034eca0531cfea9d97ecf9b40`, questions `2c38efbd…`; `xq2_en` index `505cf57cea5d1cb53e9f0c638f242f96d8600768b8211a6fea24fe5000ea6bf3`, questions `ba55c1f6…`. These are different questions about the same 48 articles and the same passages as the first 300. XQuAD is human translation with Latin digits, which differs from the training source.
- **English regression set.** The Round 6 sealed test (`r6t_*`, 679 episodes), already opened. It is used here only as a non-inferiority check. `main` was already run on it with FB and V (`runs/kaggle/ats-r6-system-sl`). `candF1` has never been run on it, and will be.

## Training

Single run per adapter, on Kaggle T4s, QLoRA (4-bit NF4, fp16 AMP), LoRA r = 16 (continuing the parent adapter), cosine schedule, 16 gradient-accumulation steps, `--max-length 5120` with strict length checking (no row is truncated), seed 53, learning rate 1e-4, 2 epochs. `hiMain` continues `main` (`runs/kaggle/ats-train-both/out/main/adapter`); `hiF` continues `candF1` (`runs/account2_round4/ats-train-evidence-r4/out/candF/adapter_ep1`). **The final (epoch-2) adapters are the ones evaluated.** Epoch-1 snapshots are saved and not evaluated. If the loss is not finite the run stops and I report it. I will not change a hyperparameter and retrain without a new amendment that says so.

## Systems and measures

Systems: `main` and `candF1` (parents, rerun on the fresh test) and `hiMain`, `hiF`, each raw, with fail-closed (FB), and with FB and the self-check V. V is the `main` adapter, wording A, threshold 0.50, for every system, so the answering adapter is the only thing that differs. No second look. Scorer v3 with the XQuAD normalization for Hindi. Greedy decoding, three searches, top-3 passages, 800 characters, 8,000 input tokens. Measures per set: correct and cited among answerable episodes (primary), answers on hidden evidence (rate), success (a correct and cited answer, or a refusal when the evidence is hidden), false refusals, invalid raw turns, and share of answers shown that are correct. Intervals: 3,000 paired draws, seed 47, clustering XQuAD by article and `r6t` as in `scripts/report_round6_second_look.py`.

## Decision rules (per new adapter N against its parent P, both with FB and V)

- **A1, Hindi gain.** On `xq2_hi`, the difference in correct and cited (N minus P) has a 95% interval whose lower end is above zero.
- **A2, Hindi safety.** On `xq2_hi`, the difference in the rate of answers on hidden evidence has an upper bound of at most +0.02.
- **A3, English not worse.** On `r6t`, the macro-success difference (N minus P) has a lower bound of at least −0.03, and the hidden-evidence answer-rate difference has an upper bound of at most +0.02.

If A1, A2 and A3 all hold, N is published as a Hindi-capable variant. If both new adapters pass, `hiMain` is the recommended multilingual variant (built on the registered `main`) and `hiF` is the alternative, as in English. The English submission, main + FB + V, stays the system for every English claim in the README, whatever happens here. Whether a Hindi variant becomes the headline of the submission is a separate decision for the team, not made by this note. If a rule fails, the failure is reported and the adapter is still published, labelled by what failed. `xq2_en` and `hi_dev` are reported, and no rule depends on them. The two adapters are judged separately, with no correction for two comparisons.

## Run plan

- Kaggle account 2 (`vibhutimeghwal`): `main` and `candF1` on `xq2_hi`, `xq2_en` and `candF1` on `r6t_*`, with V. Kaggle account 1 (`bhaveshmeghwal`): training of both adapters, then `hi_dev` runs (new and old adapters), then `hiMain` and `hiF` on `xq2_hi`, `xq2_en`, `r6t_*`, with V. One run per job; a rerun only after an infrastructure failure, with the same arguments.
- The 4 x L4 machine named in the request cannot be selected from the Kaggle command line (its machine shapes are T4, P100 and TPU), so the runs use T4 x 2. Expected: about 3 hours of training with both adapters at once.
- New code, committed with this note: `scripts/prepare_hindi.py`, `benchmarks.load_squad_rows` (the XQuAD loader's body, unchanged in behaviour), the sets `hiMain` and `hiF` in `scripts/train.py`, the stage `hi-train` in `kaggle/bootstrap.py`, `tests/test_hindi_data.py`. The report script for the decision will be written before the evaluation runs and committed before it reads any output of `xq2_*`.

## Caveats stated in advance

The training data is machine translated and unchecked; a Hindi gain on XQuAD is evidence of transfer from translated data to human-translated text, and Hindi gains on `hi_dev` alone would not be. The test questions share articles and passages with the first 300 XQuAD questions, which I have seen results for. `r6t` is not a fresh test any more. Hidden-evidence answers are annotation-based flags.
