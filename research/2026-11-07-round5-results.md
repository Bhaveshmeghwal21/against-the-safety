# Round 5 results: priorities 0 to 2 of plan section 27 (7 November 2026)

Registered design, amendments and every decision before its result: `research/2026-11-07-round5-preregistration.md`. Compute: Kaggle account 2 (`vibhutimeghwal`), about 8 hours of session time (sampling 1.2 h, training and development evaluation 3.9 h, sealed-test traces 2.0 h, checker runs 0.5 h, two failed launches 0.3 h). No result below was used to choose anything after the fact except where it says so.

## In one paragraph

Two cheap, model-free repairs, applied to the unchanged `main` adapter, produce a large and significant gain on a sealed test of 938 fresh episodes: **fail-closed** (an invalid or looping turn ends as an abstention) and a **self-check V** (Tiny Aya, with main's adapter, re-reads each answer against the passages it cited and withholds it if it is not supported). Together they lift macro success from 0.414 to 0.517 and cut the unsupported-answer rate from 29.7% to 10.2%, at a cost of 1.4 points of supported-correct answers. Preference training (DPO, priority 2) raised answer accuracy on SQuAD-style questions a lot but removed the model's ability to abstain, hurt multi-hop questions, and no checkpoint passed the pre-registered development gates. **The round-5 system is `main` + fail-closed + V.** The DPO result is a negative result with a clear cause (below).

## Sealed final test (`r5test`, 938 episodes: 385 SQuAD, 82 HotpotQA, 471 RGB; opened once)

Scorer version 2; macro = mean of four component rates (SQuAD single, SQuAD compound, HotpotQA, RGB); unsupported = unsupported answers / hidden-evidence episodes (`reports/round5_final.md`).

| system | macro | single | compound | HotpotQA | RGB | unsupported / hidden | supported correct | false refusals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base (Tiny Aya Global, no adapter) | 0.022 | 0.018 | 0.036 | 0.000 | 0.034 | 276 / 369 | 0.027 | 0.011 |
| main | 0.414 | 0.545 | 0.224 | 0.524 | 0.363 | 108 / 364 | 0.290 | 0.222 |
| main + fail-closed | 0.474 | 0.600 | 0.327 | 0.561 | 0.407 | 108 / 364 | 0.290 | 0.270 |
| **main + fail-closed + V** | **0.517** | 0.591 | 0.394 | 0.561 | 0.524 | **37 / 364** | 0.275 | 0.391 |
| candA (reference) | 0.473 | 0.636 | 0.250 | 0.622 | 0.385 | 129 / 365 | 0.337 | 0.160 |
| candA + fail-closed + V | 0.541 | 0.614 | 0.390 | 0.634 | 0.526 | 45 / 365 | 0.309 | 0.323 |

Paired differences, 95% intervals from resampling whole questions within each part (3,000 draws):

| comparison | macro | unsupported rate | supported correct |
| --- | ---: | ---: | ---: |
| main against base | +0.392 [+0.357, +0.425] | -0.451 [-0.508, -0.392] | +0.263 [+0.217, +0.310] |
| main + FB against main (claim 1 of 0a) | +0.059 [+0.041, +0.079] | +0.000 | +0.000 |
| main + FB + V against main + FB | +0.044 [+0.028, +0.059] | -0.195 [-0.233, -0.157] | -0.014 [-0.025, -0.005] |
| **main + FB + V against main** | **+0.103 [+0.079, +0.127]** | **-0.195 [-0.231, -0.159]** | -0.014 [-0.025, -0.005] |
| candA + FB + V against main + FB + V (reference) | +0.024 [+0.003, +0.045] | +0.022 [-0.014, +0.057] | +0.034 [+0.000, +0.066] |

Reading it honestly:
- The fail-closed gain is entirely on hidden-evidence episodes, where ending an invalid turn as an abstention is the right answer. It repairs the protocol, it does not make the model know more.
- V's gain comes from turning unsupported answers into abstentions. On answerable questions it removes a few supported-correct answers (1.4 points) and many wrong ones, so false refusals rise from 0.27 to 0.39. Those episodes were failures already, but a user sees more "I cannot answer".
- Supported-correct accuracy on answerable questions is **not** improved by the wrappers (0.290 to 0.275). Raising it was the aim of priority 2, which failed.
- candA with the wrappers is slightly ahead of main with them on macro (+0.024) and not different on unsupported answers. It was a reference and was not promotable; main stays the default.

## V, the self-check (priority 1)

Four configurations (checker = base model or main's adapter; wording A "SUPPORTED / NOT" or B "Yes / No"; score = probability of the positive first token, blocked below 0.5; no generation). Development gate from section 27: block at least 40% of main's unsupported answers and at most 10% of its supported-correct answers.

| configuration | unsupported blocked (dev, 20) | supported correct blocked (dev, 76) | gate |
| --- | ---: | ---: | --- |
| base, A | 60% | 24% | no |
| base, B | 75% | 21% | no |
| **main adapter, A** | **70%** | **8%** | **yes** |
| main adapter, B | 100% | 87% | no |

Out of sample, on the round-4 test traces that were never used to choose V: main 59% of 39 unsupported blocked at 14% of 51 supported-correct; candA 67% at 5%; candE1 66% at 9%. On the sealed test V takes main from 108 to 37 unsupported answers. The deterministic checks (cited chunk never shown; answer text absent from the cited passages) contribute little by themselves (3 of 20 unsupported on development, 1 of 76 supported-correct).

## Scorer version 2 (priority 0c)

Version 1 reproduces the stored scores exactly in all 12 development runs. Version 2 adds a token-precision rule (at least 0.5 against the closest accepted gold form) and drops hidden-evidence episodes whose shown passages hold the whole answer (one episode per system on `r4dev`, none on HotpotQA dev). A first draft of the second rule dropped 27 of 132 hidden episodes because it treated one visible half of a two-part answer as enough; that was caught and corrected before any result was used. The precision rule lowers every system's HotpotQA supported success (main 0.314 to 0.256). `reports/round5_rescore_r4.md` lists both versions for all round-4 runs.

## Priority 2: preference training (DPO)

Data: 1,920 sampled rollouts (main and candA, temperature 0.8, 311 SQuAD-style and 169 HotpotQA training episodes, two samples each) gave 335 pairs over 77 questions, SHA256 `8214a2c6...1508` (186 pro-answer, 149 pro-caution; `reports/round5_pairs.json`). Training: LoRA from main (candG) and from candA (candH), beta 0.1, 2 epochs, 84 steps each, about 1.6 hours per GPU, peak 8.0 GB. DPO loss fell from 0.67 to 0.19, preference accuracy rose to 0.93 to 0.95, and the log-likelihood of the rejected texts fell by about 17 nats while the chosen texts rose.

Development selection (`reports/round5_selection.md`; scorer v2, FB and V applied to every system, eight candidates):

| system | macro | SQuAD single | compound | HotpotQA | supported correct | unsupported / hidden | HotpotQA unsupported | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.538 | 0.603 | 0.387 | 0.623 | 0.228 | 5 / 131 | 1 | reference |
| candG1 | 0.481 | 0.529 | 0.423 | 0.491 | 0.148 | 2 / 132 | 2 | no |
| candG2 | 0.558 | 0.700 | 0.432 | 0.541 | 0.381 | 9 / 132 | 9 | no |
| candH1 | 0.546 | 0.678 | 0.400 | 0.560 | 0.360 | 12 / 129 | 6 | no |
| candH2 | 0.557 | 0.678 | 0.445 | 0.547 | 0.386 | 12 / 129 | 8 | no |
| candE1 (round 4) | 0.568 | 0.593 | 0.468 | 0.642 | 0.265 | 5 / 131 | 3 | no |
| candF1 (round 4) | 0.583 | 0.612 | 0.495 | 0.642 | 0.312 | 7 / 131 | 2 | no |

**No candidate is eligible, so none was opened on the sealed test.** The round-4 checkpoints pass the macro, supported-correct and HotpotQA-success gates but not the unsupported-answer gates; the DPO checkpoints raise SQuAD-style accuracy sharply (candG2 single 0.700 against 0.603, supported correct 0.381 against 0.228) but fail on HotpotQA and on unsupported answers.

**Why DPO failed, found by counting actions in the development traces.** Main finishes 160 of 321 `r4dev` episodes with an abstention. All four DPO checkpoints finish **none** with an abstention (candG2: 256 answers, 65 invalid, 0 abstentions; candH2: 288 answers, 33 invalid, 0 abstentions; same on HotpotQA dev). The pair file contained 91 pairs that reject an abstention and only 2 that choose one, because on-policy sampling almost never produces an unsupported answer at the last search, which is the only place the reference action is an abstention. DPO pushed abstentions down by about 17 nats and nothing pushed them up, so the model lost the action. The fail-closed repair and V then convert its looping and unsupported answers into abstentions, which is why the wrapped numbers still look reasonable; the abstention in those systems is carried by the wrappers, not the model. This is the objective-shape problem section 27 was written around, reproduced one level up: a preference set must contain both directions for every action it touches.

The first epoch of candG (candG1) is worse than the second: 224 of 321 episodes end invalid (repeated queries and search-budget overruns). Why the epoch-1 snapshot is worse is not established; it is reported, not explained.

## What this changes and what is next

- Headline: `main` + fail-closed + V. Pre-registered, significant on a sealed test, no retraining.
- Not done: priority 3 (GRPO with the +1 / 0 / -1 reward) is the plan's next step. The DPO failure sharpens why: that reward gives the model an on-policy reason to keep abstaining when evidence is hidden, which a pair set built from errors cannot. A corrected DPO is the cheaper alternative: add pairs that choose an abstention, with the rejected answer obtained by forcing an answer at depth-3 hidden-evidence contexts, raise the HotpotQA share, and stop on an abstention-rate guard. Either needs about 4 GPU hours and a new pre-registration.
- Nothing from this round is published on the Hub yet. The README, Article and abstract still describe rounds 1 to 4.
