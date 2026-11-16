# Training, settings and leak checks

**How the model learned the routine.** Nobody wrote thousands of examples by hand. We built them by script from 600 HotpotQA training questions. A scripted policy plays the routine perfectly in six situations: clean support, a distractor passage ranked first, a second search needed to find the second fact, evidence hidden, evidence already in the prompt, and a weak first search. That made 1,611 training turns, 537 each of search, answer and refusal. QLoRA training ran two epochs on one T4 (3.9 GPU-hours, about $2 at $0.526 per hour).

**Training settings.**

| Setting | main (submitted) | candF1 (alternative) |
| --- | --- | --- |
| Base model | Tiny Aya Global, revision `af89d219`, loaded in 4-bit NF4 with double quantisation and fp16 compute (QLoRA) | same |
| Starts from | the base model | candA, a second-round adapter trained like main on the same questions (1,668 rows, more of them answers) |
| Training rows | 1,611 (537 search, 537 answer, 537 refusal) from 600 HotpotQA training questions | 721 from SQuAD training-split articles: 304 rows from 152 pairs of the same question with the needed fact present and missing, plus 417 replay rows |
| LoRA | rank 16, alpha 32, dropout 0.05, all seven attention and MLP projections | same |
| Optimiser | paged AdamW 8-bit, learning rate 2e-4, cosine schedule, 3% warm-up, no weight decay, gradient clipping at 1.0 | same, learning rate 3e-5 |
| Batch and length | batch 1 with 16-step gradient accumulation, at most 2,560 tokens per row, loss on the model's own turns only | same |
| Length of training | 2 epochs, 202 optimiser steps | the checkpoint after epoch 1 of 2 (92 steps for both epochs) |
| Seed, time, memory | seed 13, 3.9 h on one T4, peak 5.8 GB | seed 47 (candA: seed 13), 2.0 h on one T4, peak 6.0 GB |

**Library settings.** AgentTune 1.1.3 and AuditKIT 1.1.3, pinned in `requirements-gpu.lock.txt` with PyTorch 2.11, Transformers 5.22.1, TRL 1.7.1, PEFT 0.21.2 and bitsandbytes 0.50.2. From AgentTune we set `SQLiteFTSBackend(match_all=False)`, our `search_corpus` subclass (three passages per search, 800 characters each) and `create_distill_trainer` with our QLoRA trainer factory. AuditKIT's metrics were called by name through `ak.evaluate` with their default settings. The judge metrics used AuditKIT's own prompts (version v2) and the judge deployment's default sampling. Other options stayed at the libraries' defaults. The Hindi adapters' settings are in [Other languages](SIDE_STUDIES.md#other-languages-hindi-and-spanish).

**Splits and leak checks.** main trained on HotpotQA training-split questions and candF1 on SQuAD training-split articles. The first sealed test has 938 episodes from 225 questions (SQuAD 65, HotpotQA 60, RGB 100). The second has 679 episodes from 164 questions (75, 60, 29). The build scripts checked by exact match, before any model ran, that no SQuAD article sits in more than one group (training, development, test), that no HotpotQA test question was used before and none of its supporting passages is in an earlier corpus (by title, and for the second test also by normalised text), and that no RGB question repeats an earlier RGB set. For the first-round development and test sets, none of the supporting passages appears in any training row and no question text repeats ([`reports/leakage_report.json`](../reports/leakage_report.json)). We ran no embedding-based or MinHash near-duplicate check. The test files were hashed before they were opened ([`reports/round5_test_manifest.json`](../reports/round5_test_manifest.json)).
