# General-capability retention check: pre-registration

Written 16 November 2026, before any model was run on these sets. The hackathon's write-up list asks Track 3 teams for a general-capability benchmark that shows the fine-tuning did not break something else, with the languages covered and whether each is part of the focus task or a retention check. We had not run one. This is a retention check only. Nothing here changes the submitted system or any earlier result.

## 1. Question

Do the two adapters we publish as English systems (`main`, submitted; `candF1`, alternative) lose general knowledge or reasoning compared with the base model they were trained from?

## 2. Benchmarks

All are public, ungated test sets, rebuilt by `scripts/prepare_general.py` (hashes in `data/general/manifest.json`). All are multiple choice. None overlaps our RAG data: our training rows come from HotpotQA and SQuAD training splits.

| Benchmark | Questions | Language | Role |
| --- | ---: | --- | --- |
| MMLU (`cais/mmlu`, all subjects, test) | 14,042 | English | retention check, knowledge |
| ARC-Challenge (`allenai/ai2_arc`, test) | 1,172 | English | retention check, reasoning |
| ARC-Easy (`allenai/ai2_arc`, test) | 2,376 | English | retention check, reasoning |
| TruthfulQA MC1 (`truthfulqa/truthful_qa`, validation) | 817 | English | retention check, truthfulness (choices shuffled with a fixed seed, because the source lists the right answer first) |
| Global-MMLU-Lite, English (`CohereLabs/Global-MMLU-Lite`, test) | 400 | English | retention check, counterpart of the next two |
| Global-MMLU-Lite, Hindi | 400 | Hindi | retention check in a language we did not train in; the same 400 questions as the English set |
| Global-MMLU-Lite, Spanish | 400 | Spanish | same |

The English adapters were trained on English only. The Hindi and Spanish sets therefore show what the English training did to those languages. They are not part of the focus task (retrieval-augmented answering).

## 3. Systems and protocol

- Systems: `base` (Tiny Aya Global, revision `af89d219`, adapters switched off), `main` and `candF1` (the published weights, byte-identical to the files in the Hugging Face repositories). One loaded 4-bit model per process; the adapter is switched, not the model.
- Prompt: plain text, zero-shot, no system prompt and no chat template, in the usual letter format ("The following are multiple choice questions (with answers) about {subject}. ... A. ... B. ... Answer:"). The answer is the letter whose token has the highest next-token probability after "Answer:". The same text goes to every system. Nothing is sampled.
- Accuracy is the share of questions answered right. Chance is 25% for four-choice sets.
- Absolute scores will differ from published ones, which use other prompts, few-shot examples and a log-likelihood protocol, and AuditKIT's lm-eval engine is not used here. Only the differences between our systems, under one protocol, carry meaning.
- Compute: Kaggle T4 x 2, estimated 30 to 40 minutes. No paid service.

## 4. Statistics and rules

- Paired 95% bootstrap interval of each adapter's accuracy minus the base model's, over questions (3,000 draws, seed 47). Also each system's own interval.
- Rule, registered per adapter: **the adapter is retained on a benchmark if the lower end of the interval of (adapter minus base) is at least -3.0 points.** Applied to MMLU, ARC-Challenge and ARC-Easy, the three English sets with enough questions to decide. Three rules per adapter.
- TruthfulQA and the Global-MMLU-Lite sets have no pass or fail rule. With 400 to 817 questions, an interval is about +/-4 to 5 points wide, too wide for a 3-point margin. They are reported with their intervals.
- Whatever happens, the table goes into the README, the article and the model cards. If a rule fails, the documents say which benchmark and by how much, and the model cards for that adapter carry the number.

## 5. What this does not show

- It does not test generation tasks (GSM8K, code), safety refusals, or long-context behaviour.
- The adapters were trained with a chat template and a search protocol. Under the plain prompt used here, a format effect could lower scores even if knowledge is intact. A chat-format variant is not run.
- One protocol, one prompt wording, one run per system, no seeds to vary.

## 6. Files

`scripts/prepare_general.py`, `scripts/general_capability.py`, `scripts/report_general.py`; `data/general/manifest.json`; results in `reports/general_capability.{json,md}`. Deviations are labelled where they occur.

## 7. Amendment 1 (16 November 2026, written while the MMLU re-run was in progress)

- **What happened.** The first Kaggle launch (`ats-general-t4x2`, two T4s) ran the six smaller benchmarks on one GPU and MMLU on the other. The MMLU process stopped before scoring anything: reading the benchmark file with Python's `splitlines()` broke a JSON line, because two MMLU questions contain a Unicode line-separator character. The loader now splits on newlines only. MMLU was re-run alone in a second kernel (`ats-general-mmlu`), with the same code otherwise, the same adapters and the same prompts. The report reads each benchmark from the one kernel that holds it and refuses to run if a benchmark is missing or duplicated.
- **What was seen before MMLU finished.** The run summaries of the six smaller benchmarks (accuracy only, no intervals) were read while MMLU was still to run. No definition, margin or rule changed after that. The registered rules cover MMLU, ARC-Challenge and ARC-Easy as written in section 4.