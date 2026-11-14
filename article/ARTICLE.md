# Teaching a small model to search, cite and refuse when evidence is missing

**Team Against-the-safety** · Lexsi AI Safety and Alignment Hackathon · Track 3, PS 4: fine-tuning for RAG

## Abstract

Document assistants need to answer from retrieved evidence and decline when it is missing.

We fine-tuned Tiny Aya Global (3.35B) with QLoRA through AgentTune to make up to three searches, then return a cited answer or a refusal. A fail-closed rule converts invalid endings to refusals. A self-check uses the same model to assess each answer against its citations.

On the opened round-5 evaluation, correct and cited answers on answerable episodes rose from 5.7% for the base model to 46.3% for main with both repairs. Answers given when evidence was hidden fell from 281/375 to 42/375. Of the answers shown by the repaired system, 67.3% counted as correct. Its macro success was 58.0%, compared with 42.0% for a policy that refuses every question.

Preference training did not produce a checkpoint that met the release gate. On 925 shared, previously opened episodes, macro success was 58.4% for main, 64.5% for candF1, 70.7% for GPT-6 Luna and 68.8% for DeepSeek V4 Flash. Correct among answers shown was 67.8%, 76.6%, 66.1% and 67.7%, respectively. Answer coverage and checker settings differed; hosted models were told the search limit and answer format.

The test is English and uses short factual questions. Evidence labels are annotation-based and need human review. Corrected scores were calculated after the test was opened, so further model selection needs a fresh holdout.

## 1. Problem and approach

A document assistant should search when needed, use the retrieved passages and refuse when they cannot support an answer. We trained Tiny Aya Global with QLoRA on matched evidence states: evidence found, distractors, follow-up searches, evidence already supplied, and evidence hidden. The model calls `search_corpus`, cites chunk IDs when answering, or abstains. Invalid endings are converted to refusals. The main adapter also checks each proposed answer against its cited passages; this self-check is V.

## 2. Lexsi repositories and features used

[AgentTune](https://github.com/Lexsi-Labs/AgentTune) supplies the trainer integration, SQLite retrieval backend, search tool, episode logs and trajectory storage. [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) provides RAG and agent metrics, including evidence recall, citation support, abstention and tool-call validity. The final table uses our corrected scorer v3, not an AuditKIT composite. Later preference-training experiments used our own memory-efficient DPO loop because the generic trainer did not fit the GPU.

## 3. Models and licenses

The base is [Cohere Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global), gated and licensed CC BY-NC 4.0. Its published LoRA adapters inherit the noncommercial terms. The [model collection](https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc) includes the selected main adapter and the candF1 alternative. Repository code is MIT. AgentTune and AuditKIT are source-available dependencies under their respective LSAL terms.

## 4. Datasets and licenses

Training and evaluation use HotpotQA and SQuAD-derived material under CC BY-SA 4.0, RGB-derived material under CC BY-NC-SA 4.0, and a fictional Nimbus Home support suite under CC BY 4.0. Other benchmark traces retain their source licenses: 2WikiMultihopQA Apache-2.0 and MuSiQue CC BY 4.0. The mixed data card identifies the component licenses; do not treat the whole bundle as commercially usable.

## 5. Collections and demo

[Models](https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc) · [Datasets and demo](https://huggingface.co/collections/BMEscanor/against-the-safety-rag-episodes-traces-and-demo-6aec1fdb994e3b036c6c958d) · [Static trace viewer](https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer)

The viewer displays saved traces and does not run the model. The GitHub repository is private at present, so its links are not anonymously accessible.

## 6. Experiments and findings

| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Refuses every question | 42.0% | 0.0% | 0/375 | 0.0% | — |
| Base Tiny Aya | 3.1% | 5.7% | 281/375 | 77.6% | 4.4% |
| Main, no repairs | 47.9% | 48.8% | 115/375 | 56.3% | 51.9% |
| Main with both repairs | 58.0% | 46.3% | 42/375 | 41.2% | 67.3% |
| candF1 with both repairs | 64.0% | 53.3% | 40/375 | 41.8% | 76.4% |
| GPT-6 Luna with fail-closed | 60.4% | 55.3% | 91/374 | 68.0% | 48.7% |
| DeepSeek V4 Flash with fail-closed | 64.9% | 63.7% | 60/373 | 61.1% | 62.3% |

Corrected scorer v3; hosted macro scores use the respective shared subsets while their remaining columns use each model's own available cases. Macro averages four task components and counts correct refusals as success, so the refuse-everything reference is essential. Coverage and correct-among-answered expose the tradeoff. Hidden-evidence answers are annotation-based support flags, not human-verified hallucinations. The held-out test was opened before the correction, so further selection needs a fresh test.

In the format-aware rerun, all four systems below use the same 925 eligible episodes. Hosted prompts stated the three-search limit and exact answer format; the adapters learned the format from training. Hosted models used fail-closed handling without V. The systems still differ in decoding and checking, so this is a system comparison.

| System | Macro success | Answered | Correct among answered | Wrong or unsupported shown / all cases |
| --- | ---: | ---: | ---: | ---: |
| Main with both repairs | 58.4% | 41.3% | 67.8% | 13.3% |
| candF1 with both repairs | 64.5% | 42.1% | 76.6% | 9.8% |
| GPT-6 Luna with fail-closed | 70.7% | 70.5% | 66.1% | 23.9% |
| DeepSeek V4 Flash with fail-closed | 68.8% | 64.2% | 67.7% | 20.8% |

The full report includes paired clustered-bootstrap intervals.

The illustrative cost at 10,000 episodes is $6.30 for busy T4 time, $2.67 for GPT-6 Luna and $6.09 for DeepSeek; the API estimates use the format-aware runs' recorded token mix and earlier published rates. The GPU assumption is $0.526/hour at perfect utilization, excluding other infrastructure costs.

Training without refusal examples increased coverage but weakened abstention. The reading patch and preference-training checkpoints failed their registered release conditions. Some later preference-training prompts exposed held-out HotpotQA evidence; main and candF1 were trained earlier. The checker blocks some unsupported outputs but also removes correct answers. English factual questions, annotation coverage and small subsets limit generalization.

### What failed and what we learned

We trained V2m and V2b as answer checkers and evaluated their decisions on pooled development answers that passed deterministic checks. After seeing training-validation results, we expanded the checker threshold grid from 0.50 down to 0.20 before any Round 6 development-model output, while keeping both gate conditions unchanged. At 0.95, both passed 8/40 hidden-evidence answers; V2m kept 113/173 correct answers and V2b kept 105/173. Because no tested threshold met the hidden-evidence limit, neither checker was selected, the second look remains off, and no model has been evaluated on the Round 6 sealed test. The development selection report lists yes/no diagnostics separately for each checker; those diagnostics did not affect the gates.

## 7. What more credits would enable

We would train quote-first answer targets, balance supported answers and refusals, state the semicolon format for two-part answers, and expand the parent-question pool. Further work would build more grounded yes/no and difficult hidden-evidence examples, retrain the checker on independent data, and evaluate it under a new preregistered selection protocol.

## 8. Team

[BMEscanor](https://huggingface.co/BMEscanor) and [abd420](https://huggingface.co/abd420). A friend provided Kaggle compute with permission; the friend is not listed as a teammate.
