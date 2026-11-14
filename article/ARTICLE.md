# Teaching a small model to search, cite and refuse when the evidence is missing

**Team Against-the-safety** · Lexsi AI Safety and Alignment Hackathon · Track 3 (agentic fine-tuning), PS 4: fine-tuning for retrieval-augmented generation

Code: https://github.com/Bhaveshmeghwal21/against-the-safety · Models and data: see section 5

## Abstract

A document assistant is only useful if it answers from its documents and says so when they don't hold the answer. We fine-tuned Tiny Aya Global (3.35B) with QLoRA through AgentTune so that it searches a document set up to three times, then answers with citations or refuses. Two repairs sit on top of the model and need no training: a fail-closed rule that turns any invalid ending into a refusal, and a self-check in which the same model reads the cited passages and withholds an answer they do not support.

We scored every system with deterministic code on a sealed test of 938 SQuAD, HotpotQA and RGB episodes, about 40% of which hide the evidence. Against the base model, the submitted system raises correct, cited answers on answerable episodes from 5.7% to 46.3% and cuts answers given when the evidence was hidden from 281 of 375 to 42 of 375. Of the answers it shows, 67.3% are correct. Macro success, which counts a correct refusal as success, is 58.0%. A policy that refuses every question scores 42.0%, so we report four measures and not macro success alone. In a format-aware comparison on 925 shared episodes, GPT-6 Luna and DeepSeek V4 Flash score 70.7% and 68.8% macro success, but show a wrong or unsupported answer in 23.9% and 20.8% of cases, against 13.3% for our system and 9.8% for its alternative checkpoint candF1 (64.5% macro success).

Every later training change failed its own pre-registered release rule: a patch for single-passage questions, a matched-evidence curriculum, preference training (its checkpoints stopped abstaining) and trained answer checkers. The test covers English short factual questions. The sealed test was opened before our scoring corrections, so further model selection needs a fresh test, and hidden-evidence answers are annotation-based flags, not human-checked hallucinations.

## 1. Problem and approach

A document assistant has to do four things: decide when to search, find the right passages, answer only from what it found, and refuse when the passages do not hold the answer. A small model usually does the first two passably and fails the last two. The base Tiny Aya Global answered 75% of our test episodes whose evidence we had hidden.

We trained Tiny Aya Global (3.35B) with QLoRA to run a small protocol: call `search_corpus` up to three times (three passages per search, 800 characters each), then end with `<answer>` and chunk citations, or `<abstain>` saying what is missing. The training data are scripted episodes built from 600 HotpotQA training questions in six evidence states: clean support, a distractor ranked first, second-hop recovery, evidence hidden (the right move is to refuse), evidence supplied in the prompt, and a weak first search. That gave 1,611 training turns, 537 each of search, answer and refusal. A control adapter, trained on the answering episodes only, shows what the refusal examples change.

Two repairs run on the model's output and need no training. The **fail-closed rule** turns a rollout without a valid ending (repeated query, fourth search, bad format) into a refusal. The **self-check** has the same model read the question, the cited passages and the answer; the answer is shown only if P(SUPPORTED) is at least 0.5, the cited passage was actually shown, and every part of the answer appears in it.

![How one question is answered](../docs/figures/png/pipeline.png)

## 2. Lexsi repositories used

| | Used for | Where in the repository |
| --- | --- | --- |
| [AgentTune](https://github.com/Lexsi-Labs/AgentTune) | `create_distill_trainer` with a QLoRA trainer factory; `SQLiteFTSBackend` and the corpus search tool the model calls; `EventLog` and `TrajectoryStore` for every training episode and evaluation rollout; `qa_metrics` | `src/ragharness/trainer.py`, `retrieval.py`, `tools.py`, `episodes.py` |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | each episode as an `ak.Sample` with a `RAGTrace`, scored with `ak.evaluate`: evidence-set recall, citation support, abstention, tool-call validity, redundant tool calls, loop detection, and freshness on dated documents | `src/ragharness/audit.py`, `scorecard.py` |

AuditKIT results from the first round (HotpotQA test split, scorer before the 13 November corrections):

| AuditKIT metric | Base | Main | Control |
| --- | ---: | ---: | ---: |
| Evidence-set recall | 0.512 | 0.727 | 0.824 |
| Citation support | 0.302 | 0.771 | 0.941 |
| Abstention | 0.721 | 0.794 | 0.714 |
| Tool-call validity | 1.000 | 1.000 | 1.000 |
| Redundant tool calls (lower is better) | 3.1% | 2.9% | 8.8% |
| Loop-free episodes | 94.1% | 93.4% | 78.4% |

The headline numbers below come from our own scorer, because it checks that an answer is correct and that the cited passages contain the required evidence. We did not run AuditKIT's shipped benchmark suites (ARC, GSM8K, MMLU, TruthfulQA, BFCL); none of them tests retrieval. AgentTune's DPO trainer ran out of memory on a T4, so preference training used our own short loop with the same loss.

## 3. Models and licenses

The base is [Cohere Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global), gated, CC BY-NC 4.0. Our adapters inherit that license. The submitted adapter is `main`; `candF1` is the alternative. Repository code is MIT. AgentTune (LSAL v1.1) and AuditKIT (LSAL v1.2) are source-available dependencies.

## 4. Datasets and licenses

Training and evaluation use HotpotQA and SQuAD 2.0 (CC BY-SA 4.0), RGB (CC BY-NC-SA 4.0, noncommercial), 2WikiMultihopQA (Apache-2.0), MuSiQue (CC BY 4.0), and a fictional product-support suite, Nimbus Home, that we wrote (CC BY 4.0). The mixed episodes bundle contains RGB-derived rows, so do not treat it as commercially usable.

## 5. Collections and demo

- Models: https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc
- Data, traces and demo: https://huggingface.co/collections/BMEscanor/against-the-safety-rag-episodes-traces-and-demo-6aec1fdb994e3b036c6c958d
- Trace viewer (recorded episodes, the model does not run): https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer
- Web demo that runs the model on a GPU: `scripts/demo_app.py`, described in the repository README. Its eight example questions ran on a Kaggle T4: five right, two wrong answers, one over-refusal.

## 6. Experiments and findings

We scored every system with deterministic code on a sealed test of 938 SQuAD, HotpotQA and RGB episodes that nothing earlier had used (934 scorable). About 40% of them hide the evidence. The scorer (v3) counts an answer as correct only if it matches the reference and cites every required piece of evidence.

![Four measures for six systems](../docs/figures/png/headline.png)

| System | Macro success | Correct and cited | Answers when evidence was hidden | Correct among answers shown |
| --- | ---: | ---: | ---: | ---: |
| Tiny Aya base | 3.1% | 5.7% | 281 / 375 | 4.4% |
| Main adapter alone | 47.9% | 48.8% | 115 / 375 | 51.9% |
| **Main + both repairs (submitted)** | **58.0%** | 46.3% | **42 / 375** | 67.3% |
| candF1 + both repairs (alternative) | 64.0% | 53.3% | 40 / 375 | 76.4% |
| Refuse every question | 42.0% | 0.0% | 0 / 375 | none |

Macro success averages four components (SQuAD single, SQuAD two-part, HotpotQA, RGB) and counts a correct refusal as success, so refusing everything scores 42.0%. That is why we report the other three measures beside it. *Hidden-evidence answers* are answers the annotations do not support; they are not human-checked hallucinations.

**What the repairs do.** On the adapter alone, 115 of 375 hidden-evidence episodes get an answer. The two repairs cut that to 42 and lift macro success from 47.9% to 58.0%, at the cost of coverage: the system refuses 39% of answerable questions. The accuracy gain over the base model comes from training (5.7% to 48.8% correct and cited); the repairs leave it at 46.3%.

**Main or candF1?** Main is the registered system: we chose it before the test was opened. candF1 scores 6.0 points higher on macro success (95% interval +3.0 to +9.2), but a gate written after we saw its development results selected it, and its hidden-evidence answer rate is not clearly lower (−0.5 points, interval −4.6 to +3.4; our rule needed the upper end at +2 or below). We publish it as the alternative.

### Against hosted models

On the 925 episodes both sides could answer, with the hosted models told the three-search limit and the answer format (the adapter learned them in training):

![Correct answers against answers when the evidence was hidden](../docs/figures/png/tradeoff.png)

| System | Macro success | Answered | Correct among answered | Wrong or unsupported shown, of all cases |
| --- | ---: | ---: | ---: | ---: |
| Main + both repairs | 58.4% | 41.3% | 67.8% | 13.3% |
| candF1 + both repairs | 64.5% | 42.1% | 76.6% | 9.8% |
| GPT-6 Luna | 70.7% | 70.5% | 66.1% | 23.9% |
| DeepSeek V4 Flash | 68.8% | 64.2% | 67.7% | 20.8% |

GPT-6 Luna is 6.2 points above candF1 on macro success (interval +1.1 to +11.1); DeepSeek V4 Flash is 4.4 above (interval −0.3 to +9.1). Both hosted models answer far more often, and show a wrong or unsupported answer about twice as often as candF1. This is a system comparison, not a fair fight between models: decoding and checking differ, and the test was opened before the corrections.

**Cost for 10,000 episodes.** $6.30 for the adapter's busy T4 time (perfect utilisation, $0.526 per hour, nothing else counted), $2.67 for GPT-6 Luna and $6.09 for DeepSeek V4 Flash, from the recorded token mix and published prices. The adapter is not cheaper. Its case is control: one GPU you own, and fewer unsupported answers shown.

### Sample inputs and outputs

![Four recorded episodes](../docs/figures/png/trace_examples.png)

### What we tried

![Rounds 1 to 6 and their outcomes](../docs/figures/png/rounds.png)

- **Refusal examples are what make the model stop.** The control adapter answered more but never refused; it kept searching until its budget ran out.
- **Cheap fixes beat more training.** The two repairs added 10 points of macro success with no retraining. Every training change after round 1 missed its own pre-registered release rule.
- **Preference training removed refusals.** Its pairs almost never chose an abstention, so the checkpoints stopped abstaining. A second run with abstention pairs restored it from main but passed no gate. Some of these training prompts also exposed held-out evidence; main and candF1 predate them, and the cleaned pair files are in the remediation report.
- **A trained checker did not beat the plain one.** Two adapters trained to check answers never reached our hidden-evidence limit at any threshold. The Round 6 sealed test is unopened.

## 7. Compute and cost

The submitted adapter trained for 3.9 T4 GPU-hours on Kaggle's free quota (about $2 at $0.526 per hour). Later rounds ran on a friend's Kaggle account with permission, plus local CPU for scoring. The hosted-model comparison cost about $0.84 at published prices.

## 8. Limitations

- English, short factual questions. Two-part answers and single-passage questions are weak spots, and with two editions of a policy the adapter sometimes cites the older one.
- The sealed test was opened before the scoring corrections, and candF1 was tested on it after its gate was written. Further model selection needs a fresh test.
- Hidden-evidence answers are annotation-based flags. An unannotated alternative source could make a flagged answer correct, and nobody has hand-checked them.
- The self-check withholds some correct answers.
- No hosted live endpoint exists. The web demo needs a GPU you start yourself.

## 9. What more credits would enable

Quote-first answer targets, so the model copies the supporting span before answering. Balanced answer and refusal data in both directions. The semicolon format for two-part answers stated in the prompt, and far more than 488 parent questions. Retraining the checker on independent labelled data, judged on a fresh sealed test under a new pre-registration.

## 10. Reproduction and artifacts

The repository README has the commands: tests, the audit that rebuilds the corrected reports from saved transcripts, and the Kaggle launcher for training and evaluation. Environment: Python 3.12; training and evaluation on Kaggle T4s. Pinned package versions are in `requirements-gpu.lock.txt` and `requirements-cpu.lock.txt`.

## 11. Declarations

- **Work outside the hackathon:** none of the submitted work predates the hackathon. The repository history starts on 4 November 2026, one day after the kickoff-day planning note in `research/`. The base model, AgentTune, AuditKIT and the public datasets are third-party.
- **LLM usage:** AI coding assistants drafted a large share of the code, tests and prose under the team's direction. Every number here comes from a script run on saved transcripts and appears in a file under `reports/`.
- **Acknowledgements:** a friend lent Kaggle compute. Cohere Labs, Lexsi Labs and the dataset authors made the models, libraries and data used here.

## 12. Team

[BMEscanor](https://huggingface.co/BMEscanor) and [abd420](https://huggingface.co/abd420).
