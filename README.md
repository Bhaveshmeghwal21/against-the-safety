# Against-the-safety

**A small AI model that answers questions from your documents, shows which passage it used, and says "the documents don't say" instead of guessing.**

Team Against-the-safety · Lexsi AI Safety and Alignment Hackathon · Track 3 (agentic fine-tuning) · Problem statement 4: fine-tuning for retrieval-augmented generation

[Models](https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc) · [Data and traces](https://huggingface.co/collections/BMEscanor/against-the-safety-rag-episodes-traces-and-demo-6aec1fdb994e3b036c6c958d) · [Trace viewer](https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer) · [Article](article/ARTICLE.md) · [Demo](#try-it)

## The idea in one minute

Companies want assistants that answer from their own documents: product manuals, policies, help pages. The usual recipe is called **RAG** (retrieval-augmented generation). The assistant first *searches* the documents, then *writes an answer* from what it found.

The recipe has a weak spot. When the documents do not contain the answer, a language model often answers anyway, and the answer sounds confident. In our test the untrained model did exactly this in 281 of 375 cases where we had hidden the evidence.

We took a small open model, Tiny Aya Global (3.35 billion parameters, small enough to train on one free GPU), and taught it a routine. It searches up to three times and reads the passages. Then it either answers and names the passages it used, or says the documents don't have the answer. Two safety checks, written as ordinary code, sit on top.

Real outputs from our demo, which uses 24 passages of fictional smart-home manuals:

| You ask | The system says | What happened |
| --- | --- | --- |
| What type of batteries does the Nimbus Lock L1 use? | **4 AA alkaline batteries**, citing the Lock L1 Setup Guide and Troubleshooting page | Answer found and cited. |
| Which Bluetooth version does the Hub H2 use? | "The accessible passages do not contain the evidence needed to answer." | The manuals say the hub has Bluetooth but never say which version, so it refuses after three searches. |
| How many days do I have to return a Cam C3? | **30** | Wrong. The documents hold two editions of the policy (30 days in 2025, 45 in 2026). The system cited both and picked the old one. |

The third row is a real miss, kept in on purpose. More runs are under [Try it](#try-it).

On a locked test of 934 questions, the untrained model gave a correct answer with the right citation in 5.7% of the questions whose documents held the answer. Our system does so in 46.3%. When the evidence was hidden, the untrained model answered 281 of 375 times and ours answered 42 times. The price is caution: ours refuses 39% of the questions it could have answered.

![How one question is answered: search loop, draft ending, fail-closed rule, self-check](docs/figures/pipeline.svg)

## Words used on this page

| Term | What it means here |
| --- | --- |
| RAG | An assistant that searches a set of documents first and answers from what it found. |
| Passage or chunk | A short piece of a document, stored with an ID such as `c00019`. The model cites these IDs. |
| Citation | The IDs of the passages an answer is based on. |
| Refuse, or abstain | Saying the documents do not hold the answer, instead of guessing. |
| Base model | Tiny Aya Global as Cohere Labs released it, before our training. |
| Fine-tuning, adapter, LoRA, QLoRA | Fine-tuning is extra training on top of an existing model. A LoRA adapter is a small add-on file (121 MB here) that holds only what changed, so the 3.35-billion-parameter base stays untouched. QLoRA trains the adapter while the base is squeezed to 4 bits, which fits a 16 GB T4 GPU. |
| Episode | One question, its documents, and the model's whole attempt: searches, passages read, final answer or refusal. |
| Evidence hidden | A test episode where we removed the passages that hold the answer. The right move is to refuse. |
| Unsupported answer | An answer given although the evidence was hidden, our stand-in for "made up". It comes from annotations, and nobody has checked each one by hand. |
| Sealed test | Questions locked away before training and opened only to score finished systems, so nothing can be tuned to them. |
| Scorer | Our fixed code that compares answers with references and checks citations. No AI judge is involved. "v3" is its third version, after a correction on 13 November. |
| Macro success | The average of four groups' success rates. A correct, cited answer or a correct refusal counts as success. Refusing every question scores 42.0%, so read it beside the other measures. |
| Fail-closed rule | If an attempt breaks the rules (bad format, repeated search), the system refuses instead of guessing. |
| Self-check | The model re-reads the cited passages and its draft answer and says whether they support it. If not, the answer is withheld. |
| Pre-registered rule | A pass or fail rule written down before the result was known. |
| Hosted model | A large commercial model run over the internet (here GPT-6 Luna and DeepSeek V4 Flash), used as a yardstick. |
| T4 | A 16 GB GPU that Kaggle gives away for free. |

## How it works

1. **Search.** The adapter calls a tool named `search_corpus` up to three times and reads three passages per search.
2. **Draft.** It ends with `<answer>` plus the IDs of the passages it used, or with `<abstain>` saying what is missing.
3. **Fail-closed rule.** An attempt without a valid ending (a repeated query, a fourth search, a broken format) becomes a refusal.
4. **Self-check.** The same model reads the question, the cited passages and the answer. The answer is shown only if the model's probability of "SUPPORTED" is at least 0.5, the cited passage was really shown to it, and every part of the answer appears in that passage.

Steps 3 and 4 are code and need no training. Steps 1 and 2 come from training.

**How the model learned the routine.** Nobody wrote thousands of examples by hand. We built them by script from 600 HotpotQA training questions. A scripted policy plays the routine perfectly in six situations: clean support, a distractor passage ranked first, a second search needed to find the second fact, evidence hidden, evidence already in the prompt, and a weak first search. That made 1,611 training turns, 537 each of search, answer and refusal. QLoRA training ran two epochs on one T4 (3.9 GPU-hours, about $2 at $0.526 per hour).

## The two Lexsi tools: AgentTune and AuditKIT

The hackathon asked every Track 3 team to use two libraries from Lexsi Labs. One is the workshop where the model is taught, the other is the examiner.

- **AgentTune** is a library for building and training *agents*. An agent is a model that works in several steps and uses tools (here, a search tool) instead of answering in one go. AgentTune supplies the tool, records every step an agent takes in a standard format, and turns those recordings into training data.
- **AuditKIT** is a library that scores models and agents with fixed code, not an AI judge. For RAG it asks whether the system found enough evidence, whether its citations are real and whether it refused when it should have. For agents it checks tool use.

![Where AgentTune and AuditKIT fit: documents, search tool, practice runs, training, test runs, scoring, decision](docs/figures/toolchain.svg)

### How we used AgentTune

| Step | AgentTune part | What we did with it |
| --- | --- | --- |
| Search | `SQLiteFTSBackend`, `SearchCorpusTool` | We built a keyword index of the passages. The `search_corpus` tool the model calls is a subclass of AgentTune's. We made three changes: it shows 800 characters of a passage instead of 350, it hides relevance scores, and it can hide chosen passages for evidence-hidden episodes. AgentTune's backend reports a broken index as "no results", so we added a check that fails loudly when the index is empty or wrong. |
| Recording | `EventLog`, `TrajectoryStore` | Every scripted training episode is saved as an EventLog. Every evaluation run is saved as a TrajectoryStore record. Both are published in the [episodes dataset](https://huggingface.co/datasets/BMEscanor/against-the-safety-rag-episodes). |
| Training | `create_distill_trainer` | It takes the EventLogs and hands the training rows to a "trainer factory". Ours trains a 4-bit LoRA adapter on one T4 and learns only from the model's own turns. AgentTune's default factory loads the model in bf16 (a T4 has no bf16) and trains on every token. |
| Answer matching | `qa_metrics` | Word-overlap (F1) between an answer and the reference answer. |
| What did not fit | DPO trainer | AgentTune's preference trainer ran out of memory on a T4, so preference training used our own short loop with the same loss. |

Code: `src/ragharness/trainer.py`, `qlora.py`, `retrieval.py`, `tools.py`, `episodes.py`.

### How we used AuditKIT

For every evaluation episode we build two small records and hand them to AuditKIT. An `ak.Sample` carries a `RAGTrace`: the searches, the passages retrieved, the citations and the final answer or refusal. A `RAGCase` holds what the right behaviour was: whether the question was answerable, which passages are enough, the reference answer. `ak.evaluate` then returns the scores below.

First-round scores on the HotpotQA test split, scored before the 13 November corrections:

| AuditKIT metric | The question it answers | Base | Main | Control |
| --- | --- | ---: | ---: | ---: |
| Evidence-set recall | Did the searches retrieve at least one complete set of passages that is enough to answer? | 0.512 | 0.727 | 0.824 |
| Citation support | Do the citations point to passages that exist, are current and support the answer? | 0.302 | 0.771 | 0.941 |
| Abstention | Did it refuse when it should and answer when it should? | 0.721 | 0.794 | 0.714 |
| Tool-call validity | Were the search calls well formed? | 1.000 | 1.000 | 1.000 |
| Redundant tool calls (lower is better) | What share of searches exactly repeat an earlier one? | 3.1% | 2.9% | 8.8% |
| Loop-free episodes | What share of episodes never fell into a repetition loop? | 94.1% | 93.4% | 78.4% |

*Control* is a second adapter trained on the same questions but with no refusal examples. It shows what the refusal examples change.

What AuditKIT showed us:

- **Training taught citing.** Citation support rose from 0.302 to 0.771.
- **Refusal examples teach the model when to stop.** The control adapter looped: only 78.4% of its episodes were loop-free and 8.8% of its searches repeated an earlier one. Main had 93.4% and 2.9%.
- **It reports "unknown" instead of guessing.** When our reference evidence was incomplete, AuditKIT returned no score for those episodes. That exposed an earlier RGB recall figure of 0.960 that covered only 25 of 286 answerable episodes. With complete evidence sets all 286 were scored and recall was 1.000 ([remediation report](reports/2026-11-13-evaluation-remediation.md)).
- **Freshness on dated manuals.** On our fictional Nimbus manuals, which hold old and new editions of the same policy, AuditKIT's freshness score was 0.813 for main and 0.783 for the base model. About one retrieved dated passage in five was an out-of-date edition ([report](reports/nimbus_corrected.md)).

What we did not do: AuditKIT also ships general benchmarks (ARC, GSM8K, MMLU, TruthfulQA, BFCL). None of them tests answering from supplied documents with citations, so we did not run them. The headline numbers below come from our own scorer, which checks in one pass that an answer is correct and that the cited passages contain the required evidence. The full first-round scorecard is in [`reports/scorecard_test_lenient.md`](reports/scorecard_test_lenient.md). Code: `src/ragharness/audit.py`, `scorecard.py`.

## Results

We scored every system on a sealed test of 938 episodes built from three public question sets. SQuAD 2.0 asks about single Wikipedia paragraphs. HotpotQA asks questions that need facts from two articles. RGB asks about news with noisy passages. No training or earlier choice used these questions, and 934 of the 938 could be scored. About 40% of the episodes hide the evidence, so the right move there is to refuse.

A question can end in four ways. The figure shows how often each happens for the submitted system.

![The four outcomes of a question: correct and cited, over-refusal, unsupported answer, correct refusal](docs/figures/outcomes.svg)

Next, all systems on four measures.

![Four panels comparing the base model, the adapter, the adapter with repairs, candF1 and two hosted models](docs/figures/headline.svg)

| System | Macro success | Correct and cited | Answers when evidence was hidden | Correct among answers shown |
| --- | ---: | ---: | ---: | ---: |
| Tiny Aya base | 3.1% | 5.7% | 281 / 375 | 4.4% |
| Main adapter alone | 47.9% | 48.8% | 115 / 375 | 51.9% |
| **Main + both repairs (submitted)** | **58.0%** | 46.3% | **42 / 375** | 67.3% |
| candF1 + both repairs (alternative) | 64.0% | 53.3% | 40 / 375 | 76.4% |
| Refuse every question | 42.0% | 0.0% | 0 / 375 | none |

Scorer v3, [`reports/reference_policies.md`](reports/reference_policies.md). The columns mean:

- **Macro success.** Average success over four groups of questions, counting correct refusals. Higher is better.
- **Correct and cited.** Of the questions whose documents hold the answer, the share answered correctly with the right citation. Higher is better.
- **Answers when evidence was hidden.** Of 375 episodes where the answer was hidden, how many still got an answer. Lower is better. These are annotation-based flags, not human-checked hallucinations.
- **Correct among answers shown.** Of everything the system chose to answer, the share that was correct. Higher is better.

What the table says:

- **The base model is not usable here.** It gives a correct, cited answer in 5.7% of answerable episodes and answers 75% of the episodes whose evidence was hidden.
- **Training plus two repairs fixes the second problem.** The submitted system answers 11% of the hidden-evidence episodes, and two thirds of the answers it shows are correct.
- **The price is coverage.** It answers 41% of all questions and refuses 39% of answerable ones.
- **Read macro success next to the last row.** A correct refusal counts as success, so refusing everything already scores 42.0%.

**Which system do we submit?** The main adapter with both repairs, because we chose it before the test was opened. candF1 scores higher on this test: +6.0 points of macro success, with a 95% interval of +3.0 to +9.2 (the range the true difference very likely lies in). But we picked candF1 with a gate we wrote after seeing its practice-set results, and its safety gain is not established. Its rate of answers on hidden evidence is 0.5 points lower, with an interval of −4.6 to +3.4. Our rule needed the upper end at +2 or below. We publish it as the alternative. A fresh test would settle the question.

**A second, fresh test.** After the first test was used up, we built another from questions nothing had used (679 episodes: 75 SQuAD, 60 HotpotQA and 29 RGB questions), hashed it before any model ran, and opened it once. It did two jobs.

First, it tested one more idea, a *second look*. When the system refuses after it has seen passages, it is asked once to answer anyway, and that answer is shown only if the self-check is at least 90% sure. We wrote the rule first: macro success must rise with an interval above zero, and the rate of answers on hidden evidence may rise by at most 2 points. The second look failed the first half. Macro success rose 0.9 points (interval −0.3 to +2.1), so nothing changed and the submitted system stays as it is. It did add correct, cited answers (+2.3 points, interval +0.7 to +4.1), but it also showed seven wrong answers and two on hidden evidence. Record: [pre-registration](research/2026-11-15-round6-second-look-preregistration.md), [results](research/2026-11-15-round6-second-look-results.md).

Second, it measured the submitted system again. Success by group of questions (the pooled columns above are not comparable across the two tests, because the mix of question types differs):

| | SQuAD single | SQuAD two-part | HotpotQA | RGB | Macro success | Answers on hidden evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Submitted system, fresh test | 43.3% | 36.2% | 58.9% | 78.0% | 54.1% | 29 / 291 (10.0%) |
| Submitted system, first test | 59.1% | 40.4% | 58.5% | 74.1% | 58.0% | 42 / 375 (11.2%) |
| Refuse every question, fresh test | 41.1% | 44.3% | 50.0% | 39.7% | 43.8% | 0 / 291 |

The refusing carried over: 10.0% of hidden-evidence episodes were answered, and macro success is 10.3 points above refusing everything (interval +7.5 to +13.1). The answering did not carry over for single-passage SQuAD questions: 43.3% is close to refusing everything (41.1%), on 75 questions from 25 articles, and we have not looked for the cause. On this test 34.7% of the answers shown are correct. The first test's pooled 67.3% is not directly comparable, because that test held many more RGB episodes (471 of 934, against 141 of 679). We did not run the base model or the hosted models here.

## Try it

| Option | Needs | What you get |
| --- | --- | --- |
| [Trace viewer](https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer) | a browser | recorded episodes, base against fine-tuned model, side by side; no model runs |
| Web demo, `scripts/demo_app.py` | a CUDA GPU (the packaged system ran on a free Kaggle T4) | paste your own passages, ask questions, see the answer or refusal, the citations and the search trace |
| Command line, `scripts/ask.py` | a CUDA GPU | the same system over a JSONL file of passages |

The web demo ships with 24 fictional product-support passages ([`examples/nimbus_home.jsonl`](examples/nimbus_home.jsonl)) and eight example questions: two answerable, two with a newer and an older policy edition in the documents, two that need two passages, two the documents cannot answer ([`examples/nimbus_questions.json`](examples/nimbus_questions.json)). Kaggle is a site that lends free GPUs. To run the demo on a free Kaggle T4 from this repository:

```powershell
python scripts/bundle_for_kaggle.py --adapter main=<folder with the main adapter>
python scripts/kaggle_launch.py dataset --existing
python scripts/kaggle_launch.py run demo --tag try --timeout 5400
```

The kernel first answers the eight example questions, then prints a `DEMO LINK:` line to its log every minute for about 50 minutes; open the kernel's page on kaggle.com to read it. The main adapter is on [Hugging Face](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-main), and the base model is gated, so accept its terms first. More options are in [`kaggle/README.md`](kaggle/README.md).

**A real run.** The eight example questions on a Kaggle T4, with the submitted system (adapter, fail-closed rule and self-check). The full output is in [`docs/examples/demo_smoke_t4.json`](docs/examples/demo_smoke_t4.json).

| Kind | Question | Expected | The system | Result |
| --- | --- | --- | --- | ---: |
| one passage | Seconds to hold the Hub H2 reset pin | 8 | answered 8 | right |
| one passage | Batteries in the Lock L1 | 4 AA alkaline | answered 4 AA alkaline batteries | right |
| two editions | Lock L1 body warranty in months | 24 | answered 24 | right |
| two editions | Cam C3 return window in days | 45 | answered 30, from the 2025 edition | wrong |
| two passages | Does firmware 5.0 local-only mode stop Cam C3 cloud uploads | yes | answered no | wrong |
| two passages | Firmware release that added the protocol Apple Home uses | 4.2 | refused | over-refusal |
| missing | Hub H2 Bluetooth version | refuse | refused | right |
| missing | Lock L1 fingerprint unlocking | refuse | refused | right |

Five of eight match what the documents support, both missing-evidence questions correctly refused. The Cam C3 answer cited both editions and picked the older one, and the self-check let it through (support score 0.90). That is the weakness the Limits section names. An answer took about 10 seconds and a refusal about 20, because a refusal follows three searches. The kernel also served the web page behind a public link, but nobody opened that link in this run, so the page itself has not been clicked through.

![Four recorded episodes: an answer with citations, a refusal on hidden evidence, an answer withheld by the self-check, a fail-closed refusal](docs/figures/trace_examples.svg)

## Against hosted models

As a yardstick we ran two large commercial models on the same sealed test: 925 episodes both sides could answer. The hosted models were told the three-search limit and the answer format in their prompt; the adapter learned them in training. Hosted rows use the fail-closed rule and no self-check.

![Scatter of correct answers against answers when the evidence was hidden for local and hosted systems](docs/figures/tradeoff.svg)

| System | Macro success | Answered | Correct among answered | Wrong or unsupported shown, of all cases |
| --- | ---: | ---: | ---: | ---: |
| Main + both repairs | 58.4% | 41.3% | 67.8% | 13.3% |
| candF1 + both repairs | 64.5% | 42.1% | 76.6% | 9.8% |
| GPT-6 Luna | 70.7% | 70.5% | 66.1% | 23.9% |
| DeepSeek V4 Flash | 68.8% | 64.2% | 67.7% | 20.8% |

GPT-6 Luna is 6.2 points above candF1 on macro (interval +1.1 to +11.1). DeepSeek V4 Flash is 4.4 above (interval −0.3 to +9.1), which is not clearly different. Both hosted models answer far more often, and show a wrong or unsupported answer about twice as often as candF1. [Full comparison](reports/fair_rerun.md).

**Where the hosted models are better.** Of the questions whose documents hold the answer, GPT-6 Luna gives a correct, cited answer 77.9% of the time and DeepSeek V4 Flash 72.7%. Our submitted system does so 46.8% of the time (candF1: 53.9%). Our one advantage is fewer answers when the evidence was hidden: 11% against 21% (DeepSeek) and 28% (GPT-6 Luna). Part of that comes from refusing more often, not from being smarter. On HotpotQA, the question type we trained on, the adapters are level or ahead (main 59.5%, candF1 65.8%, GPT-6 Luna 55.7%, DeepSeek 57.0%). On SQuAD single-passage questions the hosted models are clearly ahead (GPT-6 Luna 86.6%, DeepSeek 84.3%, main 59.7%, candF1 62.0%), and on two-part questions GPT-6 Luna is ahead (62.3% against 40.3% and 54.1%).

**On benchmarks the adapters never trained on.** These are older suites. Every system has the fail-closed rule applied; none has the self-check. The hosted runs predate the fix that told them the answer format, so they understate the hosted models (on the sealed test, GPT-6 Luna's macro success rose from 60.8% to 70.7% once told).

| Suite (cases) | Refuse every question | Main + fail-closed | GPT-6 Luna + fail-closed | DeepSeek V4 Flash + fail-closed |
| --- | ---: | ---: | ---: | ---: |
| 2WikiMultihopQA (165) | 47.6% | 57.6% | 58.2% | 64.8% |
| MuSiQue (164) | 47.6% | 56.1% | 57.9% | 56.1% |
| SQuAD 2.0 (202) | 63.4% | 63.9% | 55.0% | 71.3% |
| RGB (450) | 36.9% | 62.2% | 62.7% | 78.4% |
| RGB integration (142) | 7.0% | 4.9% | 23.2% | 28.9% |

The adapter is within two points of GPT-6 Luna on three of the five suites and ahead on SQuAD 2.0, but it is behind DeepSeek V4 Flash on four, and far behind both on RGB integration (two-part answers). On SQuAD 2.0 it scores the same as refusing every question (63.9% against 63.4%). So the honest summary is that a small adapter gets close to large hosted models on this task, not that it beats them. [Source: `reports/external_benchmarks_corrected.md`](reports/external_benchmarks_corrected.md).

**Cost.** For 10,000 episodes: $6.30 for the adapter's busy T4 time (perfect utilisation, $0.526 per hour, nothing else counted), $2.67 for GPT-6 Luna and $6.09 for DeepSeek, from the recorded token mix and published prices. The adapter is not cheaper. Its case is control: it runs on one GPU you own and shows fewer unsupported answers. See [`reports/azure_roi.md`](reports/azure_roi.md) and [`reports/deepseek_roi.md`](reports/deepseek_roi.md).

## Other languages: Hindi and Spanish

Tiny Aya reads 70 languages, but we trained the adapter on English only. On 15 November 2026 we ran the submitted system once on the same questions in Hindi, English and Spanish (XQuAD, Wikipedia questions translated by professionals). Nothing was retrained or tuned for this check, and the plan was committed before any data was built ([plan](research/2026-11-15-multilingual-plan.md), [results](research/2026-11-15-multilingual-results.md)).

![Hindi, English and Spanish side by side](docs/figures/xquad_results.svg)

| XQuAD, same questions in three languages | Hindi (285 questions) | English (295 questions) | Spanish (285 questions) |
| --- | ---: | ---: | ---: |
| Correct and cited, when the documents hold the answer | 2.1% | 2.7% | 4.6% |
| Answers although the evidence was hidden | 3 / 285 | 0 / 295 | 1 / 285 |
| Same, Tiny Aya base model | 188 / 285 | 205 / 295 | 237 / 285 |
| Success (refusing every question scores 50.0%) | 50.5% | 51.4% | 52.1% |
| Adapter alone: turns that ended invalid | 73.0% | 23.6% | 6.5% |

In plain words:

- **The refusing carries over.** When the evidence is hidden, the system answers 3 of 285 Hindi, 0 of 295 English and 1 of 285 Spanish questions. The base model answers two thirds or more of them.
- **The answering does not, in any of the three languages.** With the evidence in reach, the system gives a correct, cited answer to only 2% to 5% of questions, so its success rate sits at what refusing every question would score (50.0%). English is just as low as Hindi, because these are single-passage questions, a weak spot we already knew. This check therefore cannot tell a language problem apart from that one. It shows that Hindi and Spanish are not measurably worse than English.
- **Hindi breaks the search more often.** Before the repairs, 73% of the adapter's Hindi turns ended invalid, against 24% in English and 6.5% in Spanish. Nearly all of them repeat the previous search word for word. The repairs turn these into refusals. We did not test why.

One sample of 300 translated questions per language, so treat it as a first look. Hosted models were not run on Hindi.

## What we tried, and what we kept

![Timeline of rounds 1 to 6 with the outcome of each](docs/figures/rounds.svg)

- **Refusal training is what makes the model stop.** A control adapter trained without refusal examples answered more but never refused; it kept searching until the budget ran out.
- **Cheap fixes beat more training.** The two repairs moved macro success from 47.9% to 58.0% with no retraining. Every training change after round 1 missed its own pre-registered release rule.
- **Preference training removed refusals.** Preference training (DPO) shows the model pairs of a better and a worse answer and nudges it toward the better one. Our pairs almost never chose a refusal, so the checkpoints stopped refusing. Adding refusal pairs restored it from main but passed no gate. Some of those training prompts also exposed held-out evidence; main and candF1 predate them. See the [remediation report](reports/2026-11-13-evaluation-remediation.md).
- **A trained checker did not beat the plain one.** Two adapters trained to check answers (V2m, V2b) never reached the hidden-evidence limit we set, at any threshold ([report](reports/round6_selection.md)). A second look at refusals, checked by the plain self-check, also failed its rule on the fresh Round 6 test, which we opened once ([results](research/2026-11-15-round6-second-look-results.md)).

The full record, with every pre-registered rule and each result, is in [`docs/HISTORY.md`](docs/HISTORY.md).

## Where things are

| Folder | What is in it |
| --- | --- |
| `src/ragharness/` | Our code: the search tool, episode builder, training hook, rollout loop, scorer, the AuditKIT adapter and the self-check (`system.py`). |
| `scripts/` | Command-line tools: build data, launch Kaggle jobs, rebuild reports, `ask.py`, `demo_app.py`. |
| `kaggle/` | The script that runs on Kaggle GPUs, and its guide. |
| `data/` | Question sets, scripted episodes, and manifests with file hashes. Large files live in the Hugging Face datasets. |
| `reports/` | Every number on this page, as JSON and Markdown. |
| `docs/` | The figures, the saved demo run, and [`HISTORY.md`](docs/HISTORY.md), the full record of the six rounds. |
| `examples/` | The fictional Nimbus manuals and the eight demo questions. |
| `article/` | The write-up and the form abstract. |
| `tests/` | Automated tests, including checks that the numbers on this page match `reports/`. |
| `research/` | Planning notes and pre-registrations. |

## Reproduce

Python 3.12. All caches stay inside the repository folder.

```powershell
py -3.12 -m venv .venv
. .\scripts\env.ps1
python -m pip install --no-deps -r requirements-cpu.lock.txt
python -m pip install --no-deps -e .
python -m pytest tests -q                  # some tests skip when large data is absent
python scripts/remediate_audit.py          # rebuilds the corrected reports from the saved transcripts
```

Training and evaluation ran on Kaggle T4s: see [`kaggle/README.md`](kaggle/README.md). Use the adapter directly with `scripts/ask.py`; load it with the base model's [pinned revision](https://huggingface.co/CohereLabs/tiny-aya-global).

## Limits

- English, short factual questions. Tiny Aya is a multilingual model, but we trained the adapter in English only. A first Hindi and Spanish check (see [Other languages](#other-languages-hindi-and-spanish)) shows the refusing carries over and the answering does not: 2% to 5% of questions got a correct, cited answer, and Hindi searches loop far more often. Two-part answers and single-passage questions are weak spots; with two documents and dated editions the adapter sometimes cites the older one.
- The first sealed test was opened before the scoring corrections, and candF1 was tested on it after its gate was written. The fresh Round 6 test was then opened once, for the second look, so further model selection needs yet another fresh test. On that test the submitted system answered single-passage SQuAD questions about as well as refusing everything would score (43.3% against 41.1%).
- Hidden-evidence answers are annotation-based flags. An unannotated alternative source could make a flagged answer correct, and no one has hand-checked them.
- The self-check withholds some correct answers. The submitted system refuses 39% of answerable questions.
- On benchmarks the adapters never trained on, the adapter is level with GPT-6 Luna and behind DeepSeek V4 Flash, and on SQuAD 2.0 it scores the same as refusing every question. Large hosted models are more accurate when the documents hold the answer. See [Against hosted models](#against-hosted-models).
- There is no hosted live endpoint: the web demo needs a GPU you start yourself. Its GPU run answered the eight example questions as shown above (five right, two wrong, one over-refusal); eight questions say little about accuracy, which the sealed test measures.

## Artifacts and licenses

| | |
| --- | --- |
| Adapters | [main](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-main) (submitted), [candF1](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-cand-f1) (alternative), [control](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-control), [candA](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-cand-a), [patch](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-patch) |
| Data | [episodes and training rows](https://huggingface.co/datasets/BMEscanor/against-the-safety-rag-episodes), [benchmark traces](https://huggingface.co/datasets/BMEscanor/against-the-safety-benchmark-traces), [Nimbus Home suite](https://huggingface.co/datasets/BMEscanor/nimbus-home-support-suite) |
| Code | MIT |
| Adapters | CC BY-NC 4.0, inherited from Tiny Aya (gated; accept its terms) |
| HotpotQA- and SQuAD-derived data | CC BY-SA 4.0 |
| RGB-derived data | CC BY-NC-SA 4.0, noncommercial |
| Nimbus Home suite | CC BY 4.0 (fictional, written by us) |
| Benchmark traces | as the source: 2WikiMultihopQA Apache-2.0, MuSiQue CC BY 4.0, SQuAD 2.0 CC BY-SA 4.0, RGB CC BY-NC-SA 4.0 |
| AgentTune (LSAL v1.1), AuditKIT (LSAL v1.2) | source-available, used as dependencies, not redistributed |

Written by [BMEscanor](https://huggingface.co/BMEscanor) and [abd420](https://huggingface.co/abd420). A friend provided Kaggle compute with permission.
