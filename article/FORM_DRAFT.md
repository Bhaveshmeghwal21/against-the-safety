# Submission form material: Against-the-safety

Form: https://tally.so/r/o9E0Ve. Due 15 November 2026, 11:59 PM AoE (Monday 16 November, 5:29 PM IST). The latest submission counts.

| Field | Value |
| --- | --- |
| Track and problem statement | Track 3, agentic fine-tuning; PS 4, fine-tuning for retrieval-augmented generation |
| Lexsi repositories used | AgentTune (search tool and keyword index, EventLog and TrajectoryStore recordings, `create_distill_trainer` with our QLoRA trainer, `qa_metrics`), AuditKIT (each episode scored as an `ak.Sample` with a `RAGTrace`: evidence-set recall, citation support, abstention, tool-call validity, redundant calls, loop detection, freshness). Details: README section "The two Lexsi tools" |
| Model collection | https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc |
| Dataset collection | https://huggingface.co/collections/BMEscanor/against-the-safety-rag-episodes-traces-and-demo-6aec1fdb994e3b036c6c958d |
| Write-up | GitHub README (Hugging Face Articles need a Pro account, and the rules accept a README): https://github.com/Bhaveshmeghwal21/against-the-safety#readme; the long version is `article/ARTICLE.md` in the same repository |
| GitHub repository | https://github.com/Bhaveshmeghwal21/against-the-safety (must be public before submitting) |
| Demo | Trace viewer https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer (recorded episodes); live web demo `scripts/demo_app.py` on a GPU |
| Registered email, Discord username | to fill in by the team |

## Abstract for the form (under 500 words)

A document assistant is only useful if it answers from its documents and says so when they don't hold the answer. This setup is called retrieval-augmented generation (RAG): the assistant searches the documents, then writes an answer from what it found. We fine-tuned Tiny Aya Global (3.35B), a small open model, with QLoRA, which trains a small add-on file (an adapter) instead of the whole model. Training ran through AgentTune, Lexsi's library for training agents. The model searches a document set up to three times, then answers with citations or refuses. Two repairs written as ordinary code sit on top and need no training: a fail-closed rule that turns any invalid ending into a refusal, and a self-check in which the same model reads the cited passages and withholds an answer they do not support. AuditKIT, Lexsi's scoring library, scored our first-round runs and checked evidence coverage; the headline numbers below come from our own scorer.

We scored every system with fixed code, not an AI judge, on a sealed test (questions locked away before training) of 938 SQuAD, HotpotQA and RGB episodes, about 40% of which hide the evidence. Against the base model, the submitted system raises correct, cited answers on answerable episodes from 5.7% to 46.3% and cuts answers given when the evidence was hidden from 281 of 375 to 42 of 375. Of the answers it shows, 67.3% are correct. Macro success, which counts a correct refusal as success, is 58.0%. A policy that refuses every question scores 42.0%, so we report four measures and not macro success alone. In a format-aware comparison on 925 shared episodes, GPT-6 Luna and DeepSeek V4 Flash score 70.7% and 68.8% macro success, but show a wrong or unsupported answer in 23.9% and 20.8% of cases, against 13.3% for our system and 9.8% for its alternative checkpoint candF1 (64.5% macro success).

Every later training change failed its own pre-registered release rule: a patch for single-passage questions, a matched-evidence curriculum, preference training (its checkpoints stopped abstaining) and trained answer checkers. A second look at refusals, checked by the plain self-check, also failed its pre-registered rule on a fresh 679-episode test that we opened once (macro success +0.9 points, interval −0.3 to +2.1). The same test measured the submitted system again: 54.1% macro success against 43.8% for refusing everything, 10.0% of hidden-evidence episodes answered, and single-passage SQuAD questions close to the refuse-everything score. The tests cover English short factual questions. Further model selection needs another fresh test, and hidden-evidence answers are annotation-based flags, not human-checked hallucinations.
