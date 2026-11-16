# Words used in this write-up

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
