# Multilingual check, Hindi first, then Spanish: plan and pre-registration draft

Date: 15 November 2026, written at about 02:00 IST, before any data is built or any model is run on these questions.
Status: approved by the user on 15 November 2026 at about 02:05 IST ("execute this whole plan"), which covers all four approvals in section 13. It is the pre-registration as of the commit that contains this line; that commit comes before any code change and before the Kaggle run.
Budget: 8 hours from approval. Our own stop for all work is Monday 16 November 12:00 IST.

## 1. The question

Does the submitted system (main adapter + fail-closed + self-check V) keep its behaviour in Hindi and Spanish? Specifically:
does it still give correct, cited answers when the documents hold the answer, and still refuse when they don't?
Tiny Aya Global lists both languages as supported. The adapter was trained only on English HotpotQA.

This check is descriptive. Whatever it shows, it cannot change the submitted system, and we publish it either way.

## 2. Data

- **Source:** XQuAD validation (google/xquad, CC BY-SA 4.0). It has the same 1,190 questions over 240 paragraphs in each language. We use `xquad.hi`, `xquad.en` and `xquad.es`.
- **Same questions in every language.** We keep a question only if all three conditions hold in all three languages:
  - its official answer offset is valid;
  - after splitting, its answer sentence fits inside one displayed passage;
  - its id does not appear in our earlier SQuAD suites (`squad2`, `r3t_squad2`). SQuAD 2.0 dev reuses SQuAD 1.1 dev questions, so this guards against overlap. The main adapter never trained on SQuAD. The exclusion avoids reusing questions we have already scored.
- **Sample:** 300 of the remaining questions, drawn with seed 47 and spread across articles. If fewer than 300 remain, we use all of them.
- **Article labels** come from SQuAD v1.1 dev, joined by question id. They are used only to cluster the confidence intervals and are never shown to the model.
- **Passages:**
  - Each paragraph is split at sentence ends into passages of at most 800 characters. This is the display limit the adapter was trained with.
  - Hindi text is longer in characters: the first XQuAD paragraph is 2,835 characters in Hindi, 1,482 in Spanish and 1,168 in English. Hindi therefore gets more, shorter passages. We report the passage count per language.
  - Sentence ends include the Hindi danda (। and ॥) and the Spanish ¿ and ¡ openers.
- **Titles:** the passage's own first eight words, in its own language. No English titles appear in the Hindi or Spanish documents.
- **Index:** one SQLite FTS5 index per language, with tokenizer `unicode61 categories 'L* N* Co M*'`. This keeps Hindi words whole instead of splitting them at vowel signs; we checked locally that the default tokenizer splits them. All three languages use the same tokenizer.

## 3. Episodes

Two episodes per question, in each language:

1. **Evidence available.** The model searches the whole corpus of its language. The right ending is a correct answer that cites the answer passage.
2. **Evidence hidden.** The answer passage is masked from search. The right ending is a refusal. This episode is excluded if any other passage still shows the answer, using the scorer-v3 eligibility rule.

The protocol is unchanged: three searches, three passages per search, 800 characters each, and the same English system prompt.

## 4. Systems

- **base:** Tiny Aya Global, no adapter.
- **main:** the main adapter alone.
- **main+FB+V:** the submitted system. FB is applied offline to main's transcripts. V is the main adapter with wording A and threshold 0.5, using the English checker prompt unchanged. This is how the round-5 numbers were produced.

## 5. Scoring rules, fixed now

- **Correct answer:** the scorer-v3 rule, with these changes for the new suites only:
  - text is normalized to Unicode NFC, lowercased, all Unicode punctuation removed (including । ॥ ¿ ¡ and curly quotes), and whitespace collapsed;
  - articles are removed: a/an/the for English, el/la/los/las/un/una/unos/unas for Spanish, none for Hindi.
- **Same normalization elsewhere:** the self-check's "answer appears in the cited passage" test uses the same normalization in both places it is coded (`checker.contained` and `build_verifier_cases.contained`). Without this, V would block correct Hindi answers that end a sentence.
- **Existing suites are untouched.** They keep the old code path byte for byte. Check: `remediate_audit.py` must still print `sealed hashes 11 / 11`, and the stored historical scores must be unchanged.
- **Wrong-language answers count as wrong.** An answer to a Hindi question is scored against the Hindi reference. We also report the share of Hindi-question answers that contain no Devanagari letters.

## 6. What we report, whatever it shows

For each language and system:

- success: correct, cited answers plus correct refusals, over all episodes;
- correct and cited, of episodes with evidence available;
- answers when the evidence was hidden (count of n);
- correct among answers shown;
- refusals when evidence was available;
- invalid turns;
- the refuse-every-question baseline.

AuditKIT metrics for base and main (raw systems): evidence-set recall, citation support, abstention, tool-call validity, redundant tool calls and loop-free episodes. If time allows, the same metrics for main+FB+V through a replay of the transcripts.

Comparisons, paired by question, with 95% article-cluster bootstrap intervals (3,000 draws, seed 47):

- main+FB+V, Hindi minus English: correct and cited, and the hidden-evidence answer rate;
- main+FB+V minus base, within Hindi: the same two measures;
- the same for Spanish, if it runs.

One run. A rerun is allowed only after an infrastructure failure (crash, quota, kernel lost), and any deviation is labelled in the results.

## 7. Code changes

| File | Change |
| --- | --- |
| `src/ragharness/benchmarks.py` | `load_xquad`, a language-aware sentence splitter (the default English regex unchanged), passage splitting at 800 characters |
| `src/ragharness/data.py` | optional `lang` field on `Question` (default `en`, so old files still load) |
| `src/ragharness/corpus.py` | optional FTS5 tokenizer for `build_index`, recorded in the manifest |
| `src/ragharness/metrics.py`, `rescore.py`, `checker.py`, `scripts/build_verifier_cases.py` | language-gated normalization; the English path is unchanged |
| `scripts/prepare_xquad.py` | builds `data/bench/xq_hi`, `xq_en`, `xq_es` with the same question ids, plus the overlap and selection report |
| `kaggle/bootstrap.py` | `--no-second-look` option for the `r6-system` stage, so it runs answers + V without the second-look step that Round 6 turned off; a Hindi index probe at the start |
| `scripts/report_xquad.py` | v3 scoring, FB and V applied offline, the statistics, intervals and AuditKIT tables |
| `tests/test_xquad.py` | danda splitting, Devanagari answer offsets, the tokenizer probe, Hindi and Spanish normalization in F1 and in `contained`, the English path unchanged |

## 8. Checks before any GPU time

1. **English unchanged.** The full test suite passes, and `remediate_audit.py` prints `sealed hashes 11 / 11` with unchanged stored scores. Nothing proceeds until this holds.
2. **Search reaches Hindi evidence.** The scripted oracle (the perfect policy used to build training data) must find the answer passage within three searches in at least 80% of Hindi evidence-available episodes. If not, I fix the search, capped at 45 minutes. After the cap we run anyway and report the ceiling next to the results.
3. **Prompt length.** The worst-case Hindi prompt (nine passages shown) is counted with the Tiny Aya tokenizer and must fit the 4,096-token input limit. If some episodes don't fit, we report how many were truncated.

## 9. Kaggle run

- Account 1 (`bhaveshmeghwal`), one kernel, both T4s, stage `r6-system` with `--checkers V --no-second-look`.
- Jobs are weighted so Hindi runs first on both GPUs, English second and Spanish last: `main:bench_xq_hi:40,base:bench_xq_hi:40,main:bench_xq_en:30,base:bench_xq_en:30,main:bench_xq_es:20,base:bench_xq_es:20`.
- **Estimate:** about 3,600 episodes. The round-5 logs show 2.4 to 5 seconds per episode per GPU, so at about 3.5 seconds this is roughly 1.75 hours on two GPUs. With V and setup, about 2 to 2.5 hours; Hindi may run slower because its prompts are longer. Timeout 5 hours.
- The Round 6 sealed test is in the bundle but no job reads it. It stays unopened.

## 10. Timeline (T = approval)

| When | Work |
| --- | --- |
| T+0:00 to 0:20 | Finalize and commit this pre-registration |
| T+0:20 to 2:00 | Code changes and tests |
| T+2:00 to 2:40 | Checks 1 to 3 on CPU; build and upload the bundle; launch Kaggle |
| T+2:40 to 5:15 | Kaggle run. Meanwhile: the report script, tested on the oracle transcripts; the optional hosted Hindi run |
| T+5:15 to 6:15 | Download, score, intervals, AuditKIT, one figure |
| T+6:15 to 7:30 | README, Article, HISTORY and Hugging Face cards updated with the results, good or bad; tests; push; republish cards |
| T+7:30 to 8:00 | Buffer |

**Fallbacks:**

- If anything slips, Spanish is dropped first. It runs last anyway.
- If the Kaggle run fails, one retry.
- If Hindi has not finished by T+6:30, the documents say plainly that Hindi was not tested, and nothing else changes.

## 11. Optional: hosted models in Hindi

GPT-6 Luna and DeepSeek V4 Flash on `xq_hi` only, with the existing format-aware prompt and fail-closed, run from this machine while Kaggle runs. Rough cost: about $1 at published prices, based on the last hosted rerun. It needs a separate yes, because it spends money.

## 12. Out of scope

- **Hindi training.** It needs Hindi training episodes, a new gate and a fresh sealed test, roughly another day, and it would put the submitted system at risk. If Hindi turns out weak, the Article lists it as the next step.
- **Cross-lingual use** (a Hindi question over English documents) and languages beyond Spanish.

## 13. Approvals needed

1. This plan. On approval, it is committed as the pre-registration and the work starts.
2. The Kaggle run on account 1, about 2.5 GPU-hours.
3. Optional: the hosted Hindi comparison, about $1.
4. Republishing the Hugging Face cards at the end with the results.
