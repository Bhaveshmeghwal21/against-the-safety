Document assistants need to answer from retrieved evidence and decline when it is missing.

We fine-tuned Tiny Aya Global (3.35B) with QLoRA through AgentTune to make up to three searches, then return a cited answer or a refusal. A fail-closed rule converts invalid endings to refusals. A self-check uses the same model to assess each answer against its citations.

On the opened round-5 evaluation, correct and cited answers on answerable episodes rose from 5.7% for the base model to 46.3% for main with both repairs. Answers given when evidence was hidden fell from 281/375 to 42/375. Of the answers shown by the repaired system, 67.3% counted as correct. Its macro success was 58.0%, compared with 42.0% for a policy that refuses every question.

Preference training did not produce a checkpoint that met the release gate. In saved native-tool runs, GPT-6 Luna and DeepSeek V4 Flash scored 60.4% and 64.9% macro success on their own available cases. Their prompts omitted the search budget and required answer format, and invalid endings were converted to refusals. These comparisons are provisional.

The test is English and uses short factual questions. Evidence labels are annotation-based and need human review. Corrected scores were calculated after the test was opened, so further model selection needs a fresh holdout.
