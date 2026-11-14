Document assistants need to answer from retrieved evidence and decline when it is missing.

We fine-tuned Tiny Aya Global (3.35B) with QLoRA through AgentTune to make up to three searches, then return a cited answer or a refusal. A fail-closed rule converts invalid endings to refusals. A self-check uses the same model to assess each answer against its citations.

On the opened round-5 evaluation, correct and cited answers on answerable episodes rose from 5.7% for the base model to 46.3% for main with both repairs. Answers given when evidence was hidden fell from 281/375 to 42/375. Of the answers shown by the repaired system, 67.3% counted as correct. Its macro success was 58.0%, compared with 42.0% for a policy that refuses every question.

Preference training did not produce a checkpoint that met the release gate. On 925 shared, previously opened episodes, macro success was 58.4% for main, 64.5% for candF1, 70.7% for GPT-6 Luna and 68.8% for DeepSeek V4 Flash. Correct among answers shown was 67.8%, 76.6%, 66.1% and 67.7%, respectively. Answer coverage and checker settings differed; hosted models were told the search limit and answer format.

The test is English and uses short factual questions. Evidence labels are annotation-based and need human review. Corrected scores were calculated after the test was opened, so further model selection needs a fresh holdout.
