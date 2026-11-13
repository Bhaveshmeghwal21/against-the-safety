# DeepSeek comparison: corrected evidence scorer

Corrected replay dated 2026-11-13; 925 shared episodes. Saved prompts, outputs, fallback and verifier are unchanged.

Eligibility is fixed from annotated evidence and masks before scoring. Unsupported counts are annotation-based flags; unannotated alternative entailment still needs adjudication. The denominator differs from scorer v2.

| System | Macro success | Supported correct | Flagged answers / hidden |
| --- | ---: | ---: | ---: |
| main+FB+V | 0.5816 | 0.4675 | 42/371 |
| candF1+FB+V | 0.6424 | 0.5379 | 40/371 |
| azure_native+FB | 0.6073 | 0.5560 | 90/371 |
| azure_deepseek_native+FB | 0.6487 | 0.6354 | 59/371 |

Paired differences use SQuAD article clusters and HotpotQA/RGB question clusters:

- azure_deepseek_native+FB/candF1+FB+V: +0.63 percentage points [-3.85, +4.92].
- azure_deepseek_native+FB/azure_native+FB: +4.14 percentage points [+0.95, +7.57].

These are exploratory corrections on an opened holdout. Main remains submitted; a new model-selection claim needs a fresh holdout.
