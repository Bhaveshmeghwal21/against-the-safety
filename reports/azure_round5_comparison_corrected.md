# GPT-6 Luna comparison: corrected evidence scorer

Corrected replay dated 2026-11-13; 931 shared episodes. Saved prompts, outputs, fallback and verifier are unchanged.

Eligibility is fixed from annotated evidence and masks before scoring. Unsupported counts are annotation-based flags; unannotated alternative entailment still needs adjudication. The denominator differs from scorer v2.

| System | Macro success | Supported correct | Flagged answers / hidden |
| --- | ---: | ---: | ---: |
| main+FB+V | 0.5803 | 0.4650 | 42/374 |
| candF1+FB+V | 0.6404 | 0.5350 | 40/374 |
| azure_native+FB | 0.6043 | 0.5530 | 91/374 |

Paired differences use SQuAD article clusters and HotpotQA/RGB question clusters:

- azure_native+FB/main+FB+V: +2.39 percentage points [-2.02, +6.59].
- azure_native+FB/candF1+FB+V: -3.62 percentage points [-8.37, +0.96].

These are exploratory corrections on an opened holdout. Main remains submitted; a new model-selection claim needs a fresh holdout.
