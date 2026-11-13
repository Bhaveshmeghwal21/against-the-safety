# Round 5: corrected local replay

Corrected replay dated 2026-11-13; 934 shared episodes. Saved prompts, outputs, fallback and verifier are unchanged.

Eligibility is fixed from annotated evidence and masks before scoring. Unsupported counts are annotation-based flags; unannotated alternative entailment still needs adjudication. The denominator differs from scorer v2.

| System | Macro success | Supported correct | Flagged answers / hidden |
| --- | ---: | ---: | ---: |
| main+FB+V | 0.5802 | 0.4633 | 42/375 |
| candF1+FB+V | 0.6401 | 0.5331 | 40/375 |

Paired differences use SQuAD article clusters and HotpotQA/RGB question clusters:

- candF1+FB+V/main+FB+V: +5.99 percentage points [+3.01, +9.18].

These are exploratory corrections on an opened holdout. Main remains submitted; a new model-selection claim needs a fresh holdout.
