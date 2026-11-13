# Accuracy and coverage (dev, headline episodes: 159 = 86 answerable + 73 with the evidence masked)

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


Each episode lands in exactly one row. Counts, then per 100 episodes.

| outcome | base | main | control |
| --- | ---: | ---: | ---: |
| supported answer (good) | 2 (1.3) | 37 (23.3) | 44 (27.7) |
| bad answer: wrong, unsupported, or answered with the evidence missing | 141 (88.7) | 28 (17.6) | 45 (28.3) |
| correct abstain (good) | 0 (0.0) | 70 (44.0) | 0 (0.0) |
| false abstain | 0 (0.0) | 20 (12.6) | 0 (0.0) |
| no valid final turn | 16 (10.1) | 4 (2.5) | 70 (44.0) |
| answers delivered | 143 (89.9) | 65 (40.9) | 89 (56.0) |
| precision of delivered answers (supported / delivered) | 0.014 | 0.569 | 0.494 |
| evidence missing: no answer delivered (explicit abstain, or the episode ended without a valid final turn) | 9/73 (12.3%) | 71/73 (97.3%) | 61/73 (83.6%) |
| evidence missing: explicit abstain | 0/73 (0.0%) | 70/73 (95.9%) | 0/73 (0.0%) |
