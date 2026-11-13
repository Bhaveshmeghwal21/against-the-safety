# Earlier external suites: corrected replay

Scorer v3, fixed evidence eligibility, official SQuAD answer offsets and duplicate MuSiQue questions removed.
The French and Indian War question is adjudicated as answerable in the legacy corpus-wide condition (1754); future natural SQuAD episodes restrict evidence to their source paragraph.

| Suite | System | Shared cases | Success | Flagged / hidden |
| --- | --- | ---: | ---: | ---: |
| 2wiki | main | 165 | 0.5212 | 3/79 |
| 2wiki | candA | 165 | 0.5515 | 5/79 |
| 2wiki | azure_native+FB | 165 | 0.5818 | 1/79 |
| 2wiki | azure_deepseek_native+FB | 165 | 0.6485 | 2/79 |
| musique | main | 164 | 0.5000 | 1/77 |
| musique | candA | 164 | 0.5671 | 2/77 |
| musique | azure_native+FB | 164 | 0.5793 | 2/77 |
| musique | azure_deepseek_native+FB | 164 | 0.5610 | 1/77 |
| squad2 | main | 202 | 0.4802 | 4/128 |
| squad2 | candA | 202 | 0.5545 | 6/128 |
| squad2 | azure_native+FB | 202 | 0.5495 | 52/128 |
| squad2 | azure_deepseek_native+FB | 202 | 0.7129 | 39/128 |
| rgb | main | 450 | 0.5844 | 86/164 |
| rgb | candA | 450 | 0.6222 | 108/164 |
| rgb | azure_native+FB | 450 | 0.6267 | 64/164 |
| rgb | azure_deepseek_native+FB | 450 | 0.7844 | 36/164 |
| rgb_int | main | 142 | 0.0282 | 4/10 |
| rgb_int | candA | 142 | 0.0423 | 4/10 |
| rgb_int | azure_native+FB | 142 | 0.2324 | 2/10 |
| rgb_int | azure_deepseek_native+FB | 142 | 0.2887 | 1/10 |

No final Tiny Aya verifier was saved on these older suites. Flagged counts remain annotation-based; they are not human-adjudicated hallucination rates.
