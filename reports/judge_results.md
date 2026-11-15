# Judge-based RAG metrics, first sealed test (300-episode sample)

Judge: Azure `gpt-6-luna-2026-09-22`, AuditKIT prompts (version v2), one judge, one sample. Answer metrics average over answered episodes where the judge could extract a claim. Context metrics average over the episodes where they apply. 95% bootstrap intervals over the sampled episodes. The pre-registration is [here](../research/2026-11-16-judge-latency-preregistration.md).

## Answer metrics, all episodes

| System | Answers given (of those episodes) | Faithfulness (higher is better) | Hallucination (lower is better) | Response groundedness | Answer relevancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 231 of 300 | 0.625 (0.566 to 0.685), n=224 | 0.375 (0.315 to 0.435), n=224 | 0.641 (0.580 to 0.700), n=228 | 0.912 (0.882 to 0.938), n=229 |
| main | 166 of 300 | 0.722 (0.653 to 0.790), n=162 | 0.284 (0.216 to 0.353), n=162 | 0.982 (0.959 to 1.000), n=166 | 0.976 (0.950 to 0.994), n=165 |
| main+FB+V | 125 of 300 | 0.808 (0.736 to 0.872), n=125 | 0.192 (0.124 to 0.261), n=125 | 0.984 (0.958 to 1.000), n=125 | 1.000 (1.000 to 1.000), n=125 |
| candF1 | 182 of 300 | 0.750 (0.688 to 0.808), n=181 | 0.250 (0.191 to 0.310), n=181 | 0.940 (0.904 to 0.972), n=182 | 0.992 (0.980 to 1.000), n=182 |
| candF1+FB+V | 128 of 300 | 0.885 (0.830 to 0.935), n=128 | 0.115 (0.066 to 0.171), n=128 | 0.977 (0.947 to 1.000), n=128 | 0.989 (0.972 to 1.000), n=128 |

## Answer metrics, answerable episodes

| System | Answers given (of those episodes) | Faithfulness (higher is better) | Hallucination (lower is better) | Response groundedness | Answer relevancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 140 of 179 | 0.843 (0.785 to 0.898), n=138 | 0.157 (0.104 to 0.216), n=138 | 0.806 (0.743 to 0.863), n=138 | 0.944 (0.919 to 0.966), n=139 |
| main | 131 of 179 | 0.853 (0.788 to 0.909), n=129 | 0.155 (0.096 to 0.218), n=129 | 0.977 (0.949 to 1.000), n=131 | 0.992 (0.975 to 1.000), n=130 |
| main+FB+V | 113 of 179 | 0.876 (0.814 to 0.934), n=113 | 0.124 (0.067 to 0.186), n=113 | 0.982 (0.954 to 1.000), n=113 | 1.000 (1.000 to 1.000), n=113 |
| candF1 | 150 of 179 | 0.875 (0.824 to 0.923), n=150 | 0.125 (0.078 to 0.175), n=150 | 0.953 (0.917 to 0.981), n=150 | 0.991 (0.976 to 1.000), n=150 |
| candF1+FB+V | 120 of 179 | 0.927 (0.880 to 0.968), n=120 | 0.073 (0.033 to 0.120), n=120 | 0.975 (0.943 to 1.000), n=120 | 0.989 (0.970 to 1.000), n=120 |

## Answer metrics, hidden evidence episodes

| System | Answers given (of those episodes) | Faithfulness (higher is better) | Hallucination (lower is better) | Response groundedness | Answer relevancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 91 of 121 | 0.275 (0.193 to 0.360), n=86 | 0.725 (0.641 to 0.806), n=86 | 0.387 (0.298 to 0.483), n=90 | 0.862 (0.796 to 0.921), n=90 |
| main | 35 of 121 | 0.212 (0.077 to 0.360), n=33 | 0.788 (0.636 to 0.919), n=33 | 1.000 (1.000 to 1.000), n=35 | 0.914 (0.818 to 1.000), n=35 |
| main+FB+V | 12 of 121 | 0.167 (0.000 to 0.429), n=12 | 0.833 (0.583 to 1.000), n=12 | 1.000 (1.000 to 1.000), n=12 | 1.000 (1.000 to 1.000), n=12 |
| candF1 | 32 of 121 | 0.145 (0.034 to 0.286), n=31 | 0.855 (0.719 to 0.966), n=31 | 0.875 (0.750 to 0.971), n=32 | 1.000 (1.000 to 1.000), n=32 |
| candF1+FB+V | 8 of 121 | 0.250 (0.000 to 0.600), n=8 | 0.750 (0.400 to 1.000), n=8 | 1.000 (1.000 to 1.000), n=8 | 1.000 (1.000 to 1.000), n=8 |

## Context metrics (retrieval), raw systems

| System | Context relevance, all episodes | Context precision, answerable | Context recall, answerable |
| --- | ---: | ---: | ---: |
| base | 0.358 (0.326 to 0.392), n=296 | 0.696 (0.648 to 0.743), n=176 | 0.892 (0.847 to 0.937), n=176 |
| main | 0.356 (0.321 to 0.393), n=299 | 0.692 (0.643 to 0.740), n=179 | 0.950 (0.916 to 0.978), n=179 |
| candF1 | 0.359 (0.322 to 0.395), n=299 | 0.700 (0.652 to 0.747), n=179 | 0.955 (0.922 to 0.983), n=179 |

## The judge against our annotation-based labels

Each answered episode carries our scorer-v3 outcome. Here is what the judge says about the answers in each group.

| System | Annotation group | Answers | Judged | Mean faithfulness | Share fully faithful | Mean hallucination | Share with any hallucination |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | supported_correct | 88 | 88 | 0.966 | 0.966 | 0.045 | 0.045 |
| main | wrong_answer | 43 | 41 | 0.610 | 0.585 | 0.390 | 0.415 |
| main | answered on hidden evidence | 35 | 33 | 0.212 | 0.212 | 0.788 | 0.788 |
| main+FB+V | supported_correct | 85 | 85 | 0.965 | 0.965 | 0.035 | 0.035 |
| main+FB+V | wrong_answer | 28 | 28 | 0.607 | 0.571 | 0.393 | 0.429 |
| main+FB+V | answered on hidden evidence | 12 | 12 | 0.167 | 0.167 | 0.833 | 0.833 |
| candF1+FB+V | supported_correct | 104 | 104 | 0.976 | 0.971 | 0.024 | 0.029 |
| candF1+FB+V | wrong_answer | 16 | 16 | 0.609 | 0.500 | 0.391 | 0.500 |
| candF1+FB+V | answered on hidden evidence | 8 | 8 | 0.250 | 0.250 | 0.750 | 0.750 |
| base | supported_correct | 8 | 8 | 1.000 | 1.000 | 0.000 | 0.000 |
| base | wrong_answer | 132 | 130 | 0.834 | 0.762 | 0.166 | 0.238 |
| base | answered on hidden evidence | 91 | 86 | 0.275 | 0.174 | 0.725 | 0.826 |

## Differences (same sampled episodes, 3,000 bootstrap draws)

| Comparison | Difference | 95% interval |
| --- | ---: | ---: |
| faithfulness: main minus base | 0.097 | 0.019 to 0.173 |
| faithfulness: main+FB+V minus base | 0.183 | 0.105 to 0.264 |
| faithfulness: main+FB+V minus main | 0.086 | 0.043 to 0.136 |
| faithfulness: candF1+FB+V minus base | 0.260 | 0.185 to 0.334 |
| faithfulness: candF1+FB+V minus main+FB+V | 0.077 | 0.008 to 0.152 |
| hallucination: main minus base | -0.091 | -0.169 to -0.015 |
| hallucination: main+FB+V minus base | -0.183 | -0.261 to -0.101 |
| hallucination: main+FB+V minus main | -0.092 | -0.141 to -0.045 |
| hallucination: candF1+FB+V minus base | -0.260 | -0.335 to -0.184 |
| hallucination: candF1+FB+V minus main+FB+V | -0.077 | -0.147 to -0.009 |
| answer_relevancy: main minus base | 0.064 | 0.031 to 0.096 |
| answer_relevancy: main+FB+V minus base | 0.088 | 0.062 to 0.118 |
| answer_relevancy: main+FB+V minus main | 0.024 | 0.006 to 0.051 |
| answer_relevancy: candF1+FB+V minus base | 0.077 | 0.043 to 0.110 |
| answer_relevancy: candF1+FB+V minus main+FB+V | -0.011 | -0.029 to 0.000 |
| response_groundedness: main minus base | 0.341 | 0.279 to 0.404 |
| response_groundedness: main+FB+V minus base | 0.343 | 0.277 to 0.408 |
| response_groundedness: main+FB+V minus main | 0.002 | -0.008 to 0.017 |
| response_groundedness: candF1+FB+V minus base | 0.336 | 0.270 to 0.404 |
| response_groundedness: candF1+FB+V minus main+FB+V | -0.007 | -0.025 to 0.002 |

## The two pre-stated readings

1. Mean judge hallucination of main's answers on hidden-evidence episodes: 0.788 over 33 answers. The rule needed more than 0.5 to say the judge corroborates the annotation-based flag: **corroborated**.
2. Self-check effect on faithfulness (`main+FB+V` minus main): 0.086, interval 0.043 to 0.136. The rule needed the interval above 0: **improves**.

## Judge repeatability

Every third episode of the sample (all systems) was judged a second time with a different cache salt, so the judge answered again. The main results above are the first pass.

| Metric | Pairs scored in both passes | Identical value | Mean absolute difference | Mean, first pass | Mean, second pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| faithfulness | 181 | 0.956 | 0.014 | 0.701 | 0.711 |
| hallucination | 181 | 0.956 | 0.014 | 0.299 | 0.289 |
| response_groundedness | 188 | 0.968 | 0.023 | 0.863 | 0.850 |
| answer_relevancy | 182 | 0.973 | 0.010 | 0.964 | 0.969 |
| context_relevance | 299 | 0.893 | 0.024 | 0.349 | 0.349 |
| context_precision | 179 | 0.972 | 0.010 | 0.676 | 0.678 |
| context_recall | 179 | 0.989 | 0.011 | 0.916 | 0.916 |


## What was not scored

| System and metric | Status counts |
| --- | --- |
| base, answer_relevancy | scored 229; no_claims 2 |
| base, faithfulness | scored 224; no_claims 5; no_context 2 |
| base, hallucination | scored 224; no_claims 5; no_context 2 |
| base, response_groundedness | scored 228; no_context 2; unreadable_reply 1 |
| candF1, context_relevance | scored 299; provider_blocked 1 |
| candF1, faithfulness | scored 181; no_claims 1 |
| candF1, hallucination | scored 181; no_claims 1 |
| main, answer_relevancy | scored 165; no_claims 1 |
| main, context_relevance | scored 299; provider_blocked 1 |
| main, faithfulness | scored 162; no_claims 4 |
| main, hallucination | scored 162; no_claims 4 |
