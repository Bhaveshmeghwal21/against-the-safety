# Round 5b development selection (scorer v2, fail-closed and the checker on every system)

| system | macro | single | compound | HotpotQA | supported correct | unsupported / hidden | HotpotQA unsupported | own abstentions on hidden | invalid before FB | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.538 | 0.603 | 0.387 | 0.623 | 0.228 | 5/131 | 1 | 0.775 | 0.134 | reference |
| candA | 0.544 | 0.615 | 0.369 | 0.648 | 0.249 | 8/130 | 4 | 0.798 | 0.053 | reference |
| candE1 | 0.568 | 0.593 | 0.468 | 0.642 | 0.265 | 5/131 | 3 | 0.632 | 0.249 | False |
| candE2 | 0.570 | 0.641 | 0.432 | 0.635 | 0.302 | 6/131 | 3 | 0.686 | 0.150 | False |
| candF1 | 0.583 | 0.612 | 0.495 | 0.642 | 0.312 | 7/131 | 2 | 0.775 | 0.112 | True |
| candF2 | 0.578 | 0.641 | 0.432 | 0.660 | 0.323 | 10/131 | 3 | 0.725 | 0.090 | False |
| candG1 | 0.481 | 0.529 | 0.423 | 0.491 | 0.148 | 2/132 | 2 | 0.000 | 0.698 | False |
| candG2 | 0.558 | 0.700 | 0.432 | 0.541 | 0.381 | 9/132 | 9 | 0.000 | 0.202 | False |
| candH1 | 0.546 | 0.678 | 0.400 | 0.560 | 0.360 | 12/129 | 6 | 0.000 | 0.128 | False |
| candH2 | 0.557 | 0.678 | 0.445 | 0.547 | 0.386 | 12/129 | 8 | 0.000 | 0.103 | False |
| candI1 | 0.528 | 0.600 | 0.423 | 0.560 | 0.233 | 3/132 | 3 | 0.829 | 0.069 | False |
| candI2 | 0.566 | 0.695 | 0.423 | 0.579 | 0.360 | 7/132 | 5 | 0.634 | 0.047 | False |
| candK1 | 0.574 | 0.756 | 0.445 | 0.522 | 0.492 | 16/130 | 12 | 0.167 | 0.047 | False |
| candK2 | 0.591 | 0.767 | 0.495 | 0.509 | 0.519 | 13/130 | 12 | 0.158 | 0.065 | False |

Selected for the sealed final test: **candF1**.

- candE1: abstention_retained_0.75=yes, unsupported_rate_within_3pts=yes, hotpot_unsupported_within_2=yes, supported_correct_plus_5pts=NO, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candE2: abstention_retained_0.75=yes, unsupported_rate_within_3pts=yes, hotpot_unsupported_within_2=yes, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candF1: abstention_retained_0.75=yes, unsupported_rate_within_3pts=yes, hotpot_unsupported_within_2=yes, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candF2: abstention_retained_0.75=yes, unsupported_rate_within_3pts=NO, hotpot_unsupported_within_2=yes, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candG1: abstention_retained_0.75=NO, unsupported_rate_within_3pts=yes, hotpot_unsupported_within_2=yes, supported_correct_plus_5pts=NO, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candG2: abstention_retained_0.75=NO, unsupported_rate_within_3pts=NO, hotpot_unsupported_within_2=NO, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candH1: abstention_retained_0.75=NO, unsupported_rate_within_3pts=NO, hotpot_unsupported_within_2=NO, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candH2: abstention_retained_0.75=NO, unsupported_rate_within_3pts=NO, hotpot_unsupported_within_2=NO, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candI1: abstention_retained_0.75=yes, unsupported_rate_within_3pts=yes, hotpot_unsupported_within_2=yes, supported_correct_plus_5pts=NO, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candI2: abstention_retained_0.75=yes, unsupported_rate_within_3pts=yes, hotpot_unsupported_within_2=NO, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candK1: abstention_retained_0.75=NO, unsupported_rate_within_3pts=NO, hotpot_unsupported_within_2=NO, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=yes
- candK2: abstention_retained_0.75=NO, unsupported_rate_within_3pts=NO, hotpot_unsupported_within_2=NO, supported_correct_plus_5pts=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=yes
