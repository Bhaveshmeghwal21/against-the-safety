# Round 5 development selection (scorer v2, fail-closed applied to every system)

| system | macro | single | compound | HotpotQA | supported correct | unsupported / hidden | HotpotQA unsupported | invalid before FB | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.538 | 0.603 | 0.387 | 0.623 | 0.228 | 5/131 | 1 | 0.134 | reference |
| candA | 0.544 | 0.615 | 0.369 | 0.648 | 0.249 | 8/130 | 4 | 0.053 | reference |
| candG1 | 0.481 | 0.529 | 0.423 | 0.491 | 0.148 | 2/132 | 2 | 0.698 | False |
| candG2 | 0.558 | 0.700 | 0.432 | 0.541 | 0.381 | 9/132 | 9 | 0.202 | False |
| candH1 | 0.546 | 0.678 | 0.400 | 0.560 | 0.360 | 12/129 | 6 | 0.128 | False |
| candH2 | 0.557 | 0.678 | 0.445 | 0.547 | 0.386 | 12/129 | 8 | 0.103 | False |
| candE1 | 0.568 | 0.593 | 0.468 | 0.642 | 0.265 | 5/131 | 3 | 0.249 | False |
| candE2 | 0.570 | 0.641 | 0.432 | 0.635 | 0.302 | 6/131 | 3 | 0.150 | False |
| candF1 | 0.583 | 0.612 | 0.495 | 0.642 | 0.312 | 7/131 | 2 | 0.112 | False |
| candF2 | 0.578 | 0.641 | 0.432 | 0.660 | 0.323 | 10/131 | 3 | 0.090 | False |

Selected for the sealed final test: **none**.

- candG1: fewer_unsupported_pooled=yes, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=NO, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candG2: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candH1: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candH2: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=NO, macro_plus_3pts=NO
- candE1: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=yes, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candE2: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candF1: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candF2: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
