# Round 5 development selection (scorer v2, fail-closed applied to every system)

| system | macro | single | compound | HotpotQA | supported correct | unsupported / hidden | HotpotQA unsupported | invalid before FB | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.538 | 0.603 | 0.387 | 0.623 | 0.228 | 5/131 | 1 | 0.134 | reference |
| candA | 0.544 | 0.615 | 0.369 | 0.648 | 0.249 | 8/130 | 4 | 0.053 | reference |
| candE1 | 0.568 | 0.593 | 0.468 | 0.642 | 0.265 | 5/131 | 3 | 0.249 | False |
| candE2 | 0.570 | 0.641 | 0.432 | 0.635 | 0.302 | 6/131 | 3 | 0.150 | False |
| candF1 | 0.583 | 0.612 | 0.495 | 0.642 | 0.312 | 7/131 | 2 | 0.112 | False |
| candF2 | 0.578 | 0.641 | 0.432 | 0.660 | 0.323 | 10/131 | 3 | 0.090 | False |

Selected for the sealed final test: **none**.

- candE1: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=yes, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candE2: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=NO, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candF1: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
- candF2: fewer_unsupported_pooled=NO, no_more_unsupported_in_either_family=NO, hotpot_unsupported_not_more=NO, supported_correct_not_lower=yes, invalid_not_more_than_plus_1pt=yes, hotpot_success_not_lower_by_3pts=yes, macro_plus_3pts=yes
