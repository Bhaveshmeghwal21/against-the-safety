# Round 4 pre-registered wrappers on the final test

| system | macro evidence success | single | compound | supported success | unsupported / hidden | false refusals | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 0.357 | 0.481 | 0.233 | 0.202 | 40/187 | 0.302 | 0.111 |
| candA | 0.375 | 0.554 | 0.196 | 0.252 | 55/187 | 0.225 | 0.042 |
| FB(main) | 0.427 | 0.542 | 0.312 | 0.202 | 40/187 | 0.374 | 0.000 |
| FB(candA) | 0.413 | 0.588 | 0.238 | 0.252 | 55/187 | 0.233 | 0.000 |
| G(main, candA) | 0.446 | 0.558 | 0.333 | 0.176 | 25/187 | 0.531 | 0.000 |

Differences against main, 95% intervals grouped by question:

- FB(main): macro +0.070 [0.04560593038246674, 0.09883205799436162]; unsupported rate +0.000 [0.0, 0.0]; supported success +0.000 [0.0, 0.0]
- G(main, candA): macro +0.089 [0.057946682946682926, 0.12174788562223116]; unsupported rate -0.080 [-0.12565445026178013, -0.042328042328042326]; supported success -0.027 [-0.04980842911877392, -0.007604562737642595]

Pre-registered claims:

- FB(main) raises the score without more unsupported answers: **True**
- G(main, candA) lowers the unsupported-answer rate: **True**
- G(main, candA) does not lower the score: **True**

Pre-registered before the test traces were read. Inference-time wrappers, not new adapters. They do not change the headline automatically.
