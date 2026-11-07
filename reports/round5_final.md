# Round 5 final test (scorer v2)

938 episodes. Macro = mean of the component rates.

| configuration | macro | single | compound | hotpot | rgb | unsupported / hidden | supported correct | false refusals | invalid before repair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.022 | 0.018 | 0.036 | 0.000 | 0.034 | 276/369 | 0.027 | 0.011 | 0.207 |
| base+FB | 0.113 | 0.127 | 0.176 | 0.049 | 0.101 | 276/369 | 0.027 | 0.208 | 0.207 |
| main | 0.414 | 0.545 | 0.224 | 0.524 | 0.363 | 108/364 | 0.290 | 0.222 | 0.085 |
| main+FB | 0.474 | 0.600 | 0.327 | 0.561 | 0.407 | 108/364 | 0.290 | 0.270 | 0.085 |
| main+FB+V | 0.517 | 0.591 | 0.394 | 0.561 | 0.524 | 37/364 | 0.275 | 0.391 | 0.085 |
| candA | 0.473 | 0.636 | 0.250 | 0.622 | 0.385 | 129/365 | 0.337 | 0.160 | 0.021 |
| candA+FB | 0.494 | 0.655 | 0.293 | 0.634 | 0.394 | 129/365 | 0.337 | 0.167 | 0.021 |
| candA+FB+V | 0.541 | 0.614 | 0.390 | 0.634 | 0.526 | 45/365 | 0.309 | 0.323 | 0.021 |

Paired differences (first minus second), 95% intervals:

- main+FB/main: macro +0.059 [+0.041, +0.079]; unsupported rate +0.000 [+0.000, +0.000]; supported correct +0.000 [+0.000, +0.000]
- main+FB+V/main+FB: macro +0.044 [+0.028, +0.059]; unsupported rate -0.195 [-0.233, -0.157]; supported correct -0.014 [-0.025, -0.005]
- main+FB+V/main: macro +0.103 [+0.079, +0.127]; unsupported rate -0.195 [-0.231, -0.159]; supported correct -0.014 [-0.025, -0.005]
- candA+FB+V/main+FB+V: macro +0.024 [+0.003, +0.045]; unsupported rate +0.022 [-0.014, +0.057]; supported correct +0.034 [+0.000, +0.066]
- main/base: macro +0.392 [+0.357, +0.425]; unsupported rate -0.451 [-0.508, -0.392]; supported correct +0.263 [+0.217, +0.310]
- main+FB+V/base: macro +0.495 [+0.468, +0.521]; unsupported rate -0.646 [-0.698, -0.591]; supported correct +0.249 [+0.203, +0.295]
