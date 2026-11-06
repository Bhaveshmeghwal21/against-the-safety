# Round 5 final test (scorer v2)

449 episodes. Macro = mean of the component rates.

| configuration | macro | single | compound | unsupported / hidden | supported correct | false refusals | invalid before repair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 0.350 | 0.471 | 0.229 | 39/181 | 0.195 | 0.302 | 0.111 |
| main+FB | 0.419 | 0.529 | 0.309 | 39/181 | 0.195 | 0.374 | 0.111 |
| main+FB+V | 0.461 | 0.533 | 0.388 | 16/181 | 0.168 | 0.611 | 0.111 |
| candA | 0.368 | 0.541 | 0.196 | 54/182 | 0.244 | 0.225 | 0.042 |
| candA+FB | 0.407 | 0.576 | 0.238 | 54/182 | 0.244 | 0.233 | 0.042 |
| candA+FB+V | 0.490 | 0.600 | 0.381 | 18/182 | 0.233 | 0.527 | 0.042 |
| candE1 | 0.356 | 0.447 | 0.265 | 32/184 | 0.302 | 0.237 | 0.214 |
| candE1+FB | 0.507 | 0.580 | 0.434 | 32/184 | 0.302 | 0.347 | 0.214 |
| candE1+FB+V | 0.555 | 0.591 | 0.519 | 11/184 | 0.294 | 0.542 | 0.214 |

Paired differences (first minus second), 95% intervals:

- main+FB/main: macro +0.069 [+0.044, +0.097]; unsupported rate +0.000 [+0.000, +0.000]; supported correct +0.000 [+0.000, +0.000]
- main+FB+V/main+FB: macro +0.042 [+0.019, +0.067]; unsupported rate -0.127 [-0.181, -0.078]; supported correct -0.027 [-0.029, -0.004]
- main+FB+V/main: macro +0.111 [+0.076, +0.147]; unsupported rate -0.127 [-0.178, -0.076]; supported correct -0.027 [-0.030, -0.004]
- candA+FB+V/main+FB+V: macro +0.030 [+0.004, +0.054]; unsupported rate +0.011 [-0.036, +0.056]; supported correct +0.065 [+0.014, +0.063]
