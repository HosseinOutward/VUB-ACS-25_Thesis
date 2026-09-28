# Round-14 Oracle quantizer parameter search

## Result

The L27 screen completed all 27 configurations. Six beat the fixed `run1` reference, where a negative objective means a lower slope-adjusted rate at the same distortion. The best observed configuration was trial 13:

| Parameter | Value |
|---|---:|
| bins per plane | 3 |
| reconstruction lambda | 300 |
| temperature | 1 |
| learning rate | 0.009 |
| epochs | 120 |
| training samples | 600,000 |
| batch size | 25,000 |
| LR step | 25% of training |
| hidden dimension | 50 |
| layers | 3 |
| training match bits | 26 |
| source perturbation | `draw_discard` |

Its mean rate-equivalent gap was **−0.04284 bits/symbol**, approximately 13.5% of the reference's mean stage rate. A fresh-seed repeat produced **−0.04063 bits/symbol**, so the result was stable on the screened client.

| Stage | Reference rate | Trial 13 rate | Reference WMAPE | Trial 13 WMAPE | Rate-equivalent gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1961 | 0.1732 | 30.529% | 29.971% | −0.03674 |
| 2 | 0.3016 | 0.3158 | 26.669% | 24.643% | −0.04505 |
| 3 | 0.4532 | 0.3787 | 21.609% | 22.424% | −0.04672 |

The stage-3 point accepts 0.815 percentage points more WMAPE while reducing rate by 0.0744 bits/symbol. After accounting for the reference RD slope, this is still a 0.0467-bit improvement.

## Interpretation

The largest main effects came from training sample count, epochs, learning rate, LR schedule, and model width. More samples and 120 epochs were consistently helpful. Batch size had little independent effect.

The independently selected main-effect levels did **not** combine into a better model. That synthesized configuration scored +0.00184 bits/symbol on the paired confirmation and took 440 seconds, versus −0.04063 and 168 seconds for trial 13. This indicates important parameter interactions. In particular:

- the screen marginally preferred learning rate 0.003, width 150, four layers, and 28 match bits;
- the reproducible best used learning rate 0.009, width 50, three layers, and 26 match bits;
- therefore the main-effect recommendation should not replace the best observed row.

The result also supports retaining the production 26-bit training dedup setting: it participated in the strongest observed and confirmed configuration, even though 28 bits had the better marginal mean.

## Possible solutions

1. **Best overall: trial 13.** It has the strongest confirmed slope-aware RD result and is compact. Use it as the primary candidate.
2. **Lower-rate alternative: trial 7.** Its final stage is 0.2477 bits/symbol at 27.619% WMAPE, with a mean gap of −0.02397.
3. **Higher-quality alternative: trial 19.** Its final stage is 0.5741 bits/symbol at 17.770% WMAPE, with a mean gap of −0.02391.

These alternatives occupy meaningfully different regions of the RD curve; selecting among them should depend on the desired operating rate rather than the scalar objective alone.

## Limits and next check

The screen used one round-14 client and one seed per L27 row. Trial 13 was repeated on a fresh seed, but not on other clients or rounds. The fixed reference is averaged over five clients, so the absolute gaps should be treated as screening estimates.

Before changing protocol defaults, confirm trials 13, 7, and 19 on all five round-14 clients and at least one additional round. Trial 13 should remain the default candidate unless that broader check changes the ranking.

Raw results and factor effects are in [`summary.json`](summary.json) and [`trials/`](trials/).
