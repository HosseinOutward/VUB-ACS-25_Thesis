# Alternating training diagnostic — complete

Single trained target round; these are successive fits on the same data, not a trained+9-frozen protocol window.

Seed 43; target round 7; fixed full decoder SI from rounds 5–6, all five clients (10 compressed-history vectors). Raw prefix is an explicitly available decoder seed.

Each prior phase refits fixed hard symbols with the existing best-of-three prior fitter. Quantizer/joint phases each continue one trajectory with the existing warm schedule and RD objective. Rates are hard-symbol conditional cross-entropies in bits/float; no entropy coder or symbol transport is run.

Matrices: row versus column, negative favors the row. Both directions are recomputed. Clients are matched per pair; rates and raw WMAPE are averaged before dB conversion and interpolation. Invalid/nonmonotone/nonoverlapping curves are —.

| Phase | Clients measured |
|---|---:|
| 0:initial_joint | 5/5 |
| 1:prior | 5/5 |
| 2:quantizer | 5/5 |
| 3:prior | 5/5 |
| 4:joint | 5/5 |
| 5:prior | 5/5 |


## All clients

| Row / column | 0:initial_joint | 1:prior | 2:quantizer | 3:prior | 4:joint | 5:prior |
|---|---:|---:|---:|---:|---:|---:|
| 0:initial_joint | +0.00% | +63.36% | +61.20% | +64.11% | -16.53% | +61.67% |
| 1:prior | -38.78% | +0.00% | -1.77% | -0.00% | -49.43% | -2.06% |
| 2:quantizer | -37.97% | +1.80% | +0.00% | +1.80% | -48.53% | -0.32% |
| 3:prior | -39.07% | +0.00% | -1.77% | +0.00% | -49.43% | -2.07% |
| 4:joint | +19.80% | +97.75% | +94.27% | +97.75% | +0.00% | +93.99% |
| 5:prior | -38.15% | +2.10% | +0.32% | +2.12% | -48.45% | +0.00% |

No clients triggered exclusions. The symmetric-exclusion matrix is identical to the all-client matrix above; every pair retains clients 0–4 (5/5).

## Stage RD points, all available clients

Cumulative stage rates; all reported stages were measured. The current prior follows each phase's weights.

| Phase | Stage | Marginal bits/float | Current prior bits/float | WMAPE dB (raw %) |
|---|---:|---:|---:|---|
| 0:initial_joint | 1 | 0.642302 | 0.692066 | -3.854 (64.1637%) |
| 0:initial_joint | 2 | 1.025687 | 0.818987 | -5.164 (55.1815%) |
| 0:initial_joint | 3 | 1.514943 | 1.058367 | -8.834 (36.1644%) |
| 1:prior | 1 | 0.642302 | 0.354612 | -3.854 (64.1637%) |
| 1:prior | 2 | 1.025687 | 0.473787 | -5.164 (55.1815%) |
| 1:prior | 3 | 1.514943 | 0.709928 | -8.834 (36.1644%) |
| 2:quantizer | 1 | 0.646777 | 0.358770 | -3.859 (64.1248%) |
| 2:quantizer | 2 | 1.033339 | 0.470481 | -5.024 (56.0790%) |
| 2:quantizer | 3 | 1.522615 | 0.712312 | -8.629 (37.0305%) |
| 3:prior | 1 | 0.646777 | 0.356331 | -3.859 (64.1248%) |
| 3:prior | 2 | 1.033339 | 0.462191 | -5.024 (56.0790%) |
| 3:prior | 3 | 1.522615 | 0.698376 | -8.629 (37.0305%) |
| 4:joint | 1 | 0.649199 | 0.831574 | -3.800 (64.5680%) |
| 4:joint | 2 | 1.011413 | 0.969678 | -4.829 (57.3514%) |
| 4:joint | 3 | 1.492719 | 1.207029 | -8.389 (38.0664%) |
| 5:prior | 1 | 0.649199 | 0.354649 | -3.800 (64.5680%) |
| 5:prior | 2 | 1.011413 | 0.462488 | -4.829 (57.3514%) |
| 5:prior | 3 | 1.492719 | 0.691763 | -8.389 (38.0664%) |
