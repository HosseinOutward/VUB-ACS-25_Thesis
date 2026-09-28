# Warm-start retraining report

Status: **FINAL** (24/24 cases).

The experiment preserves the Brent-scale data, production protocol, architectures, training schedule, normalization, and hard-symbol evaluation. The requested differences are 12 repeats, an additive SI shift of +0.1, disk-cached parent states, and the declared LR/tau retraining variants.

The frozen-parent shifted/unshifted final-stage MSE ratio currently has median 1.611 across 24 paired cases.

Bad convergence is predeclared as an invalid/non-overlapping complete curve or a trained-prior equal-distortion rate gap greater than 0.05 bits versus paired full scratch training.

| condition | architecture | complete | failures | RD rejects | bad frequency | median train s |
|---|---:|---:|---:|---:|---:|---:|
| frozen_parent | 2×3 | 12/12 | 0 | 0 | nan | 405.3 |
| scratch | 2×3 | 12/12 | 0 | 0 | nan | 406.1 |
| warm_naive | 2×3 | 12/12 | 0 | 1 | 0.083 | 406.3 |
| warm_lr10 | 2×3 | 12/12 | 0 | 1 | 0.167 | 405.4 |
| warm_tau10 | 2×3 | 12/12 | 0 | 1 | 0.250 | 405.3 |
| warm_lr10_tau10 | 2×3 | 12/12 | 0 | 1 | 0.083 | 406.0 |
| frozen_parent | 4×2 | 12/12 | 0 | 0 | nan | 285.0 |
| scratch | 4×2 | 12/12 | 0 | 0 | nan | 284.0 |
| warm_naive | 4×2 | 12/12 | 0 | 0 | 0.083 | 284.6 |
| warm_lr10 | 4×2 | 12/12 | 0 | 0 | 0.167 | 284.1 |
| warm_tau10 | 4×2 | 12/12 | 0 | 1 | 0.250 | 285.6 |
| warm_lr10_tau10 | 4×2 | 12/12 | 0 | 0 | 0.000 | 284.0 |

## Findings

- The +0.1 SI shift increased frozen-parent final-stage MSE in all 12/12 2×3 repeats; median multiplier 1.653 (range 1.578–2.013).
- The +0.1 SI shift increased frozen-parent final-stage MSE in all 12/12 4×2 repeats; median multiplier 1.498 (range 1.432–1.612).
- There were no production training exceptions or fatal candidate-rejection failures. One 2×3 parent (repeat 6) converged to a zero-rate constant-symbol solution; every warm variant inherited that collapse, while full scratch training recovered.
- Against scratch on trained-prior RD, bad-convergence totals over both architectures were: naïve warm 2/24, 10× lower LR 4/24, lower tau 6/24, and the LR+tau combination 1/24.
- No warm variant dominates every criterion. The combination had the fewest declared bad convergences, naïve warm stayed closest to scratch for the separately retrained conditional-prior rate, and lower tau alone was the least stable.
- Warm starts did not reduce runtime because every method intentionally retained all 180 epochs.

## Equal-distortion trained-prior RD gaps

Positive values are worse than the reference. Rejections mean the complete hard-symbol curves had no shared distortion interval.

| condition | architecture | vs frozen mean (valid/rejected) | vs scratch median (valid/rejected) |
|---|---:|---:|---:|
| warm_naive | 2×3 | -0.9358 (10/2) | +0.0117 (11/1) |
| warm_lr10 | 2×3 | -0.9411 (10/2) | +0.0005 (11/1) |
| warm_tau10 | 2×3 | -0.5368 (11/1) | +0.0211 (11/1) |
| warm_lr10_tau10 | 2×3 | -0.9306 (10/2) | +0.0145 (11/1) |
| warm_naive | 4×2 | -0.8872 (12/0) | +0.0106 (12/0) |
| warm_lr10 | 4×2 | -0.8838 (12/0) | +0.0151 (12/0) |
| warm_tau10 | 4×2 | -0.8059 (12/0) | +0.0384 (11/1) |
| warm_lr10_tau10 | 4×2 | -0.8901 (12/0) | +0.0127 (12/0) |

All three rate definitions and repeat-level values are preserved in `rd_summary.csv` and `rd_gaps.csv`; the table above uses the quantizer's jointly trained prior, the primary RD measure.
