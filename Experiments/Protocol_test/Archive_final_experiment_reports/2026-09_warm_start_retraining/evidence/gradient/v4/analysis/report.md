# Reduced-epoch gradient warm-start report

Status: **FINAL** (12/12 repeats).

Seed-42069 SVHN/ResNet-18 trace with five total clients; only client 0 is compressed. The round-12 OR parent and round-22 candidates use ten raw SI vectors (all five clients from the preceding two rounds). Rounds 13–21 advance only in the raw FL trace and never call a codec.

Rate is cumulative realized rANS `coder_stage_rates`; distortion is hard-symbol stage WMAPE in dB. The 30-epoch gaps are piecewise-linear equal-distortion rate differences versus paired 120-epoch scratch training; the single frozen parent is compared with the mean full-scratch curve. Positive is worse.

| condition | complete | failures | rejects | bad frequency | valid-only median gap (bit) | median train s | median total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_parent | 1/1 | 0 | 0 | 0.000 | +0.0435 | 0.0 | 29.6 |
| scratch30 | 12/12 | 0 | 9 | 0.750 | -0.0087 | 56.9 | 104.7 |
| warm30_lr1_tau5p5 | 12/12 | 0 | 6 | 0.583 | +0.0025 | 55.9 | 104.9 |
| warm30_lr1_tau1 | 12/12 | 0 | 7 | 0.667 | -0.0008 | 56.3 | 104.8 |
| warm30_lr1_tau0p5 | 12/12 | 0 | 8 | 0.667 | -0.0054 | 56.3 | 104.1 |
| warm30_lr1_tau0p1 | 12/12 | 0 | 10 | 0.833 | -0.0128 | 56.5 | 103.6 |
| warm30_lr0p3_tau5p5 | 12/12 | 0 | 2 | 0.333 | +0.0371 | 57.0 | 105.0 |
| warm30_lr0p3_tau1 | 12/12 | 0 | 2 | 0.333 | +0.0086 | 56.5 | 104.4 |
| warm30_lr0p3_tau0p5 | 12/12 | 0 | 2 | 0.167 | -0.0074 | 56.2 | 104.3 |
| warm30_lr0p3_tau0p1 | 12/12 | 0 | 3 | 0.250 | -0.0172 | 56.6 | 104.0 |
| warm30_lr0p1_tau5p5 | 12/12 | 0 | 2 | 0.167 | +0.0145 | 56.7 | 104.2 |
| warm30_lr0p1_tau1 | 12/12 | 0 | 2 | 0.333 | +0.0217 | 56.7 | 104.8 |
| warm30_lr0p1_tau0p5 | 12/12 | 0 | 2 | 0.250 | +0.0253 | 56.6 | 104.4 |
| warm30_lr0p1_tau0p1 | 12/12 | 0 | 2 | 0.250 | +0.0064 | 56.6 | 104.2 |
| warm30_lr0p03_tau5p5 | 12/12 | 0 | 2 | 0.417 | +0.0411 | 56.8 | 104.4 |
| warm30_lr0p03_tau1 | 12/12 | 0 | 2 | 0.250 | +0.0308 | 57.2 | 105.1 |
| warm30_lr0p03_tau0p5 | 12/12 | 0 | 2 | 0.500 | +0.0345 | 56.7 | 104.8 |
| warm30_lr0p03_tau0p1 | 12/12 | 0 | 2 | 0.583 | +0.0504 | 56.5 | 104.4 |
| scratch120 | 12/12 | 0 | 0 | nan | +nan | 206.4 | 278.7 |

## Verdict

By the declared rejection-or->+0.05-bit rule, the lowest observed bad-convergence count is warm30_lr0p3_tau0p5 at 2/12; scratch-30 is 9/12. Selection must also consider the valid paired RD gaps below; survivor-only medians are not evidence when overlap is sparse.

## Direct warm versus scratch-30 RD

| warm policy | valid overlaps | rejections | valid-only median gap (bit) | warm wins |
|---|---:|---:|---:|---:|
| warm30_lr1_tau5p5 | 2/12 | 10 | -0.0062 | 1/2 |
| warm30_lr1_tau1 | 3/12 | 9 | +0.0000 | 1/3 |
| warm30_lr1_tau0p5 | 1/12 | 11 | -0.0208 | 1/1 |
| warm30_lr1_tau0p1 | 3/12 | 9 | +0.0000 | 1/3 |
| warm30_lr0p3_tau5p5 | 5/12 | 7 | +0.0146 | 1/5 |
| warm30_lr0p3_tau1 | 4/12 | 8 | +0.0181 | 0/4 |
| warm30_lr0p3_tau0p5 | 6/12 | 6 | -0.0014 | 3/6 |
| warm30_lr0p3_tau0p1 | 4/12 | 8 | -0.0110 | 3/4 |
| warm30_lr0p1_tau5p5 | 3/12 | 9 | +0.0165 | 0/3 |
| warm30_lr0p1_tau1 | 5/12 | 7 | +0.0189 | 1/5 |
| warm30_lr0p1_tau0p5 | 3/12 | 9 | +0.0129 | 0/3 |
| warm30_lr0p1_tau0p1 | 2/12 | 10 | +0.0175 | 0/2 |
| warm30_lr0p03_tau5p5 | 5/12 | 7 | +0.0332 | 0/5 |
| warm30_lr0p03_tau1 | 4/12 | 8 | +0.0266 | 0/4 |
| warm30_lr0p03_tau0p5 | 5/12 | 7 | +0.0304 | 0/5 |
| warm30_lr0p03_tau0p1 | 5/12 | 7 | +0.0230 | 1/5 |

All warm policies use round-12 model weights with a fresh optimizer and freshly trained conditional prior. The grid uses LR multipliers 1, 0.3, 0.1, and 0.03 and initial tau values 5.5, 1, 0.5, and 0.1. Every tau schedule anneals toward 0.1, so initial tau 0.1 is constant. Every round-22 fit uses one quantizer candidate.

The selected 30-epoch policy reduces median quantizer training time by 3.67x and median end-to-end time by 2.67x relative to scratch-120.

## Final-stage diagnostic (not the RD decision metric)

Negative deltas favor the candidate on the named endpoint metric.

| candidate vs reference | pairs | median WMAPE delta (dB) | lower distortion | median rate delta (bit/symbol) | lower rate |
|---|---:|---:|---:|---:|---:|
| scratch30 vs scratch120 | 12 | +4.533 | 1/12 | -0.332 | 10/12 |
| warm30_lr1_tau5p5 vs scratch120 | 12 | +3.473 | 2/12 | -0.271 | 9/12 |
| warm30_lr1_tau5p5 vs scratch30 | 12 | -0.974 | 9/12 | +0.098 | 2/12 |
| warm30_lr1_tau1 vs scratch120 | 12 | +3.275 | 2/12 | -0.274 | 10/12 |
| warm30_lr1_tau1 vs scratch30 | 12 | -0.937 | 8/12 | +0.041 | 3/12 |
| warm30_lr1_tau0p5 vs scratch120 | 12 | +4.694 | 1/12 | -0.372 | 10/12 |
| warm30_lr1_tau0p5 vs scratch30 | 12 | -0.420 | 7/12 | +0.008 | 5/12 |
| warm30_lr1_tau0p1 vs scratch120 | 12 | +8.217 | 2/12 | -0.698 | 10/12 |
| warm30_lr1_tau0p1 vs scratch30 | 12 | +2.219 | 4/12 | -0.250 | 9/12 |
| warm30_lr0p3_tau5p5 vs scratch120 | 12 | -0.194 | 8/12 | +0.033 | 5/12 |
| warm30_lr0p3_tau5p5 vs scratch30 | 12 | -5.392 | 12/12 | +0.410 | 0/12 |
| warm30_lr0p3_tau1 vs scratch120 | 12 | +0.830 | 5/12 | +0.004 | 6/12 |
| warm30_lr0p3_tau1 vs scratch30 | 12 | -5.324 | 12/12 | +0.391 | 0/12 |
| warm30_lr0p3_tau0p5 vs scratch120 | 12 | +0.306 | 6/12 | -0.025 | 6/12 |
| warm30_lr0p3_tau0p5 vs scratch30 | 12 | -4.387 | 12/12 | +0.411 | 0/12 |
| warm30_lr0p3_tau0p1 vs scratch120 | 12 | +2.146 | 3/12 | -0.159 | 9/12 |
| warm30_lr0p3_tau0p1 vs scratch30 | 12 | -3.976 | 12/12 | +0.307 | 0/12 |
| warm30_lr0p1_tau5p5 vs scratch120 | 12 | -0.003 | 6/12 | +0.019 | 4/12 |
| warm30_lr0p1_tau5p5 vs scratch30 | 12 | -4.866 | 12/12 | +0.447 | 0/12 |
| warm30_lr0p1_tau1 vs scratch120 | 12 | -0.125 | 6/12 | +0.013 | 3/12 |
| warm30_lr0p1_tau1 vs scratch30 | 12 | -5.226 | 12/12 | +0.398 | 0/12 |
| warm30_lr0p1_tau0p5 vs scratch120 | 12 | -0.206 | 7/12 | +0.017 | 4/12 |
| warm30_lr0p1_tau0p5 vs scratch30 | 12 | -5.361 | 12/12 | +0.395 | 0/12 |
| warm30_lr0p1_tau0p1 vs scratch120 | 12 | -0.287 | 8/12 | +0.038 | 3/12 |
| warm30_lr0p1_tau0p1 vs scratch30 | 12 | -5.965 | 12/12 | +0.412 | 0/12 |
| warm30_lr0p03_tau5p5 vs scratch120 | 12 | +0.235 | 5/12 | +0.011 | 2/12 |
| warm30_lr0p03_tau5p5 vs scratch30 | 12 | -4.680 | 12/12 | +0.467 | 0/12 |
| warm30_lr0p03_tau1 vs scratch120 | 12 | -0.001 | 6/12 | +0.021 | 4/12 |
| warm30_lr0p03_tau1 vs scratch30 | 12 | -4.978 | 12/12 | +0.373 | 0/12 |
| warm30_lr0p03_tau0p5 vs scratch120 | 12 | +0.159 | 5/12 | +0.041 | 3/12 |
| warm30_lr0p03_tau0p5 vs scratch30 | 12 | -5.086 | 12/12 | +0.383 | 0/12 |
| warm30_lr0p03_tau0p1 vs scratch120 | 12 | -0.587 | 9/12 | +0.080 | 0/12 |
| warm30_lr0p03_tau0p1 vs scratch30 | 12 | -5.725 | 12/12 | +0.437 | 0/12 |
