# Reduced-epoch gradient warm-start report

Status: **FINAL** (12/12 repeats).

Seed-42069 SVHN/ResNet-18 trace with five total clients; only client 0 is compressed. The round-12 OR parent and round-22 candidates use ten raw SI vectors (all five clients from the preceding two rounds). Rounds 13–21 advance only in the raw FL trace and never call a codec.

Rate is cumulative realized rANS `coder_stage_rates`; distortion is hard-symbol stage WMAPE in dB. The 30-epoch gaps are piecewise-linear equal-distortion rate differences versus paired 120-epoch scratch training; the single frozen parent is compared with the mean full-scratch curve. Positive is worse.

| condition | complete | failures | rejects | bad frequency | valid-only median gap (bit) | median train s | median total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_parent | 1/1 | 0 | 0 | 0.000 | +0.0435 | 0.0 | 29.6 |
| scratch30 | 12/12 | 0 | 9 | 0.750 | -0.0087 | 57.2 | 104.6 |
| warm30 | 12/12 | 0 | 6 | 0.583 | +0.0025 | 56.2 | 104.0 |
| scratch120 | 12/12 | 0 | 0 | nan | +nan | 205.2 | 279.4 |

## Verdict

The direct warm-30 minus scratch-30 comparison has median -0.0062 bit over 2 valid paired curves, with 10 no-overlap rejections and 1/2 valid warm-start wins. That survivor-only result is too sparse to establish an RD winner.

Warm-30 overlaps its paired full-scratch reference in 6/12 repeats, versus 3/12 for scratch-30. Under the declared rejection-or->+0.05-bit rule, warm-30 is bad in 7/12 repeats and scratch-30 in 9/12. Warm initialization therefore improves 30-epoch convergence reliability, but neither reduced-epoch method is reliable enough to replace full training.

As a diagnostic, at the final hard-symbol stage warm-30 versus scratch-30 has median -0.974 dB WMAPE and +0.098 bit/symbol realized rANS rate; it has lower distortion in 9/12 repeats but lower rate in only 2/12. This is a move along the tradeoff, not evidence that its RD curve dominates.

Warm-30 reduces median quantizer training time by 3.65x and median end-to-end codec time by 2.69x relative to scratch-120. Practical recommendation: warm-start when constrained to 30 epochs, retain the frozen parent as a zero-training fallback, and reject/fall back unless the newly trained hard-symbol curve passes the same held-out RD overlap and rate-gap gate. Keep 120-epoch scratch training when a reliable standalone curve is required.

## Final-stage diagnostic (not the RD decision metric)

Negative deltas favor the candidate on the named endpoint metric.

| candidate vs reference | pairs | median WMAPE delta (dB) | lower distortion | median rate delta (bit/symbol) | lower rate |
|---|---:|---:|---:|---:|---:|
| scratch30 vs scratch120 | 12 | +4.533 | 1/12 | -0.332 | 10/12 |
| warm30 vs scratch120 | 12 | +3.473 | 2/12 | -0.271 | 9/12 |
| warm30 vs scratch30 | 12 | -0.974 | 9/12 | +0.098 | 2/12 |
