# Slepian–Wolf results handoff

Numeric artifacts are under `Experiments/slepian_wolf/results/`; paper checks are in `paper_replication_benchmark.py`. Do not generalize single-seed integration results to real FL deltas.

The non-oracle transports are compared over three seeds in `HANDOFF_PRACTICAL.md` (fixed-rate pilot) and `HANDOFF_LDPCA.md` (rate-adaptive ladder), from the same `practical_sw_benchmark.py` runs saved as `results/practical_sw_*seed*.json`. Headline: the ladder sits 11% under the pilot on both configurations, and +23% rather than +39% over the prior cross-entropy.

## Paper checks

- Fig. 7 worker 10: exact `0.3631`, semi-analytical `0.3621` bit/symbol.
- Fig. 9: reproduced decreasing rates with correlation and worker index.
- Fig. 10 one-bit point: `0.2640` bit/symbol, `73.6%` saving; paper reports about `72%`.
- Table 1 is not claimed because the paper omits the uniform quantizer range.
- Current suite: `21 passed, 5 skipped`. Skips are opt-in LDPC/neural benchmarks controlled by `SW_PAPER_RUN_LDPC`, `SW_CROSS_RUN` and `SW_NEURAL_RUN`.

Unit coverage includes LSB chain rule, paper LLR, packing/padding, GF(2) rank, deterministic matrices, BP on a tree, full/zero-rate cases, oracle payload recovery and mapping inversion.

## Tao-style LDPC versus author neural decoder

Common BSC stream, target BER below 1%; Tao blocks are 2400 bits and neural blocks are 64 bits.

| Entropy | Tao reliable fixed rate | Neural reliable fixed rate | Neural oracle/fallback rate |
|---:|---:|---:|---:|
| 0.1001 | 0.1254 | 0.1875 | 0.1491 |
| 0.1999 | 0.2500 | 0.3750 | 0.2483 |
| 0.2998 | 0.3750 | 0.5625 | 0.3727 |
| 0.4001 | 0.4754 | 0.6875 | 0.4772 |
| 0.4999 | 0.5750 | 0.8125 | 0.5958 |
| 0.5997 | 0.6750 | 0.8750 | 0.7079 |
| 0.6994 | 0.8246 | no reliable fixed point | 0.8163, 12.88% fallback |

This compares operational implementations, not exact Zhang–Tao finite-length code reproduction.

## Quantizer integration

Synthetic quantizer-check data, 48,000 positions, 60 epochs, block 2400, seed 0 and 10-row oracle grid. Rates are bits/original value. Final wire rate includes mapping metadata; syndrome rate does not.

Strict 0.01% BER-bound target, 4 bins × 2 stages:

| Labeling | Syndrome rate | Final wire rate | Final MSE | SW-added MSE | BER |
|---|---:|---:|---:|---:|---:|
| Original | 1.8908 | 2.0767 | 0.00303951 | 0.00001548 | 0.0031% |
| Value/Gray | 1.8058 | 2.0058 | 0.00302879 | 0.00000476 | 0.0057% |

The mapped point dominates here: `-3.41%` final wire rate, `-0.35%` final MSE and `-69.2%` SW-added distortion. Its mapping is stage 1 `(3,2,0,1)`, stage 2 `(0,3,1,2)`, expressed as original symbol → transmitted label.

At the looser 1% target, mapped 4×2 gives `(wire=1.8720, MSE=0.00355697)`: `-9.86%` rate but `+17.0%` MSE versus the original point. This is a tradeoff, not dominance.

For 2 bins × 3 stages, mapping only swaps binary labels and did not help: wire changed `1.4485 → 1.4620` and MSE `0.00510338 → 0.00510758`.

In every integration run, independently regenerated direct-core syndrome bytes matched the integrated payload. CUDA BP can produce different decoded symbols near threshold because its reductions are not deterministic; use CPU or deterministic reductions when bitwise reproducibility is required.

Next validation: repeat the rate–distortion comparison on held-out cached real FL deltas and multiple seeds. Oracle-selected rates are optimistic because selection and reporting currently use the same vector.
