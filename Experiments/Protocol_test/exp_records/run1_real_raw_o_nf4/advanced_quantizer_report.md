# Run6 advanced O quantizer

## Configuration

Run6 tests whether a larger, longer-trained O model improves the global quantizer rather than relying on coordinate-wise expert switching.

| Setting | Baseline O | Run6 advanced O |
|---|---:|---:|
| hidden width | 100 | 200 |
| learned-model parameters at 25 SI channels | 158,813 | 617,613 |
| parameter multiplier | 1x | 3.89x |
| training epochs | 80 | 160 |
| quantizer learning rate | `3e-3` | `3e-3` |
| seed | 45 | 45 |

The protocol is `wz_cancer|replace_T_with='O'|quantizer_hidden_dim=200|quantizer_train_epochs=160`. All other replay conditions match run4: raw-delta history, raw committed-history replacement, rounds 14--18, and five clients.

The shared configuration widens and lengthens training for both the learned quantizer and the separately fitted conditional prior. Run6 therefore tests the complete learned O coding model at greater capacity; it does not isolate quantizer capacity from prior capacity.

## Aggregate result

Pooled WMAPE and `J = rate + 0.0875 * 20 log10(WMAPE / 100)` are used below.

| Global O candidate | Rate | WMAPE | `J` |
|---|---:|---:|---:|
| seed 44, width 100, 80 epochs | 0.45411 | 21.598% | -0.71067 |
| seed 45, width 100, 80 epochs | 0.47695 | **20.630%** | -0.72269 |
| seed 45, width 200, 160 epochs | **0.41833** | 21.517% | **-0.74931** |

Run6 obtains the best global RD objective of the three O runs. Relative to the better baseline objective, seed-45 width-100 O, it improves `J` by `0.02662`. Relative to seed-44 O, it improves `J` by `0.03863`.

Its main improvement is rate:

- `7.88%` below seed-44 O;
- `12.29%` below seed-45 O.

Its distortion is close to seed-44 O but worse than seed-45 O. Increased capacity and epochs therefore moved the operating point toward a substantially better rate with an intermediate distortion, rather than uniformly improving both raw metrics.

## Per-round means

| Round | Rate | WMAPE |
|---:|---:|---:|
| 14 | 0.42591 | 21.622% |
| 15 | 0.42094 | 21.557% |
| 16 | 0.41871 | 21.508% |
| 17 | 0.41402 | 21.444% |
| 18 | 0.41205 | 21.443% |

Both rate and distortion improve slightly through the frozen rounds under the refreshed raw-delta side information.

## Model and runtime cost

| Mean per record | Baseline seed-44 O | Run6 advanced O |
|---|---:|---:|
| encoder model size | 0.0910 MB | 0.3573 MB |
| decoder model size | 0.1918 MB | 0.7330 MB |
| encode/training time | 108.3 s | 410.4 s |
| total encode time over 25 records | 0.75 h | 2.85 h |

The runtime and model-description costs grow by approximately the same order as the parameter count. These costs are not part of the reported prior rate and must be included before treating the improved global `J` as a deployed compression gain.

## Learning-rate interpretation

Training completed all required finite attempts for all 25 client-round records without numerical failure, so `3e-3` is operational for this configuration. This run does not establish that it is optimal. Width and epoch count changed together, and there is no learning-rate sweep or held-out training-curve comparison.

The appropriate conclusion is:

> More capacity and training improve the measured global O RD point substantially at the existing learning rate, mainly through lower conditional rate. The result justifies a controlled width/epoch/learning-rate ablation, but it does not yet identify which change caused the gain.
