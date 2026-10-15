# Received-FP16 validated fitting: 14 October 2026

This seed-43 diagnostic compares receiver-only 80/20 splits of actual FP16 sample messages at 20k and 100k coordinates. The received sample coordinates, decoded labels, and normalization factors are identical across the soft-input, hard-input, categorical-p, and hard-transport-NLL variants. The sample fit and separate prior fit use only the training partition; held-out received rows select checkpoints, and no candidate refits its prior on the full target. All current-target coordinates are coded, the complete measured sample packet is charged once, and O is a read-only reference.

The two budgets retain 16,001/3,999 and 79,995/20,005 train/validation rows, respectively; singleton normalization strata remain in training. Each main attempt ran 35 epochs with at most one capped pass over training rows per epoch. The prior ran up to 30 epochs with held-out hard conditional-CE selection. Main validation scores are the hard-symbol normalized-MSE RD-plus-conditional-CE objective in nats; the trace records its matching train and held-out values at initialization, epoch 0, and epoch 34.

## Checkpoint traces

Each cell is `initial → epoch 0 → epoch 34` for **train / validation**. “Selected” is the lowest validation checkpoint across attempts. Large training improvements paired with rising held-out scores show the fit overfits the received rows before the 35-epoch cap; most deployed models therefore remain at initialization.

| Variant / rows | Attempt 1: train / validation | Attempt 2: train / validation | Selected validation checkpoint |
|---|---|---|---|
| Hard inputs / 20k | `41.686→38.908→11.861 / 38.805→39.990→68.022` | `41.686→37.355→21.972 / 38.805→38.632→80.679` | Epoch 0, 38.632 |
| Categorical p / 20k | `41.686→38.755→12.380 / 38.805→40.052→74.224` | `41.686→38.509→15.345 / 38.805→39.232→67.073` | Initial, 38.805 |
| Hard-transport NLL / 20k | `41.686→38.801→12.302 / 38.805→39.648→83.050` | `41.686→37.998→11.320 / 38.805→39.262→67.039` | Initial, 38.805 |
| Hard inputs / 100k | `43.883→40.453→15.533 / 40.067→40.613→86.920` | `43.883→41.880→32.444 / 40.067→41.909→135.730` | Initial, 40.067 |
| Categorical p / 100k | `43.883→41.389→31.570 / 40.067→41.273→134.083` | `43.883→40.503→29.721 / 40.067→40.527→117.512` | Initial, 40.067 |
| Hard-transport NLL / 100k | `43.883→41.794→18.061 / 40.067→41.776→81.740` | `43.883→43.486→21.857 / 40.067→43.579→86.398` | Initial, 40.067 |

The hard-input 20k case is the exception: one attempt improves validation at epoch 0 by 0.173 points, then degrades. Across the six candidates/twelve attempts, **419 of 420 epoch checkpoints score worse than their own initial checkpoint**; the sole better checkpoint is that hard-input 20k epoch 0. The other five candidates select initialization. The direct hard-transport objective does not resolve this train/validation divergence. Trace arrays, fit provenance and selected-prior epochs are retained in each case's `compression_records.csv`.

## Full-vector RD outcomes

Full-SI rANS is primary. Gaps use the matched seed-43 O curve and piecewise-linear equal-distortion comparison; positive means S needs more rate than O.

| Method | Sample size | Full-SI rANS gap vs O | Gap vs current-best S43 | Sample packet bytes | Final WMAPE |
|---|---:|---:|---:|---:|---:|
| Soft validated | 20k | +21.318% | +6.294% | 20,437 | 36.395226% |
| Hard inputs | 20k | +22.674% | +8.712% | 20,437 | 40.362839% |
| Categorical p | 20k | +21.318% | +6.294% | 20,437 | 36.395226% |
| Hard-transport NLL | 20k | +21.319% | +6.295% | 20,443 | 36.395226% |
| Soft validated | 100k | +32.619% | +14.622% | 92,594 | 39.013221% |
| Hard inputs | 100k | +30.938% | +13.685% | 92,596 | 36.395226% |
| Categorical p | 100k | +30.905% | +13.656% | 92,595 | 36.395226% |
| Hard-transport NLL | 100k | +30.879% | +13.661% | 92,598 | 36.395226% |

The full stage tables, directed pairwise matrices, ordinary-metadata proxy, model-distribution fields and exact prior provenance are in the [hard-transport report](../../_/experiments/s_received_fp16_validated_hard_transport_pilot_20261014/final_analysis/report.md). The [categorical report](../../_/experiments/s_received_fp16_validated_categorical_prior_pilot_20261014/final_analysis/report.md) and [hard-input report](../../_/experiments/s_received_fp16_validated_hard_inputs_pilot_retry_20261014/final_analysis/report.md) preserve the earlier controls. These are single-fit-seed, fixed-coordinate diagnostics; no multi-seed claim is inferred.

## Failure and repair record

- The earlier unvalidated sample-only fits showed extreme train/held-out mismatch and refit a prior on full-target rows. They are diagnostic failures, not deployable estimates; the validated variants here fit quantizer/prior weights only on received training rows and freeze the prior for full-target evaluation.
- The first validated-split plan stopped before a completed case because a singleton normalization stratum could not supply validation rows. The repaired split keeps singleton strata in training and reuses packet normalization factors for empty validation strata; the fresh retry completed both budgets.
- The first hard-input retry stopped after fitting 20k because a post-fit artifact path referenced the wrong quantizer field. The field was fixed, the focused record-path suite passed, and a fresh two-case retry completed.
- The categorical-prior compatibility check exposed a default-off prior hook passing `selection_metric=None` to legacy callbacks. The call now omits the keyword when unset; the legacy and sampled-plane suites passed on final source before launch.
- Hard-transport and categorical pilots completed without worker errors. Their RD issue is held-out selection and generalization, as shown in the checkpoint traces above, rather than a failed detached run.

The recurring issue is optimization/generalization, not evidence that 20k or 100k is intrinsically too small: each selected main checkpoint is initial or epoch 0, and held-out RD rises while the training RD falls. The next experiment should change the available adaptation capacity or scale before another loss variant; do not expand these checkpoints to five seeds as if they represented a learned solution.

## Best current spatial diagnostic

The best current spatial result is the five-seed receiver-side spatial-beta blend on the current-best MAE-prefix S curves. It leaves samples, symbols, prior, packet and rates fixed, improves mean WMAPE by 0.559 percentage points at stage 1 and 0.338 at stage 2, and changes stage 3 by 0.013 points. Its full-SI rANS gap versus O is **+6.109% all five** and **+7.482% best three**, versus +7.017% and +8.210% for the unmodified current-best S curve. The seed-43-only gap is +8.180% versus O.

This is a post-decoder use of received spatial neighbors, not a trained quantizer improvement. The beta fit uses received sample labels; validation labels remain in the neighbor context for predictions at other coordinates, so the validation partition is not independent of all context values. Every point's own sample label is excluded from its prediction, and raw target labels enter only after beta freeze for whole-vector scoring. See the [five-seed spatial report](../../_/experiments/s_setup_spatial_refiner_20261014/five_seed_spatial_rd_report.md) and [seed-43 provenance audit](../../_/experiments/s_setup_spatial_refiner_20261014/whole_vector_rd_report.md).

There is a small prior-access provenance difference in the current-best S diagnostic: mean sample-only joint mixed-mask CE is 0.763593 bpf, while the full-target-refitted server-SI prior CE is 0.759565 bpf, a −0.004028 bpf shift. The corresponding measured full-SI rANS rate is 0.769666 bpf. This is the observed conditional-CE effect of the refit/access difference; it is not the much larger 0.23 bpf figure from comparing incompatible all-coordinate rates. The full-target-refitted CE remains a provenance-confounded diagnostic, not the deployable rate estimate.
