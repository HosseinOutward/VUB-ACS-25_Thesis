# Received-prefix raw-domain MAE acquisition pilot

This three-seed pilot compares raw-domain weighted-MAE acquisition with the completed normalized-MAE received-prefix fit. Seeds 43, 44, and 45 were predeclared and are all reported without seed trimming. The objective-only comparison is raw MAE versus normalized MAE; the O and old MSE curves are end-to-end context. Old MSE used reencode-teacher final fitting with a trainable encoder, so it is not a causal objective-only control.

The raw-MAE objective uses only the baseline fit target and its preprocessing scales. For each active plane its distortion term is

```text
(2 * reconst_ld / max(mean(abs(baseline_raw_target)), float32.tiny))
    * mean_i(scale_i * mean_coordinate(abs(xnorm_hat_i - xnorm_i)))
```

The unchanged KL rate is added, and active-plane losses are averaged. `scale_i` comes from the same baseline target's `PreprocessMetadata.norm_factors`. Positive uniform changes in raw units cancel between numerator and denominator. Current-target labels and current-target-derived weights are not used to train the acquisition refiner. Outlier/nonlinear target preprocessing is rejected for this objective.

## Plan and execution

- Plan: `plans/s_received_raw_mae_pilot_20261013.json`
- Output: `_/experiments/s_received_raw_mae_pilot_20261013`
- Detached control/logs: `_/experiments/s_received_raw_mae_pilot_20261013_control`
- Unit: `fl-wz-s-received-raw-mae-pilot-20261013`
- Started: 2026-10-13 04:10:46 CEST
- Device: `cuda:0`, NVIDIA GeForce RTX 3090; worker is sequential and the live plotter is enabled.
- Cases: `current_e1_raw_mae_received_prefix_bins_seed43`, `current_e1_raw_mae_received_prefix_bins_seed44`, and `current_e1_raw_mae_received_prefix_bins_seed45`.

The plan holds the received-prefix confirmation settings fixed: 3.6M acquired coordinates, 140 acquisition-refiner epochs, 28 final-S epochs, received ordinary prefix bins, and a frozen final-S encoder. The sample coordinate seed/index hash must match the normalized-MAE case at each seed. The saved source target hash must match the prior MSE and normalized-MAE fits and differ from the held-out current-target hash. The new run fits only its raw-MAE cases; no baseline, O, MSE, or normalized-MAE references are refit or reused as training states.

Matched read-only references are O and `S_e1` from `s_dedicated_refiner_coldfit_five_seed_20261011`, and `current_e1_mae_received_prefix_bins` from `s_received_mae_five_seed_20261012`. The report averages cumulative rates and raw WMAPE across the fixed three-seed cohort before dB conversion. It includes the full directed full-SI rANS matrix, stage tables, per-seed raw-MAE comparisons, sample WMAPE/rANS/overhead/metadata diagnostics, and monotonicity/symbol-error flags. All acquired sample costs remain included in the primary full-vector rates.

The durable comparison report is `_/experiments/s_received_raw_mae_pilot_20261013/final_analysis/matched_rd_comparison.md`. Recreate it with:

```bash
.venv/bin/python -m Experiments.Protocol_test.postprocess_received_raw_mae_pilot
```

The detached unit finished all 3 cases successfully with zero symbol errors and no error markers. The 5-minute health check recorded 3/3 completed, runner exit code 0, no memory pressure, and the RTX 3090 released after completion. All final-S curves were monotone.

The full directed cohort matrix below is full-vector/full-SI rANS, row versus column, negative favoring the row:

```text
                     O       S_e1  S_received_MAE  S_received_raw_MAE
O                +0.000    -12.345       -5.368          -8.449
S_e1            +14.084     +0.000       +6.457          +3.163
S_received_MAE   +5.672     -6.065       +0.000          -2.956
S_received_raw_MAE +9.229   -3.066       +3.046          +0.000
```

The objective-only result is raw MAE **+3.046%** versus normalized MAE over their mean curves; recomputed reverse is **−2.956%**. Raw MAE loses to normalized MAE in every paired seed: +5.247% (43), +3.182% (44), and +0.735% (45). All gaps compare full-vector/full-SI rates with acquired sample cost included.

Raw-MAE mean stage WMAPE is 54.235336/47.976297/34.561572%; cumulative full-SI rANS is 0.530389/0.608515/0.791174 b/float. Normalized-MAE mean stage WMAPE is 54.550328/47.981152/34.724486%; cumulative full-SI rANS is 0.509250/0.587461/0.770122 b/float. Old S_e1/MSE is context only because it used a trainable final-S encoder and reencode-teacher fitting.

Sample WMAPE is FYI: raw-MAE 32.400665% versus normalized-MAE 33.602171%. Sample prior/rANS is 0.278664/0.283245 versus 0.257737/0.262041 b/float; overhead is 0.005882 b/float for both. The sample error improves while its acquisition rate increases, so the whole-vector RD result does not improve. The detailed seed sample costs, stage tables, paired reverse gaps, and distortion spans are in the merged report.

The launch command was:

```bash
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch \
  Experiments/Protocol_test/plans/s_received_raw_mae_pilot_20261013.json \
  --worker-module Experiments.Protocol_test.sampled_plane_sweep \
  --plotter-module Experiments.Protocol_test.sampled_plane_sweep \
  --plan-module Experiments.Protocol_test.sampled_plane_sweep
```

Protocol/schema and report checks passed: 31 sweep plan, trimming, disjoint-span, and report tests passed; the new postprocessor unit test passed. The codec owner reported 61 sampled-codec tests passed. One earlier schema-test failure used a stale draft plan with top-level raw-MAE and `extra_planes=[0]`; the final plan keeps the established top-level MSE default and selects raw MAE only through the explicit variant, and the final plan/schema checks pass. Independent review cleared the loss boundary and matched comparisons before launch.

Read progress from `_/experiments/s_received_raw_mae_pilot_20261013_control/{runner.log,health_checks.jsonl}` and case completion files under the output root. The launch-time check confirmed the previous received-MAE unit was inactive and both GPUs were idle before this unit started.
