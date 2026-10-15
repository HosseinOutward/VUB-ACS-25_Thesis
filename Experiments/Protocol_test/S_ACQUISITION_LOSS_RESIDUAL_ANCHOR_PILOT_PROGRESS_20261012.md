# Acquisition-loss and residual-anchor pilot

The fresh seed-43 acquisition-loss and residual-anchor pilot completed through the detached supervisor. Its six cases were O, S_e0, current_e1_mse, current_e1_mae, residual_e1_mse, and residual_e1_mae. All S cases used 3,600,000 sampled coordinates, one shared deterministic coordinate pool, the same 140-epoch acquisition-refiner schedule, and the same 28-epoch final S fit. The four S_e1 modes varied only acquisition-refiner anchoring and distortion objective. The MAE option is a normalized-space absolute-error surrogate scaled by 2; it is not the exact raw WMAPE objective. Final replacement S fitting remains MSE.

O and S_e0 are fresh controls against the same target, fit seed, and 150 decoder-visible SI columns. The primary result is each S curve's equal-distortion full-vector full-SI rANS rate change against this fresh O/seed43 curve, with sample symbols and metadata charged in the cumulative rates. Sample WMAPE is reported as a diagnostic. The generated report includes directed rate-gap matrices, stage RD tables, sample metrics, and curve plots.

The target and retained starting model come from `frozen_starting_svhn_seed43_rtx3090_20261007`, whose trace ran on `cuda:0` / GeForce RTX 3090. The pilot reuses that fixed target and generates no trace. The local PyTorch mapping is `cuda:0` = physical GPU 1, GeForce RTX 3090; the plan keeps `fit_device=cuda:0`.

Plan: `plans/s_acquisition_loss_residual_anchor_pilot_20261012.json`\
Output: `_/experiments/s_acquisition_loss_residual_anchor_pilot_20261012`\
Control and scheduled health logs: `_/experiments/s_acquisition_loss_residual_anchor_pilot_20261012_control`\
Actual systemd unit: `fl-wz-s-acquisition-loss-residual-anchor-pilot-2026101`\
Stop command, if needed: `systemctl --user stop fl-wz-s-acquisition-loss-residual-anchor-pilot-2026101`

Launch command:

```bash
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch \
  Experiments/Protocol_test/plans/s_acquisition_loss_residual_anchor_pilot_20261012.json \
  --worker-module Experiments.Protocol_test.sampled_plane_sweep \
  --plotter-module Experiments.Protocol_test.sampled_plane_sweep \
  --plan-module Experiments.Protocol_test.sampled_plane_sweep
```

The first launch attempt omitted these module flags and failed before creating a service because the default sparse replay plan loader expected `fl_config`; the corrected command above started the service. The completed six-case report is at `_/experiments/s_acquisition_loss_residual_anchor_pilot_20261012/final_analysis/report.md`. It reports current_e1_mse at +12.1551% full-vector/full-SI rate gap versus fresh O/seed43; current_e1_mae is unrankable because its final-S stage curve is nonmonotone; residual MSE/MAE are worse than current MSE on both primary gap and sample WMAPE. The 1.8M/2.4M budget follow-up below is intended to measure sample-cost tradeoffs for current_e1_mse.

## Diagnostic: pruned-stage MAE comparison

The primary report keeps the ordinary current_e1_mae curve unrankable because its stage-2 WMAPE (58.0537%) is worse than stage 1 (55.2521%), while cumulative rate rises from 0.4980 to 0.6221 bits/float. Do not reorder stages, drop the point in the primary comparison, or describe this curve as a win. A separate dominated-stage-pruned diagnostic that removes only the stage-2 point, retaining its cumulative rate semantics on the remaining points, gives +9.1382% versus O and +11.5043% for current_e1_mse on the same shared span [-8.520415, -5.153031] WMAPE dB; pruned MAE versus MSE is -1.9726% on their shared span [-8.664644, -5.153031]. This is exploratory and does not alter the full-curve RD metric or ranking. Sample WMAPE is 33.4692% for current MAE versus 35.1152% for current MSE.

## Current-refiner sample-budget pilot

This follow-up holds current_e1_mse acquisition/refiner and final-S schedules fixed, comparing 1,800,000 and 2,400,000 samples against a fresh O/seed43 control. The previous completed 3,600,000-sample current_e1_mse case remains a report-only reference; it is not rerun. Every candidate gets the same coordinate seed, but each requested size uses an independent `numpy.random.default_rng(seed).choice(..., replace=False)` draw, so the subsets are not guaranteed to be nested. The primary remains the full-vector/full-SI rANS RD gap with all sample acquisition and metadata charged; sample WMAPE remains FYI.

Plan: `plans/s_budget_pilot_20261012.json`\
Output: `_/experiments/s_budget_pilot_20261012`\
Control and scheduled health logs: `_/experiments/s_budget_pilot_20261012_control`\
Systemd unit: `fl-wz-s-budget-pilot-20261012`

Launch command:

```bash
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch \
  Experiments/Protocol_test/plans/s_budget_pilot_20261012.json \
  --worker-module Experiments.Protocol_test.sampled_plane_sweep \
  --plotter-module Experiments.Protocol_test.sampled_plane_sweep \
  --plan-module Experiments.Protocol_test.sampled_plane_sweep
```

The three cases are O_seed43, current_e1_mse_n1800000_seed43, and current_e1_mse_n2400000_seed43. The plan uses `cuda:0` / GeForce RTX 3090, one sequential worker, and the existing live plotter. The focused sweep tests passed (26 passed), and independent review cleared the budget-mode identity, sample-count routing, and fresh O control before launch. The detached runner completed all 3/3 cases with return code 0, no error markers, a live plotter, and no symbol errors.

### Budget pilot result

The fresh O control matches the previous O/seed43 RD metrics and starting-state hash. Both budget candidates have the same acquisition-state hash as the completed 3.6M current_e1_mse reference, so their acquisition/refiner weights match exactly. All three S cases use coordinate seed `494155588`; each budget's sampled-index hash differs, confirming same-seed independent draws rather than nested sets.

| Method | Sample count | Stage WMAPE % | Cumulative full-vector/full-SI rANS b/float | Primary gap vs fresh O (%) | Direct gap vs 3.6M (%) | Sample WMAPE % | Sample prior/rANS b/float |
|---|---:|---|---|---:|---:|---:|---|
| O | — | 71.9726 / 55.7776 / 37.4955 | 0.248873 / 0.419259 / 0.706208 | 0.000 | — | — | — |
| current_e1_mse_n1800000 | 1,800,000 | 63.1905 / 55.1923 / 38.3444 | 0.416518 / 0.538789 / 0.784217 | +19.844 | +5.552 | 35.1593 | 0.126575 / 0.131124 |
| current_e1_mse_n2400000 | 2,400,000 | 60.2615 / 55.0778 / 38.6316 | 0.470930 / 0.553560 / 0.756835 | +19.432 | +5.029 | 35.1721 | 0.168772 / 0.173361 |
| current_e1_mse (reference) | 3,600,000 | 57.3348 / 50.3801 / 36.8780 | 0.488672 / 0.582709 / 0.739549 | +12.155 | 0.000 | 35.1152 | 0.253369 / 0.258044 |

Each candidate-vs-O and candidate-vs-3.6M gap uses its own pairwise shared WMAPE-dB interval, with no extrapolation: S1.8/O [-8.325968, -3.986960], S2.4/O [-8.261145, -4.399208], S3.6/O [-8.520415, -4.831636], S1.8/S2.4 [-8.261145, -4.399208], S1.8/S3.6 [-8.325968, -4.831636], and S2.4/S3.6 [-8.261145, -4.831636]. The full directed square matrix, row versus column, negative favoring the row:

```text
              O      S1.8      S2.4      S3.6
O         +0.000   -16.558   -16.270   -10.838
S1.8     +19.844    +0.000    +0.177    +5.552
S2.4     +19.432    -0.177    +0.000    +5.029
S3.6     +12.155    -5.260    -4.788    +0.000
```

The four curves share the narrower WMAPE-dB interval [-8.261145, -4.831636]. On that common interval, the secondary gap versus O is +33.460% for S1.8, +39.316% for S2.4, and +22.390% for S3.6. Sample rANS overhead is 0.005882 b/full-gradient-float for each case, and sample metadata is 0.000960 MiB for the two budgets and 0.000961 MiB for 3.6M; these costs are already included in each S curve's cumulative full-vector rates. Both reduced budgets retain sample WMAPE close to the 3.6M diagnostic, but their direct whole-vector RD gaps versus 3.6M are worse on the stated shared intervals. The smaller sample budgets therefore do not improve the primary RD tradeoff.

## Received-prefix final-S pilot (running)

This pilot changes final S fitting to use the actually received ordinary prefix bins, with the final-S encoder frozen. It includes an MSE frozen re-encode control to isolate the fit-input path from encoder freezing, and an MAE received-bin case. Acquisition is held invariant across final-fit variants (reencode-teacher acquisition pilot with encoder trainable); only the explicitly named acquisition distortion can change. All three cases use the 3.6M sample, seed 43, 140 acquisition-refiner epochs, and 28 final-S epochs. The re-encode control regenerates bins from decoded teacher values available with SI; those fit bins can differ from deployment bins, so this is a fit-path ablation, not a side-channel method. Final evaluation always uses deployed symbols and charges the same acquired sample payload.

Plan: `plans/s_received_bins_pilot_20261012.json`\
Output: `_/experiments/s_received_bins_pilot_20261012`\
Control and scheduled health logs: `_/experiments/s_received_bins_pilot_20261012_control`\
Systemd unit: `fl-wz-s-received-bins-pilot-20261012`\
Launched: 2026-10-12 03:55 UTC (05:55 CEST), sequentially on `cuda:0` / GeForce RTX 3090 with the live plotter.

Launch command:

```bash
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch \
  Experiments/Protocol_test/plans/s_received_bins_pilot_20261012.json \
  --worker-module Experiments.Protocol_test.sampled_plane_sweep \
  --plotter-module Experiments.Protocol_test.sampled_plane_sweep \
  --plan-module Experiments.Protocol_test.sampled_plane_sweep
```

The three fresh cases are `current_e1_mse_received_prefix_bins_seed43`, `current_e1_mae_received_prefix_bins_seed43`, and `current_e1_mse_reencode_teacher_frozen_control_seed43`. Built-in summary has no O case by design. Postprocessing must join the named read-only references from `_/experiments/s_acquisition_loss_residual_anchor_pilot_20261012/cases/{O,current_e1_mse,current_e1_mae}_seed43/target/complete.json` and the new case records, first checking target hash, starting-model hash, SI hash, fit seed, acquisition state hash (MSE received-bin versus frozen re-encode), sample coordinate/index hashes, sample count, and protocol settings. Reuse is report-only; no fit state or sample packet is reused. Keep full curves primary, mark any nonmonotone curve unrankable, and report sample WMAPE only as FYI. The separate report must show full directed pairwise gaps and stage/rate costs, with every reused reference named explicitly.

The acquisition-invariance regression verifies pilot state, payload, and reported rate across final fit mode/encoder-freeze settings; the worker regression verifies exact MSE acquisition-state parity across the paired final-S cases. Focused checks passed: 86 tests across `test_sampled_plane_sweep.py` and `test_sampled_codec.py`. Independent review cleared the acquisition boundary and fixture-only deterministic hash check. The first launch command without module flags failed in plan loading before creating a service; the corrected command above started the named unit.

### Received-prefix pilot result

All three cases finished sequentially with zero symbol errors; the detached unit exited after case 3. The merged report is `_/experiments/s_received_bins_pilot_20261012/final_analysis/merged_reference_comparison.md`. It includes all stage WMAPE and dB values, marginal/joint/SI-prior/full-SI rANS stage costs, sample costs, the full 6×6 directed matrix, pairwise spans, and a valid-curves common-span comparison. The generated report can be reproduced with:

```bash
.venv/bin/python -m Experiments.Protocol_test.postprocess_received_prefix_study pilot
```

The seed-43 candidate acquisition/control checks are exact: received-prefix MSE and frozen re-encode MSE match the prior current-e1 MSE acquisition-state, coordinate seed, index hash, sample WMAPE, sample prior, and sample rANS. Received-prefix MAE matches the prior current-e1 MAE for the same fields. Target, SI, retained starting weights, source manifest, and numerical runtime match across the pilot references. No control fit, weights, or sample packet were reused.

| Method | Full-vector/full-SI WMAPE % | Cumulative full-SI rANS b/float | Sample WMAPE % | Sample prior/rANS b/float | Primary gap vs O % |
|---|---|---|---:|---:|---:|
| O | 71.9726 / 55.7776 / 37.4955 | 0.248873 / 0.419259 / 0.706208 | — | — | 0.000 |
| Prior current-e1 MSE | 57.3348 / 50.3801 / 36.8780 | 0.488672 / 0.582709 / 0.739549 | 35.1152 | 0.253369 / 0.258044 | +12.155 |
| New received-bin MSE | 56.2059 / 49.4673 / 36.0485 | 0.505250 / 0.583514 / 0.766180 | 35.1152 | 0.253369 / 0.258044 | +11.775 |
| New received-bin MAE | 54.7957 / 47.9578 / 34.6499 | 0.510694 / 0.588958 / 0.771625 | 33.4692 | 0.259032 / 0.263488 | +8.947 |
| Frozen re-encode MSE control | 55.9970 / 49.5792 / 35.9257 | 0.505441 / 0.583891 / 0.766672 | 35.1152 | 0.253369 / 0.258044 | +11.735 |

The old current-e1 MAE comparison remains unrankable in the primary matrix because its stage-2 WMAPE rises to 58.0537%. The new received-bin MAE curve is monotone and lowers the full-vector/full-SI gap versus O by about 3.21 percentage points relative to the prior MSE case; on the common span of all valid curves [-8.520415, -5.225069] dB it is +8.947% versus O, versus +11.301% for new MSE and +11.350% for old MSE. New MAE is -2.159% versus old current-e1 MSE and -2.116% versus new received-bin MSE on this common interval. Its sample WMAPE is 1.646 percentage points lower than MSE; this is diagnostic only, while the primary evidence is the full-vector RD curve. The frozen re-encode MSE control is essentially unchanged from received-bin MSE (+0.006% direct pairwise gap), suggesting that received-bin fitting alone does not explain the MAE result.

## Received-prefix MAE five-seed confirmation (prepared)

The fresh confirmation plan is `plans/s_received_mae_five_seed_20261012.json`, output `_/experiments/s_received_mae_five_seed_20261012`, with one candidate (`current_e1_mae_received_prefix_bins`) at seeds 43–47, 3.6M samples, 140 acquisition / 28 final-S epochs, and best-three retention. It uses the same cuda:0 / RTX 3090 device and full live-plot supervisor. The matched O and current-e1 MSE records are read-only references from `_/experiments/s_dedicated_refiner_coldfit_five_seed_20261011`; they will not be refit. The reusable cold-fit and pilot manifests match in reference, baseline, source, target, SI, numerical runtime, and fit schedules. Their seed43 O and S_e1 records match the first pilot exactly, including whole-vector stage metrics, MSE acquisition hash, sample coordinate seed, and sample-index hash. Independent plan review cleared the provenance and selection procedure.

The durable merged postprocessor is `Experiments/Protocol_test/postprocess_received_prefix_study.py`. After the confirmation unit completes, run:

```bash
.venv/bin/python -m Experiments.Protocol_test.postprocess_received_prefix_study confirmation
```

It checks the matched manifests and seed43 controls, then reports all-five and separately selected best-three cohorts. Best-three selection is per method against the all-five O mean full-SI rANS curve on the shared monotone WMAPE-dB span; it averages cumulative rates and raw WMAPE before dB conversion and recomputes the complete directed matrix. The plan launched after the three-case pilot completed, seed43 acquisition/sample parity passed, and independent review cleared the comparison. Launch time: 2026-10-12 04:24 UTC (06:24 CEST). Unit: `fl-wz-s-received-mae-five-seed-20261012`; all five fits run sequentially on cuda:0 with the live plotter. Use scheduled supervisor health checks and case-completion events. After all five cases complete, the merged report is written to `_/experiments/s_received_mae_five_seed_20261012/final_analysis/merged_confirmation.md`.

### Five-seed confirmation result

The confirmation unit `fl-wz-s-received-mae-five-seed-20261012` finished 5/5 at 2026-10-12 07:03 CEST (return code 0, no errors, zero symbol errors). The merged all-five and retained-best-three report is `_/experiments/s_received_mae_five_seed_20261012/final_analysis/merged_confirmation.md`; it contains full directed matrices and per-stage raw WMAPE/full-SI rANS/SI-prior/joint-prior/marginal tables for both cohorts, seed selection, and per-seed plus cohort sample cost diagnostics. Reproduce it with:

```bash
.venv/bin/python -m Experiments.Protocol_test.postprocess_received_prefix_study confirmation
```

All five candidate curves, all five O controls, and all five current-e1 MSE controls are monotone; no seed was excluded for primary RD. Exactly three seeds are retained independently per method: O 43/46/47, current-e1 MSE 43/44/46, and received-bin MAE 44/45/47. Selection follows the established per-method gap against the all-five O mean full-SI rANS curve, with no extrapolation. Means average cumulative rates and raw WMAPE before conversion to dB. The fresh candidate seed43 record exactly reproduces the one-seed pilot, including acquisition state, sample coordinates/indices, sample diagnostics, and full-vector stage curve.

All-five mean curves give +7.017% full-vector/full-SI rANS gap for received-bin MAE versus O, versus +15.277% for current-e1 MSE; received-bin MAE is -6.137% versus current-e1 MSE on its pairwise shared distortion span. The separately selected best-three mean curves give +8.210% versus O and -3.530% versus current-e1 MSE. These values come from the complete merged directed matrices; the all-five and best-three stage tables retain the underlying rates and WMAPE means.

Sample WMAPE remains FYI: candidate mean is 33.628755% across all five and 33.772364% for the retained three. All-five sample prior/rANS means are 0.257181/0.261625 b/full-gradient-float; retained-three means are 0.255901/0.260308. Sample overhead averages 0.005882 b/float and metadata 0.000961 MiB. These costs are charged in the primary cumulative full-vector rate curves.
