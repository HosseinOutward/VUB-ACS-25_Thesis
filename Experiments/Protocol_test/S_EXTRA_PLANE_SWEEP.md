# Practical S extra-plane sweep

**Complete: all 30 fits finished successfully on 9 October 2026.** All native/sample rANS
decodes succeeded. The corrected analysis retains three seeds independently for O and every
S variant; no quantizer or prior was refitted during this analysis.

| Extra planes | All-five rate change versus O | Best-three rate change versus O | Retained S seeds |
|---:|---:|---:|---|
| 0 | +16.471% | +14.308% | 43, 44, 47 |
| 1 | +15.071% | +11.612% | 44, 46, 47 |
| 2 | +21.533% | +20.655% | 44, 45, 47 |
| 3 | +48.523% | +48.820% | 44, 45, 46 |
| 6 | +125.817% | +124.079% | 43, 45, 47 |

These are full-SI rANS equal-distortion changes; positive favors O. O's retained seeds are
43, 46 and 47. One extra plane gives the closest S curve, but O remains better. More planes
improve sample and final distortion while their charged acquisition cost worsens RD here.

The original global-span analysis could not trim: the 30 individual curves do not share a
distortion interval. The corrected analysis ranks each method's own seeds on the interval
shared by that method's valid curves, and compares the resulting mean curves over each pair's
shared distortion interval. There is no extrapolation. Seed 45 has nonmonotone distortion
for S_e0 and S_e1; those runs remain in the untrimmed results and are excluded from best-three
selection. Raw fits, source snapshots and original reports are preserved.

[Complete corrected report, full matrices and stage RD tables](../../_/experiments/s_extra_planes_five_seed_20261009_r2/trimmed_analysis/report.md)
and [RD plot](../../_/experiments/s_extra_planes_five_seed_20261009_r2/trimmed_analysis/rd.png).

The original run stopped after O/seed 43 when a post-run check incorrectly assumed sample
coordinates used the first RNG draw. Pilot model construction consumes draws before sampling.
The corrected r2 run validates payload-owned coordinates and matches their hashes across
variants; the original result and failed-run logs remain preserved. A worker-level regression
test covers saving the real acquired samples and resuming completed cases.

The detached study compares real S/SR against a freshly fitted O for each fit seed 43–47.
S acquires 3.6M LWZQ-reconstructed samples with 3 ordinary planes plus 0, 1, 2, 3 or 6 extra
planes. Both the sample symbols and ordinary symbols use full-SI rANS. Acquisition cost is
included from the first RD stage; sample and refinement columns must not be added again.

All 30 fits reuse the retained SVHN client-0/round-5 target, 150 decoder-visible SI columns,
and seed-43 nine-plane starting model. This is a fixed-gradient, one-trained-round diagnostic;
the seeds vary fitting and sample coordinates, not the FL trace or starting weights. No client
exclusions or nine-frozen-round measurements are claimed. The existing trace device is preserved
because the trace is imported without regeneration. Fitting runs sequentially on the RTX 3090
(PyTorch `cuda:0`, physical GPU 1), with approximately 27 GiB host RAM available at launch.

S uses the existing S08 settings: 28 epochs, LR 0.0006807383623518464, tau 0.30 and batch 7500.
All ordinary and extra planes participate in its training objective; the deployed prefix has
three planes. O retains its ordinary warm schedule. Quantizer fitting and fixed-symbol prior
fitting remain separate. Model distribution and prior availability retain O's assumptions.
Symbol rates exclude normalization/coordinate metadata and model payloads; saved payload size
includes those. Full-SI rANS is the primary comparison, with native transport reported separately.

After completion, every method independently retains its best three of five seeds, dropping
its two worst runs over the distortion interval shared by that method's valid seeds. Every selected/dropped seed
and score is listed. Untrimmed five-seed means remain visible. Nonmonotone/unrankable curves
are explicitly flagged; a method with insufficient rankable seeds has no trimmed result.
Cumulative rates and raw WMAPE are averaged before dB conversion and `rd_curve_gap`.
Both matrix directions are recomputed; negative favors the row. Separate seed selection is
intentional and does not represent a matched-seed population estimate.

Plan: [s_extra_planes_five_seed_20261009_r2.json](plans/s_extra_planes_five_seed_20261009_r2.json).

Launch:

```bash
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch \
  Experiments/Protocol_test/plans/s_extra_planes_five_seed_20261009_r2.json \
  --worker-module Experiments.Protocol_test.sampled_plane_sweep \
  --plotter-module Experiments.Protocol_test.sampled_plane_sweep \
  --plan-module Experiments.Protocol_test.sampled_plane_sweep
```

Unit: `fl-wz-s-extra-planes-five-seed-20261009-r2`.

The supervisor records health at 1, 5 and 30 minutes, then every 90 minutes. Its live plotter
refreshes changed measurements in disposable processes, preserving the previous good report
after a plotting failure, and exits with the worker. The worker publishes the final report
automatically after validating complete matched seed coverage.

- [Live report](../../_/experiments/s_extra_planes_five_seed_20261009_r2/analysis/report.md)
- [Final corrected report](../../_/experiments/s_extra_planes_five_seed_20261009_r2/trimmed_analysis/report.md)
- [Original global-span report](../../_/experiments/s_extra_planes_five_seed_20261009_r2/final_analysis/report.md)
- [Live RD plot](../../_/experiments/s_extra_planes_five_seed_20261009_r2/analysis/rd.png)
- [Runner log](../../_/experiments/s_extra_planes_five_seed_20261009_r2_control/runner.log)
- [Scheduled health checks](../../_/experiments/s_extra_planes_five_seed_20261009_r2_control/health_checks.jsonl)

Each completed fit retains its weights, actual payload, CSV record and identity-checked completion
file. Final analysis also retains all raw records and selection/aggregate points in JSON.
Do not launch this plan again while its service is active. Stop only the specific unit if needed:
`systemctl --user stop fl-wz-s-extra-planes-five-seed-20261009-r2`.

Validation: 45 focused tests passed, covering practical sample coding/decoding, charged
acquisition, extra-plane counts, frozen handoff, independent O/S trimming, stage anomalies and
complete report publication.
