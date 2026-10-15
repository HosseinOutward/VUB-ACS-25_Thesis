# Dedicated acquisition refiner: multi-seed S versus O

Plan: [s_dedicated_refiner_five_seed_20261010.json](plans/s_dedicated_refiner_five_seed_20261010.json).

**Completed 10 October 2026: all 30 cases, zero symbol errors.** The run took about
2 hours 11 minutes. All 20 nonzero-refiner fits preserved the starting prefix bit-for-bit;
sample coordinates were matched across counts for every seed. Every measured final
three-stage distortion curve was monotone; no failed stages were removed from the raw results.

| Extra planes | All-five rate change versus O | Best-three rate change versus O | Mean sample WMAPE % | Mean sample rANS bpf | Retained S seeds |
|---:|---:|---:|---:|---:|---|
| 0 | +13.489% | +13.831% | 36.399 | 0.251364 | 44, 45, 46 |
| 1 | +12.739% | +13.192% | 36.244 | 0.263838 | 43, 44, 45 |
| 2 | +14.240% | +14.973% | 36.755 | 0.261379 | 43, 44, 45 |
| 3 | +12.378% | +12.536% | 35.703 | 0.264595 | 43, 44, 47 |
| 6 | +16.600% | +15.313% | 36.164 | 0.264663 | 43, 44, 45 |

Rate changes compare mean full-SI rANS curves at equal distortion; positive favors O.
Sample means use all five seeds. Acquisition rates are normalized by the full gradient
size and are already charged in the final RD curves. O retains seeds 43, 46 and 47;
each method drops its own two worst seeds, rather than dropping matched seeds together.

Three extra planes is the closest S variant to O in both cohorts, but the sample WMAPE
improvement over zero extra planes is only 0.697 percentage points. This training setup
keeps acquisition-rate growth modest while leaving the samples far from SO's zero-error
labels. Six extra planes offers no further mean sample improvement. These conclusions
apply to the specified baseline-only refiner fit and fixed-gradient diagnostic.

[Complete result report](../../_/experiments/s_dedicated_refiner_five_seed_20261010/final_analysis/report.md)
contains full directed rate-change matrices, all staged RD measurements, every seed
selection and acquisition diagnostics. [RD curves](../../_/experiments/s_dedicated_refiner_five_seed_20261010/final_analysis/rd.png).

This fixed-gradient diagnostic compares O with S using 0, 1, 2, 3 and 6 independent
refinement planes, each with seeds 43–47: 30 measured cases. The retained SVHN/ResNet18
client-0 round-5 gradient, seed-43 starting model and 150 server SI columns stay fixed.
The trace is imported without regeneration. There is one trained round, one client;
this does not measure a trained-plus-nine-frozen protocol window.

O starts from the retained model's three-plane prefix and uses its ordinary warm fit.
For every nonzero S refinement count, a new secondary encoder, decoder and prior are
fitted on the same **baseline client-1 gradient** that originally trained the retained
starting model. Its ordinary three-plane weights are frozen and checked bit-for-bit
after fitting. Acquisition-refiner training follows the existing warm candidate selection
with two accepted fits and these overrides: 28 epochs, LR 0.0006807383623518464,
initial tau 0.30, batch 7500, hard decoder inputs. The current client-0 raw gradient
never enters this refiner fit. Availability and distribution of the starting model and
refiner retain the study's original O model-training/distribution assumption.

The refiner encoder receives the raw value and the ordinary prefix symbols. Its decoder
receives those symbols plus the usual SI. Its prior uses the full decoded prefix and
earlier secondary symbols, with quantization and entropy coding kept separate.

S acquires 3.6M coordinates using the retained three-plane model and its fitted refiner,
then fits **only three ordinary planes** on the received reconstructions. Final sample
fits use the same optimizer overrides with the existing soft-input policy. Zero extra
planes uses the retained prefix directly. Every S variant uses the same payload-owned
coordinate seed and index hash within each fit seed, regardless of its network size.

Acquisition and ordinary symbols use full-SI rANS. All acquisition symbol bytes are
charged from the first RD stage. Coordinate/normalization metadata and model payloads
are reported separately through payload sizes, rather than silently added to symbol-rate
curves. Ordinary full-SI prior fitting follows the existing separate warm schedule;
inactive refiner priors are retained but not refitted. Original acquisition weights,
joint/fine-tuned weights, transmitted payloads, sample coordinates/reconstructions and
identity-checked records are saved for each case. The final three-plane fit has the same
stage objective across refinement counts.

Report all-five means alongside the best three of five seeds selected independently
for O and each S count by mean full-SI rANS rate on that method's shared distortion span.
Retain raw measurements and identify dropped/unrankable seeds. Average cumulative rates
and raw WMAPE before conversion to dB and RD comparison. Publish full directed matrices
at shared distortion, native/full-SI rate tables and sample distortion/cost diagnostics.
No client exclusions are applied; coverage is one client per fitted seed.

The worker runs sequentially on the RTX 3090 (`cuda:0`; host GPU numbering differs),
with about 26 GiB host RAM available at launch. The detached supervisor records health
at 1, 5 and 30 minutes and every 90 minutes thereafter. Its live plotter stays enabled.

```bash
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch \
  Experiments/Protocol_test/plans/s_dedicated_refiner_five_seed_20261010.json \
  --worker-module Experiments.Protocol_test.sampled_plane_sweep \
  --plotter-module Experiments.Protocol_test.sampled_plane_sweep \
  --plan-module Experiments.Protocol_test.sampled_plane_sweep
```

Unit: `fl-wz-s-dedicated-refiner-five-seed-20261010`.

- [Live report](../../_/experiments/s_dedicated_refiner_five_seed_20261010/analysis/report.md)
- [Final report](../../_/experiments/s_dedicated_refiner_five_seed_20261010/final_analysis/report.md)
- [Live RD plot](../../_/experiments/s_dedicated_refiner_five_seed_20261010/analysis/rd.png)
- [Runner log](../../_/experiments/s_dedicated_refiner_five_seed_20261010_control/runner.log)
- [Health checks](../../_/experiments/s_dedicated_refiner_five_seed_20261010_control/health_checks.jsonl)

Validation: 69 model/codec/sweep tests and 25 training tests passed in separate processes.
