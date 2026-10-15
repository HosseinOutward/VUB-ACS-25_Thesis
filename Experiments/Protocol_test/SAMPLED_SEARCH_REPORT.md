# S/SR machinery and completed search record

Consolidated on 7 October 2026 before deleting the dedicated sampled-study workers,
plans, result trees, captured models/arrays, caches and temporary diagnostics. This report
retains the tested settings, setups and measured conclusions. The original study scripts and raw
artifacts named below are historical identifiers; they are no longer retained or executable.
Separate protocol/supervisor work and the history-distortion study remain, as requested.
The reconstructed F and matched 8 October follow-up sections describe new retained workers,
plans and artifacts; the historical cleanup statement does not apply to those outputs.

## Current received-FP16 follow-up (14 October 2026)

The current practical-S follow-up uses seed 43, the same target/baseline and O read-only reference,
received FP16 labels, fixed coordinates and a deterministic received-only train/validation split.
The [validated-fit progress report](S_RECEIVED_FP16_VALIDATED_PILOT_PROGRESS_20261014.md) records
the train/validation traces, execution incidents and current matched results. The full
[hard-transport NLL report](../../_/experiments/s_received_fp16_validated_hard_transport_pilot_20261014/final_analysis/report.md)
compares O, current-best S43, and soft-input, hard-input, categorical-p, and hard-transport fits at
20k/100k samples. Every small-sample main fit selected its initial state or epoch 0; training RD
fell while held-out RD rose sharply. Their O gaps remain about **+21% at 20k** and **+31% at 100k**,
so these candidates are not multi-seed or deployable winners. Sample packets were actually measured,
with decoded sample values and transmitted normalization factors checked across variants.

The best current spatial result is a separate post-decoder blend of received spatial neighbors,
applied to the five-seed current-best MAE-prefix S curves. It reduces the full-SI rANS gap versus O
from **+7.017% to +6.109% all-five** and from **+8.210% to +7.482% best-three**, while leaving
symbols, rates, priors and packet unchanged. Its validation uses received sample neighbors, including
other held-out sample labels as context; this context dependency is disclosed in the progress report.
It is not a learned-quantizer result.

For the current-best S diagnostic, the mean sample-only joint mixed-mask CE is 0.763593 bpf versus 0.759565 bpf for the full-target-refitted server-SI prior (−0.004028 bpf); measured full-SI rANS is 0.769666 bpf. This small difference is the observed prior-access provenance confound. Do not substitute the incompatible all-coordinate 0.23 bpf comparison for this quantity, and do not treat the full-target-refitted CE as deployable.

## Retained implementation

`Experiments/Protocol_test/sampled_codec.py` registers S and SR. S inherits O's training
process and substitutes received, reconstructed sampled rows; SR adds existing rANS machinery.
The protocol supplies the preceding deployed codec and controls when training/frozen rounds
occur. No codec-specific round numbers are prescribed. The complete gradient is quantized after
sample fitting; sampled decoder outputs are replaced by the earlier acquisition reconstruction.
Extra trained heads remain in the machine, with only the ordinary prefix deployed. Real
full-gradient normalization metadata is used for both sample and complete-vector processing.

Acquisition uses the preceding deployed LWZQ weights. Without a preceding model, an explicit
cold bootstrap trains on the last decoder-visible history vector. It does not bootstrap on the
raw current target. Previous machines must have enough compatible trained heads. Weight transfer
uses the existing O assumption; serialized payload/model/metadata size remains separate from
symbol-stream RD.

For N coordinates, a sampled set I, n ordinary planes and e refinement planes, symbol cost is
`[new-model n-plane bits outside I + old-model (n+e)-plane bits on I] / N`.
Equivalently, use the mixed old/new n-plane base and add the sampled fraction's e-plane cost.
Since acquisition and final quantization use different model weights, the sampled old prefix
replaces the discarded new prefix too. Adding only the old tail to an unchanged new full-vector
rate would mischarge that prefix. `sample_extra_prior_rate` separately records weighted extra
planes; SR also records actual sample/refinement stream rates and overhead. All acquisition
planes are needed before training and their costs enter the first recorded stage. Recorded
stage rates already include them; sample cost columns must not be added a second time.

Current sampling defaults (9 October, S08): **3,600,000 coordinates**, **one extra plane**, LWZQ acquisition,
all trained planes in the sample objective, 16 sample-rANS lanes, no added label noise,
soft training, encoder/prior/backbones trainable. Sample-only optimizer defaults are the
S08 setting: **28 epochs / LR 0.0006807383623518464 / initial tau 0.30 / batch 7,500**.
Explicit `None` restores the inherited value for each field, including batch size.
The expanded comparison completed all 20 seeds per configuration before the stop request;
its workers and plotter exited and released memory. S08 versus S04 is **−0.621704%**
for all twenty mean curves and **−1.127571%** for the independently selected best twelve.
The all-twenty paired bootstrap 95% interval is **[−2.411654%, +1.223641%]**.
Complete raw tables and validation are in [the expanded report](S04_S08_EXPANDED_SEED_REPORT.md).
Acquisition and the later entropy-prior optimizer keep their own settings. Prefix training,
FP16 acquisition, capacity-freezing and hard-input options remain opt-in S/SR machinery.
The original experiment-only privileged raw-sample subclass was removed; the 8 October
follow-up rebuilds its sampling rule in a separate measurement worker. The numerical-policy fix remains
committed as `aaebce6` on both master and the sampled branch. The initial cherry-pick
`be12b69` was replaced by the original commit after verifying identical committed trees;
all pending work was preserved.

## Common static-gradient setup

All primary searches use a fixed **SVHN / ResNet18** FL trace, trace seed **43**, target
**round 5 / client 0**, baseline **client 1** in that round, and **11,181,642 floats**.
The protocol history builder supplies five preceding raw rounds for all 30 clients:
**150 decoder-side SI columns**, frozen at the current round's start, with no current target
in SI. This is a single-gradient diagnostic, rather than a population result or a window
of one trained plus nine frozen rounds. Repeated fit seeds change optimization and sample
selection, not the gradient, trace, baseline client or baseline weights. No clients are excluded.

The baseline is cold-trained on client 1 using the same SI, then each case independently
warm-starts from its weights. The first grid uses a six-plane baseline; the expanded grid
uses nine planes. Subsequent ablations import the identical nine-plane baseline, target and
SI bytes. A fresh three-plane full-target O is the matched control in precision-fixed repeats.
All ordinary stages are binary, with hidden size **128** and **four layers**.

Inherited cold setup: **140 epochs**, LR **0.0028364098431326932**, up to **550,000 rows/epoch**,
batch **6,250**, three accepted candidates. Inherited warm setup: **35 epochs**, LR
**0.0008509229529398079** (cold LR × 0.3), reconstruction weight **72.31601255189986**,
initial tau **0.5** annealed toward **0.1**, two accepted candidates. The candidate limit is
**eight attempts**; failures are flagged rather than deployed. The ordinary entropy prior is
then fitted again on complete-target symbols and decoder SI: **30 epochs / three attempts**.
It does not merely reuse the prior from joint quantizer training. Actual full-SI rANS after
this fit is the primary rate; joint/pre-refit prior and refitted-prior cross-entropy are separate.
Normal rANS uses 2,048 lanes. Native and full-SI rANS coincide for these server-SI codecs;
temporal rates are unavailable unless actually measured.

In the practical acquired-sample studies, the inherited epoch budget is capped by available
sample rows: `min(m, 550000)`. The corrected raw-label studies hold **550,000 nominal draws
per epoch**, batch 6,250, using uniform tiling of a fixed selected pool. Duplicate rows are
allowed. Pool coordinates stay fixed within a fit; the pool is not replaced each epoch.
Larger acquisition counts and optimizer draws are distinct quantities.

## F: starting-model control without target adaptation

F evaluates the same cross-client starting machine on the current target, deploying only
its first three binary planes. Keep the encoder, decoder and conditional-prior weights
unchanged: no target quantizer training and no post-symbol prior fitting. Use the current
target's normal normalization metadata and the same 150 decoder-visible SI columns as O/S;
updating inputs and metadata is inference, not weight adaptation. There is no sample
acquisition stream. Evaluate the frozen prior on the new target symbols and SI, and measure
actual full-SI rANS plus all three reconstruction stages. The frozen prior's cross-entropy
belongs in a frozen-prior column, not a refitted-prior column.

| Control | Quantizer adaptation on target | Separate prior adaptation on target | Starting weights |
|---|---|---|---|
| F | None | None | Unchanged three-plane prefix of the common baseline |
| O | Full raw target | Full-target symbols and decoder SI | Independently initialized from the common baseline |
| S | Acquired samples; exact raw samples in the oracle studies | Full-target symbols and decoder SI | Independently initialized from the common baseline |

**Historical F measurements are unavailable.** The completed searches did not retain a
full-target F curve, and cleanup deleted the exact baseline weights, target/SI caches and
per-run measurements. Historical acquisition errors on sampled coordinates and client-1
training metrics are not measurements of F on client 0. Rebuilding the baseline produces
a new starting model; its F, O and S curves must be measured together and labeled as a fresh
matched study before claiming a numerical starting-point comparison. The historical
baseline's producer precision is unknown, so rebuilding cannot be assumed to reproduce
its exact weights.

### Reconstructed F measurement (7 October 2026)

The follow-up reconstructs one nine-plane starting model using baseline seed **43**, the
documented client-1/round-5 target, cold-training defaults and the same raw-history protocol
setup. Cold initial tau is **5.5**; **0.5** is the warm-training initial tau. Both trace
generation and codec fitting use the **RTX 3090** (currently logical `cuda:0`), with explicit
TF32/high matmul precision; codec AMP is disabled. This is a new seed-43 reconstruction,
not a claim to recover the deleted historical weights. F has no fine-tuning seed sweep:
without adaptation, one fixed starting model gives one deterministic F curve. Historical
O/S tables remain the original three-seed medians.

Worker: `Experiments/Protocol_test/frozen_starting_model.py`.
Plan: `Experiments/Protocol_test/plans/frozen_starting_svhn_seed43_rtx3090_20261007.json`.
Output: `_/experiments/frozen_starting_svhn_seed43_rtx3090_20261007`.
The detached worker and live plotter both completed with exit code **0**. The starting
weights, target/SI cache, source snapshot, raw record and live RD plot are retained at this
output. Cold starting-model fit: **1,575.09 seconds**. F encoding/decoding: **51.21 seconds**.
Target checks reproduce the retained **62.127%** zero fraction and **0.00380619** absolute
99th percentile. This checks recorded trace statistics, not bitwise historical identity.

**F* below is this reconstructed baseline**, not a three-seed median. Frozen-prior CE is
**0.765840173 bpf**; actual full-SI/native rANS is
**0.771314982 bpf**, including **0.005881784 bpf**
fixed stream overhead. Quantizer/prior fits on client 0: **0 / 0**. Symbol errors: **0**.
All encoder, decoder and prior weights are byte-identical before and after F inference.
No stage has native rANS growth below 0.01 or above 1.05 bpf; distortion decreases at
every stage. Stage marginal rates were not measured, so no unmeasured symbol-stage
diagnostic is inferred. Coverage: one target gradient, one starting seed, no exclusions;
with/without client exclusions is identical. This remains a single-gradient diagnostic.

| Stage | Marginal | Frozen joint prior | Temporal prior | Refitted full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | — | 0.366520 | — | — | 0.372292 | — | 0.372292 | -3.093789 (70.034263%) |
| 2 | — | 0.490296 | — | — | 0.496130 | — | 0.496130 | -4.631368 (58.672093%) |
| 3 | 1.784724 | 0.765840 | — | — | 0.771315 | — | 0.771315 | -8.779112 (36.395226%) |

**Entries marked `~` are approximate contextual comparisons:** F* uses rebuilt starting
weights; O/S/X remain historical medians, and only their rounded retained stage points
are available. Both comparison directions are recomputed with `rd_metrics.rd_curve_gap`.
These entries are not new matched fine-tuning trials or paired-seed medians. The two
historical stage tables have different rounding precision, so their approximate F*/O
gaps can differ slightly. In F* rows, the joint-prior column evaluates the frozen starting
prior; the general full-SI prior column records its frozen final total. Refitted-prior
columns remain `—`. No fine-tuned prior rate is attributed to F*.

RD comparisons use `Experiments/Protocol_test/rd_metrics.py`: piecewise-linear cumulative
rate against `20*log10(raw_WMAPE/100)` over the shared distortion span, without extrapolation.
Both directions are recomputed; the reported symmetric change is `(S_vs_O - O_vs_S)/2`.
Negative favors S. Equal-rate relative WMAPE change is a separate diagnostic. Median curves
use componentwise medians of cumulative rates and raw WMAPE before conversion to dB; paired
seed-gap medians instead compare each S with its fresh same-seed O, then take the median.
These are different summaries. Missing/nonmonotone/nonoverlapping measurements are `—`.

## Expanded S04/S08 seed comparison (9 October 2026)

A paired extension adds fit seeds 48–62 to S04 and S08, retaining their original five
runs for **20 seeds each**. Thirty new fits use the same starting model, target, SI,
CUDA device and fit schedules; only initial tau differs (.20 versus .30).
[**S04_S08_EXPANDED_SEED_REPORT.md**](S04_S08_EXPANDED_SEED_REPORT.md) records execution
status and will receive all-twenty/fresh-fifteen averages, best-three/best-twelve views,
paired seed differences and bootstrap intervals automatically.

## Nearby parameter grid (8 October 2026)

The follow-up varies S initial tau **0.20/0.25/0.30**, LR **0.8/1.0/1.2×**
the retained center, epochs **28/35/42**, raw sample count **2.4M/3M/3.6M**, and
training batch size **5,000/6,250/7,500**. The 43 configurations comprise all 32
low/high combinations, ten one-axis changes about the center, and the center itself.
Each uses matched fit seeds 43–47 and the retained starting model/target/SI on cuda:0.
Five center fits were reused; **210 new fits completed** across three memory-bounded
workers. The supervisor and live plotter both exited with code 0. Sample coordinates stay identical within each seed/count pair; different
counts have different pools. Separate entropy-prior settings are unchanged.
The explicit center reproduced the retained raw-S model and metrics bitwise exactly.

Completed raw tables and parameter-interaction plots are in
[**S_LOCAL_GRID_REPORT.md**](S_LOCAL_GRID_REPORT.md). Comparisons with original O, T and F,
including all-five and independently selected best-three means, are in
[**S_LOCAL_GRID_COMPARISON_REPORT.md**](S_LOCAL_GRID_COMPARISON_REPORT.md). All-five means drive the parameter
interaction analysis; independently selected best-three means remain visible.

## Five-seed validation and seed audit (8 October 2026)

**Completed: twenty matched fits (five per method), plus one independent S43 repeat.**
Eight new fits extend the retained twelve 43–45 measurements with seeds 46 and 47.
Three detached workers and their live plotters all exited with code **0**. Plans:
`plans/five_seed_os_extension_20261008.json`, `plans/five_seed_ot_extension_20261008.json`,
and `plans/seed_repeat_s43_20261008.json`. Each output uses the same plan stem under
`_/experiments/`. All fits use the retained F reference on RTX 3090 / cuda:0. The
supervisor's measured five-minute checks had roughly 9 GB RAM headroom, full comparison-GPU
utilization and no errors. Timing includes GPU contention and is not a throughput benchmark.

As requested, select each codec's best three independently. Retained seeds are **O:
43/46/47; O_tau025: 44/45/46; S_tau025 and T: 43/44/45**. Discarded fits remain in the
individual-run tables and the untrimmed five-seed mean. The common selection interval is
**52.483861–63.612634% raw WMAPE** (−5.599485 to −3.929132 dB). Selection covers the
same measured span for every seed and does not extrapolate. Client exclusions are separate:
none, with **1/1 client** retained. This is still a single fixed-gradient diagnostic.

Full-SI S/O gaps: **+1.041111%** for the selected-three mean, **−0.181201%** for the
all-five mean (negative favors S). Same-seed S beats original O in **2/5** fits; the added
seeds 46 and 47 favor O by **+6.169733%** and **+3.618556%**. With the same arithmetic
averaging rule, original seeds 43–45 give **−3.056441%**, while added seeds 46–47 give
**+4.261074%**. The earlier approximately 2.5% S/O median advantage is therefore seed
sensitive and is not retained by the requested best-three comparison. S/O_tau025 remains
**−3.142258%** selected-three and **−2.935658%** all-five, with **4/5** same-seed wins.
These are comparisons of mean curves, not averages of individual gap percentages.

### Seed audit and measurement validation

No seed-handling defect was demonstrated. Python, NumPy, CPU Torch and both CUDA RNGs
reset correctly. All **15,000,000** S coordinates reproduce exactly from the five fit
seeds. Each method has five distinct joint and native-prior tensor hashes, with one fixed
initial-state hash. T's initial SI projection is validated directly against the retained
baseline. All cumulative stage totals agree with their scalar rates and the final-stage
WMAPE agrees with delivered reconstruction WMAPE. Prior fitting leaves the quantizer
weights exactly unchanged. All 24 archived FL source files match across F and the six
comparison/repeat outputs; the new workers share identical raw comparison-runner code.

The full-size S43 repeat reproduces **sample indices, joint weights, fitted-prior weights,
all stage rates and raw WMAPE bitwise exactly**, despite a different preceding case and
concurrent GPU work. It is an audit control, not a sixth seed or a cohort observation.
CPU quantizer/prior probes also reproduce after intervening seeds and constructor randomness.
Local seed-zero generators are confined to normalization/deduplication and do not reset
training randomness. Native symbol errors are zero in all twenty fits and the repeat;
no stage monotonicity, stream-growth or full-target prior/marginal failures were flagged.
Per-stage marginal entropies were not measured, so symbol-stage death is not inferred.

**41 tests passed** for S/SR defaults, explicit overrides/inheritance, codec behavior and
aggregation/ranking arithmetic. On 8 October the practical S default used the historical
winner's **3M samples / 35 epochs / LR 0.0008509229529398079 / initial tau 0.25**; its
acquired reconstructed labels and rate accounting remain distinct from this raw-S control.
The current S08 defaults above supersede that configuration.
Explicit `None` restores an inherited sample-fit optimizer value.

Reproducible audit and complete raw tables/CSV/JSON/PNG/PDF:
`_/experiments/five_seed_svhn_20261008_analysis`. Key files: `seed_audit.md`,
`seed_audit_probe.py/json`, `gpu_repeat_validation.json`, `final_validation.py`,
`validation.json`, `selection.csv/json`, `mean_stage_points.csv/json`,
`comparison_gaps.csv`, `paired_seed_gaps.csv`, and `seed_cohort_means.json`.
Analysis worker: `Experiments/Protocol_test/five_seed_tables.py`; its exact source is also
retained as `analysis_source.py`. Numerical policy is unchanged; the bitwise repeat is
an observed result for this setup, not a general GPU determinism guarantee.

Fit seeds **43–47**; original completed seeds 43–45 retained, seeds 46–47 newly fitted.
Starting weights, SVHN client-0 round-5 target, all 150 SI columns and numerical policy stay fixed at baseline seed 43.
T uses its five temporal columns for its quantizer/native coder and all 150 for the separately refitted comparison prior.
O_tau025 retains the matched S-optimizer control. S_tau025 is the privileged raw-label diagnostic; practical S still acquires reconstructed labels.
This measures fit/sample randomness on one fixed gradient, not FL-trace or starting-weight seed robustness.

Rank each codec's runs by mean full-SI rANS at equal distortion over the **same** interval [-5.599485, -3.929132] WMAPE dB.
Lower is better. Equivalently rank the RD gap against fixed F* on that interval. Discard the two largest scores independently per codec; ties by seed.
These are intentionally selected best-three means; the untrimmed five-seed means and all paired seeds remain visible.
Average cumulative stage rates and raw WMAPE first, then convert WMAPE to dB and compare aggregate curves.
Do not average per-seed gap percentages. Each matrix uses the pair's complete shared distortion span; the fixed common span applies only to selection.

| Codec | Seed | Common-span full-SI mean bpf | Common-span gap vs F* % | Included in best three |
|---|---:|---:|---:|---|
| O | 43 | 0.396194274 | -21.210926 | yes |
| O | 44 | 0.456442657 | -9.229647 | discarded |
| O | 45 | 0.440024994 | -12.494541 | discarded |
| O | 46 | 0.407873129 | -18.888414 | yes |
| O | 47 | 0.424775980 | -15.527033 | yes |
| O_tau025 | 43 | 0.446048240 | -11.296730 | discarded |
| O_tau025 | 44 | 0.427843634 | -14.916984 | yes |
| O_tau025 | 45 | 0.432418719 | -14.007161 | yes |
| O_tau025 | 46 | 0.428474534 | -14.791521 | yes |
| O_tau025 | 47 | 0.459574033 | -8.606927 | discarded |
| S_tau025 | 43 | 0.430635005 | -14.361879 | yes |
| S_tau025 | 44 | 0.403422542 | -19.773479 | yes |
| S_tau025 | 45 | 0.423512338 | -15.778326 | yes |
| S_tau025 | 46 | 0.440236738 | -12.452433 | discarded |
| S_tau025 | 47 | 0.442492266 | -12.003888 | discarded |
| T | 43 | 0.476415011 | -5.257850 | yes |
| T | 44 | 0.466266650 | -7.276001 | yes |
| T | 45 | 0.429658039 | -14.556163 | yes |
| T | 46 | 0.480654847 | -4.414696 | discarded |
| T | 47 | 0.477038787 | -5.133803 | discarded |

### Best three of five: mean stage RD

Cumulative bits/float. Marginal is a final-stage scalar total; — is unmeasured. F* has no prior refit.

| Method | Stage | Marginal | Joint/pre-fit prior | Temporal prior | Full-SI fitted prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O | 1 | — | 0.292310 | — | 0.285901 | 0.290949 | — | 0.290949 | -3.255679 (68.741036%) |
| O | 2 | — | 0.413419 | — | 0.393929 | 0.399170 | — | 0.399170 | -4.618776 (58.757216%) |
| O | 3 | 1.481469 | 0.677713 | — | 0.642499 | 0.648230 | — | 0.648230 | -7.762201 (40.915694%) |
| O_tau025 | 1 | — | 0.380266 | — | 0.322475 | 0.327664 | — | 0.327664 | -3.344545 (68.041320%) |
| O_tau025 | 2 | — | 0.471331 | — | 0.399461 | 0.404825 | — | 0.404825 | -4.328896 (60.751248%) |
| O_tau025 | 3 | 1.142834 | 0.645685 | — | 0.565205 | 0.570918 | — | 0.570918 | -6.714338 (46.161837%) |
| F* | 1 | — | 0.366520 | — | — | 0.372292 | — | 0.372292 | -3.093789 (70.034263%) |
| F* | 2 | — | 0.490296 | — | — | 0.496130 | — | 0.496130 | -4.631368 (58.672093%) |
| F* | 3 | 1.784724 | 0.765840 | — | — | 0.771315 | — | 0.771315 | -8.779112 (36.395226%) |
| S_tau025 | 1 | — | 0.295019 | — | 0.289155 | 0.294170 | — | 0.294170 | -3.179751 (69.344568%) |
| S_tau025 | 2 | — | 0.452438 | — | 0.373656 | 0.378816 | — | 0.378816 | -4.207564 (61.605829%) |
| S_tau025 | 3 | 1.445236 | 0.706036 | — | 0.606954 | 0.612598 | — | 0.612598 | -7.393678 (42.689011%) |
| T | 1 | — | 0.327238 | 0.323751 | 0.316303 | 0.328967 | 0.328967 | 0.321501 | -2.810242 (72.358227%) |
| T | 2 | — | 0.450529 | 0.436613 | 0.417841 | 0.442075 | 0.442075 | 0.423300 | -4.317785 (60.829013%) |
| T | 3 | 1.485475 | 0.731827 | 0.711754 | 0.681534 | 0.717872 | 0.717872 | 0.687827 | -7.862216 (40.447268%) |

Equal-distortion rate changes (%), row versus column; negative favors row; reverse entries recomputed.
Client exclusions: none; with/without symmetric whole-window client exclusions is identical, 1/1 client retained.
Fit coverage: 3/5 separately selected per codec; fixed F* (one model). One static round, no frozen evaluation window.

Full-SI rANS (primary):

```text
                     O   O_tau025         F*   S_tau025          T
          O     +0.000     -4.570    -14.819     -1.030     -8.463
   O_tau025     +4.789     +0.000    -12.852     +3.244     -5.251
         F*    +17.397    +14.748     +0.000    +17.332     +7.393
   S_tau025     +1.041     -3.142    -14.772     +0.000     -8.019
          T     +9.245     +5.542     -6.884     +8.718     +0.000
```

Native transport (T uses temporal SI):

```text
                     O   O_tau025         F*   S_tau025          T
          O     +0.000     -4.570    -14.819     -1.030    -12.230
   O_tau025     +4.789     +0.000    -12.852     +3.244     -9.139
         F*    +17.397    +14.748     +0.000    +17.332     +3.001
   S_tau025     +1.041     -3.142    -14.772     +0.000    -11.783
          T    +13.934    +10.058     -2.914    +13.357     +0.000
```


### All five, untrimmed: mean stage RD

Cumulative bits/float. Marginal is a final-stage scalar total; — is unmeasured. F* has no prior refit.

| Method | Stage | Marginal | Joint/pre-fit prior | Temporal prior | Full-SI fitted prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O | 1 | — | 0.336600 | — | 0.315372 | 0.320513 | — | 0.320513 | -3.429478 (67.379236%) |
| O | 2 | — | 0.459648 | — | 0.424482 | 0.429816 | — | 0.429816 | -4.811072 (57.470689%) |
| O | 3 | 1.332573 | 0.672968 | — | 0.618361 | 0.624057 | — | 0.624057 | -7.344558 (42.931108%) |
| O_tau025 | 1 | — | 0.381203 | — | 0.337489 | 0.342799 | — | 0.342799 | -3.496751 (66.859399%) |
| O_tau025 | 2 | — | 0.475790 | — | 0.410570 | 0.416088 | — | 0.416088 | -4.381786 (60.382445%) |
| O_tau025 | 3 | 1.231164 | 0.650195 | — | 0.575914 | 0.581765 | — | 0.581765 | -6.720010 (46.131703%) |
| F* | 1 | — | 0.366520 | — | — | 0.372292 | — | 0.372292 | -3.093789 (70.034263%) |
| F* | 2 | — | 0.490296 | — | — | 0.496130 | — | 0.496130 | -4.631368 (58.672093%) |
| F* | 3 | 1.784724 | 0.765840 | — | — | 0.771315 | — | 0.771315 | -8.779112 (36.395226%) |
| S_tau025 | 1 | — | 0.321951 | — | 0.316098 | 0.321211 | — | 0.321211 | -3.393436 (67.659407%) |
| S_tau025 | 2 | — | 0.460705 | — | 0.404811 | 0.410101 | — | 0.410101 | -4.494210 (59.605935%) |
| S_tau025 | 3 | 1.425664 | 0.677659 | — | 0.603935 | 0.609626 | — | 0.609626 | -7.292433 (43.189517%) |
| T | 1 | — | 0.399784 | 0.342518 | 0.336041 | 0.347721 | 0.347721 | 0.341248 | -2.949642 (71.206219%) |
| T | 2 | — | 0.515497 | 0.446973 | 0.429640 | 0.452444 | 0.452444 | 0.435078 | -4.335929 (60.702074%) |
| T | 3 | 1.471649 | 0.771781 | 0.695519 | 0.666162 | 0.701567 | 0.701567 | 0.672300 | -7.633942 (41.524357%) |

Equal-distortion rate changes (%), row versus column; negative favors row; reverse entries recomputed.
Client exclusions: none; with/without symmetric whole-window client exclusions is identical, 1/1 client retained.
Fit coverage: 5/5 per codec; fixed F* (one model). One static round, no frozen evaluation window.

Full-SI rANS (primary):

```text
                     O   O_tau025         F*   S_tau025          T
          O     +0.000     -3.039    -12.925     +0.182     -7.386
   O_tau025     +3.134     +0.000    -11.206     +3.024     -5.087
         F*    +14.844    +12.620     +0.000    +15.219     +6.139
   S_tau025     -0.181     -2.936    -13.209     +0.000     -7.647
          T     +7.974     +5.360     -5.784     +8.280     +0.000
```

Native transport (T uses temporal SI):

```text
                     O   O_tau025         F*   S_tau025          T
          O     +0.000     -3.039    -12.925     +0.182    -10.993
   O_tau025     +3.134     +0.000    -11.206     +3.024     -8.746
         F*    +14.844    +12.620     +0.000    +15.219     +2.064
   S_tau025     -0.181     -2.936    -13.209     +0.000    -11.232
          T    +12.350     +9.584     -2.022    +12.653     +0.000
```

### Same-seed comparisons, all five

Pairwise full-SI curve gaps remain diagnostic; the aggregate result comes from averaged curves.

| Candidate | Reference | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Seed 47 | Wins | Best-three mean-curve gap % | All-five mean-curve gap % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| O_tau025 | O | +7.670517 | -4.659207 | -1.564142 | +4.214708 | +8.342351 | 2/5 | +4.789359 | +3.133918 |
| S_tau025 | O | +4.570148 | -9.963724 | -3.497871 | +6.169733 | +3.618556 | 2/5 | +1.041111 | -0.181201 |
| S_tau025 | O_tau025 | -3.676251 | -5.188801 | -3.474724 | +2.220431 | -3.757370 | 4/5 | -3.142258 | -2.935658 |
| T | O | +13.392703 | +3.326231 | -2.321965 | +17.348375 | +11.792650 | 1/5 | +9.245055 | +7.974494 |

### Added-seed diagnostic with the same averaging rule

This separates the additional seeds from the switch from historical medians to arithmetic means.
Each cell compares curves built from mean cumulative rates and mean raw WMAPE; it is not a mean of individual gaps.

| Candidate | Reference | Original seeds 43–45 mean-curve gap % | Added seeds 46–47 mean-curve gap % |
|---|---|---:|---:|
| S_tau025 | O | -3.056441 | +4.261074 |
| S_tau025 | O_tau025 | -4.033612 | -1.081134 |
| T | O | +5.161997 | +12.303471 |

### Individual run stage measurements

Every retained and discarded seed is included. Cumulative bpf; marginal final-stage only.

| Codec | Seed | Stage | Marginal | Joint prior | Temporal prior | Full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | Raw WMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O | 43 | 1 | — | 0.251859218 | — | 0.243858129 | 0.248872929 | — | 0.248872929 | 71.972638277 |
| O | 43 | 2 | — | 0.436843291 | — | 0.414023563 | 0.419259354 | — | 0.419259354 | 55.777552776 |
| O | 43 | 3 | 1.519805968 | 0.733658656 | — | 0.700295910 | 0.706208265 | — | 0.706208265 | 37.495508603 |
| O | 44 | 1 | — | 0.442519009 | — | 0.361378759 | 0.366501449 | — | 0.366501449 | 66.059389602 |
| O | 44 | 2 | — | 0.605076134 | — | 0.502810553 | 0.507976199 | — | 0.507976199 | 53.533272806 |
| O | 44 | 3 | 1.170208916 | 0.743691310 | — | 0.599013776 | 0.604181032 | — | 0.604181032 | 44.503628069 |
| O | 45 | 1 | — | 0.363549829 | — | 0.357779145 | 0.363216064 | — | 0.363216064 | 64.613683327 |
| O | 45 | 2 | — | 0.452904828 | — | 0.437813133 | 0.443593526 | — | 0.443593526 | 57.548522280 |
| O | 45 | 3 | 1.048249781 | 0.588008396 | — | 0.565294221 | 0.571411605 | — | 0.571411605 | 47.404829670 |
| O | 46 | 1 | — | 0.295239657 | — | 0.288890600 | 0.293995461 | — | 0.293995461 | 68.280693377 |
| O | 46 | 2 | — | 0.427300200 | — | 0.405635059 | 0.410995809 | — | 0.410995809 | 57.653061224 |
| O | 46 | 3 | 1.552451164 | 0.729105875 | — | 0.698800147 | 0.704817414 | — | 0.704817414 | 35.953980239 |
| O | 47 | 1 | — | 0.329831958 | — | 0.324955553 | 0.329977297 | — | 0.329977297 | 65.969776953 |
| O | 47 | 2 | — | 0.376113489 | — | 0.362129178 | 0.367254112 | — | 0.367254112 | 62.841033771 |
| O | 47 | 3 | 1.372149616 | 0.570373774 | — | 0.528401028 | 0.533665449 | — | 0.533665449 | 49.297594536 |
| O_tau025 | 43 | 1 | — | 0.355878800 | — | 0.353357673 | 0.358548414 | — | 0.358548414 | 64.662826973 |
| O_tau025 | 43 | 2 | — | 0.447240688 | — | 0.426559143 | 0.431862870 | — | 0.431862870 | 59.475175240 |
| O_tau025 | 43 | 3 | 1.709383637 | 0.694297872 | — | 0.662047632 | 0.667742358 | — | 0.667742358 | 40.205969370 |
| O_tau025 | 44 | 1 | — | 0.366170317 | — | 0.358769864 | 0.364204828 | — | 0.364204828 | 63.612633758 |
| O_tau025 | 44 | 2 | — | 0.509793714 | — | 0.488088399 | 0.494054809 | — | 0.494054809 | 52.280270240 |
| O_tau025 | 44 | 3 | 1.165070057 | 0.629251830 | — | 0.598527424 | 0.604500126 | — | 0.604500126 | 43.885009947 |
| O_tau025 | 45 | 1 | — | 0.480909079 | — | 0.333808154 | 0.339036431 | — | 0.339036431 | 68.893372218 |
| O_tau025 | 45 | 2 | — | 0.524415497 | — | 0.370753638 | 0.375959810 | — | 0.375959810 | 63.525431436 |
| O_tau025 | 45 | 3 | 0.876503430 | 0.643560428 | — | 0.484138191 | 0.489693374 | — | 0.489693374 | 52.483860521 |
| O_tau025 | 46 | 1 | — | 0.293718666 | — | 0.274846345 | 0.279750684 | — | 0.279750684 | 71.617953024 |
| O_tau025 | 46 | 2 | — | 0.379784413 | — | 0.339541145 | 0.344459606 | — | 0.344459606 | 66.448043153 |
| O_tau025 | 46 | 3 | 1.386929482 | 0.664243914 | — | 0.612949915 | 0.618561746 | — | 0.618561746 | 42.116640726 |
| O_tau025 | 47 | 1 | — | 0.409336299 | — | 0.366664290 | 0.372455494 | — | 0.372455494 | 65.510206614 |
| O_tau025 | 47 | 2 | — | 0.517714165 | — | 0.427906934 | 0.434103685 | — | 0.434103685 | 60.183304702 |
| O_tau025 | 47 | 3 | 1.017932281 | 0.619623393 | — | 0.521906387 | 0.528326698 | — | 0.528326698 | 51.967033318 |
| S_tau025 | 43 | 1 | — | 0.335716397 | — | 0.329693943 | 0.334946871 | — | 0.334946871 | 66.518997932 |
| S_tau025 | 43 | 2 | — | 0.602448881 | — | 0.406337745 | 0.411787106 | — | 0.411787106 | 59.852731157 |
| S_tau025 | 43 | 3 | 1.498782009 | 0.832965672 | — | 0.614682131 | 0.620494915 | — | 0.620494915 | 41.671929143 |
| S_tau025 | 44 | 1 | — | 0.283551097 | — | 0.277416766 | 0.282259081 | — | 0.282259081 | 69.152961434 |
| S_tau025 | 44 | 2 | — | 0.425917819 | — | 0.399175636 | 0.404137425 | — | 0.404137425 | 57.820201120 |
| S_tau025 | 44 | 3 | 1.580731213 | 0.715245023 | — | 0.679271720 | 0.684919084 | — | 0.684919084 | 37.096466588 |
| S_tau025 | 45 | 1 | — | 0.265790582 | — | 0.260353267 | 0.265304148 | — | 0.265304148 | 72.361743499 |
| S_tau025 | 45 | 2 | — | 0.328948744 | — | 0.315454207 | 0.320524660 | — | 0.320524660 | 67.144555769 |
| S_tau025 | 45 | 3 | 1.256194025 | 0.569898568 | — | 0.526908733 | 0.532379055 | — | 0.532379055 | 49.298637182 |
| S_tau025 | 46 | 1 | — | 0.358725131 | — | 0.354130566 | 0.359272457 | — | 0.359272457 | 65.550274600 |
| S_tau025 | 46 | 2 | — | 0.462045603 | — | 0.439501114 | 0.444731105 | — | 0.444731105 | 57.514473779 |
| S_tau025 | 46 | 3 | 1.668749869 | 0.667484544 | — | 0.623012520 | 0.628578164 | — | 0.628578164 | 42.021802185 |
| S_tau025 | 47 | 1 | — | 0.365973890 | — | 0.358897537 | 0.364272081 | — | 0.364272081 | 64.713059698 |
| S_tau025 | 47 | 2 | — | 0.484164529 | — | 0.463584065 | 0.469322842 | — | 0.469322842 | 55.697712292 |
| S_tau025 | 47 | 3 | 1.123862624 | 0.602699243 | — | 0.575799994 | 0.581758565 | — | 0.581758565 | 45.858750811 |
| T | 43 | 1 | — | 0.365237385 | 0.361717790 | 0.356196016 | 0.367030889 | 0.367030889 | 0.361450313 | 70.670909967 |
| T | 43 | 2 | — | 0.470498353 | 0.454961777 | 0.438558646 | 0.460631453 | 0.460631453 | 0.444045696 | 61.129410644 |
| T | 43 | 3 | 1.778716147 | 0.764601320 | 0.745362699 | 0.713125892 | 0.751724300 | 0.751724300 | 0.719646005 | 37.828302519 |
| T | 44 | 1 | — | 0.345455915 | 0.342410475 | 0.335444987 | 0.347759300 | 0.347759300 | 0.340717759 | 70.020058476 |
| T | 44 | 2 | — | 0.427475132 | 0.416589230 | 0.400965899 | 0.422079691 | 0.422079691 | 0.406321004 | 62.771953221 |
| T | 44 | 3 | 1.171905711 | 0.651254244 | 0.637731373 | 0.612114787 | 0.643614238 | 0.643614238 | 0.617767587 | 46.852197931 |
| T | 45 | 1 | — | 0.271021187 | 0.267124385 | 0.257268310 | 0.272111019 | 0.272111019 | 0.262336426 | 76.383713183 |
| T | 45 | 2 | — | 0.453614488 | 0.438286513 | 0.413997680 | 0.443514825 | 0.443514825 | 0.419532659 | 58.585676092 |
| T | 45 | 3 | 1.505803138 | 0.779626355 | 0.752168059 | 0.719362259 | 0.758277899 | 0.758277899 | 0.726066529 | 36.661303444 |
| T | 46 | 1 | — | 0.633924007 | 0.371235043 | 0.366719127 | 0.376513575 | 0.376513575 | 0.372064854 | 69.408443387 |
| T | 46 | 2 | — | 0.732127674 | 0.455887757 | 0.441956908 | 0.461514329 | 0.461514329 | 0.447382594 | 61.312211331 |
| T | 46 | 3 | 1.130225629 | 0.894875072 | 0.611419626 | 0.586855024 | 0.617449924 | 0.617449924 | 0.592624947 | 47.381245616 |
| T | 47 | 1 | — | 0.383281887 | 0.370103329 | 0.364577919 | 0.375191407 | 0.375191407 | 0.369670930 | 69.547967947 |
| T | 47 | 2 | — | 0.493768677 | 0.469141759 | 0.452722378 | 0.474479866 | 0.474479866 | 0.458108746 | 59.711117085 |
| T | 47 | 3 | 1.771595091 | 0.768548742 | 0.730914690 | 0.699349664 | 0.736768357 | 0.736768357 | 0.705392643 | 38.898736567 |

Stage diagnostics:

- O seed 43: monotone stages; no flagged rate growth/prior failure.
- O seed 44: monotone stages; no flagged rate growth/prior failure.
- O seed 45: monotone stages; no flagged rate growth/prior failure.
- S_tau025 seed 43: monotone stages; no flagged rate growth/prior failure.
- S_tau025 seed 44: monotone stages; no flagged rate growth/prior failure.
- S_tau025 seed 45: monotone stages; no flagged rate growth/prior failure.
- T seed 43: monotone stages; no flagged rate growth/prior failure.
- T seed 44: monotone stages; no flagged rate growth/prior failure.
- T seed 45: monotone stages; no flagged rate growth/prior failure.
- O_tau025 seed 43: monotone stages; no flagged rate growth/prior failure.
- O_tau025 seed 44: monotone stages; no flagged rate growth/prior failure.
- O_tau025 seed 45: monotone stages; no flagged rate growth/prior failure.
- O seed 46: monotone stages; no flagged rate growth/prior failure.
- O seed 47: monotone stages; no flagged rate growth/prior failure.
- S_tau025 seed 46: monotone stages; no flagged rate growth/prior failure.
- S_tau025 seed 47: monotone stages; no flagged rate growth/prior failure.
- O_tau025 seed 46: monotone stages; no flagged rate growth/prior failure.
- O_tau025 seed 47: monotone stages; no flagged rate growth/prior failure.
- T seed 46: monotone stages; no flagged rate growth/prior failure.
- T seed 47: monotone stages; no flagged rate growth/prior failure.

All native symbol errors are zero. Per-stage marginals were not measured; symbol-stage death is not inferred.

## Matched T and recommended raw-S follow-up (8 October 2026)

**Completed: nine fits, three per method.** Two detached workers reuse the retained reconstructed
F target, all 150 SI vectors and seed-43 nine-plane starting weights directly. They do not
regenerate the FL trace or cold-fit another starting machine. Fit seeds **43, 44, 45** are
paired across fresh O, T and S. All fits use the same RTX 3090 / `cuda:0`, TF32/high matmul
precision, AMP disabled, three deployed binary planes and 2,048 rANS lanes. Workers run
in parallel; cases within each worker run serially to bound RAM. This is still the fixed
client-0/round-5 gradient, not all-client trained-plus-nine-frozen coverage.

S uses the report's recommended **3M raw samples / 35 epochs / inherited warm LR /
initial tau 0.25**. This is the raw-label setting with a consistent paired advantage in the
historical report. The LR×0.3 setting was closer to historical O's aggregate median curve
(+0.26% versus −2.60% for tau 0.25), but had a positive paired median (+1.54%). The selected
S is the recommended setting, not an additional combined-factor setting or the practical
acquired-sample codec. Labels remain a zero-cost oracle; full-vector outputs are quantized
without exact sample replacement. Fixed sample pools and 550k nominal draws/epoch reproduce
the historical sampling rule. Pool selection and fitting vary with the fit seed.

T is the **standard five-column temporal variant**, not the historical 150-column padded
control. Its encoder and deployed heads retain the starting weights; its decoder and
conditional-prior input matrices keep the symbol columns and only client 0's five SI
columns. A numerical check verified that the projected starting decoder and prior agree
with the original architecture when the other 145 SI columns are zero. Quantizer fitting
uses only these five legitimate temporal columns. T's native entropy prior then fits on
fixed target symbols and temporal SI. A separate full-SI reference prior starts from the
three-plane prefix of the original full-SI joint prior and fits those same symbols with
all 150 SI columns. Both prior fits use **30 epochs / three accepted candidates**, with
the ordinary eight-attempt cap. Full-SI rANS is a server-side comparison; T actually
transports the temporal-prior stream. No additional fitting occurs when measuring stages.

O is rerun with the same seeds to make the rebuilt F/O/S/T comparison matched. The worker
records full stage priors, actual native/temporal/full-SI rANS, reconstruction errors,
training provenance and model weights. Final results take medians of cumulative rates
and raw WMAPE before converting to dB. Historical O/S rows stay historical.

Worker: `Experiments/Protocol_test/starting_model_comparison.py`.
Plans: `plans/matched_t_svhn_seed43base_20261008.json` and
`plans/matched_os_svhn_seed43base_20261008.json` under `Experiments/Protocol_test/`.
Outputs: `_/experiments/matched_t_svhn_seed43base_20261008` and
`_/experiments/matched_os_svhn_seed43base_20261008`.
The supervisor keeps both live plotters enabled and records scheduled health checks.


Both workers and their live plotters completed successfully (exit code **0**). All nine
native streams have zero symbol errors, all stages decrease distortion, and no measured
stage has native stream growth below 0.01 or above 1.05 bpf. No full-target joint prior
exceeds its measured marginal. Prior fitting leaves all encoder/decoder weights unchanged.
All 24 retained FL source files match across the F reconstruction and both new workers.
F and O/S share the exact retained three-plane starting tensors; their hash containers
are canonicalized explicitly because F recorded Torch's ordered dictionary.

Analysis worker: `Experiments/Protocol_test/starting_model_tables.py`.
Tables, raw CSV/JSON, historical reproduction check, and PNG/PDF curves:
`_/experiments/matched_starting_svhn_seed43base_20261008_analysis`.
Recorded timings include shared-GPU contention and stage-prior evaluation; encoding time
includes actual separate prior fitting. They are not isolated throughput benchmarks.

### Historical O/S mapping check

The new O and recommended S medians reproduce all retained historical joint-prior,
rANS and raw-WMAPE stage values at the original table's rounding precision: **18/18
stage metrics** (nine per method). The exact deleted weight bytes cannot be checked;
this establishes numerical reproduction of the retained tables, not bytewise historical
weight identity. The new F/O/S/T comparison itself uses the retained rebuilt starting
model and saved target/SI, and is fully matched under its stated T projection.

| Method | Historical final WMAPE % | New median WMAPE % | Historical final full-SI rANS | New median full-SI rANS | Historical final refitted prior | New median refitted prior |
|---|---:|---:|---:|---:|---:|---:|
| O | 44.50 | 44.503628 | 0.604181 | 0.604181 | 0.599014 | 0.599014 |
| S_tau025 | 41.67 | 41.671929 | 0.620495 | 0.620495 | 0.614682 | 0.614682 |

Historical headline RD gaps were symmetric two-direction summaries; the new square and
paired-seed tables below show directed `rd_curve_gap(row, column)` percentages. Their
percentage denominators differ. Recomputing the symmetric S/O summary gives **-2.600822%**
on the new median curves and **-3.561264%** as the median of the three symmetric paired gaps,
reproducing the report's **−2.60% / −3.56%** at retained precision.

### Full-target O with S's optimizer settings (8 October 2026)

**Completed: three fit seeds.** The previous report was committed as `c4947b5`
before this test. `O_tau025` uses the ordinary full-target O codec with the completed
recommended S run's effective quantizer optimizer: **35 epochs**, warm LR
**0.0008509229529398079**, initial **tau 0.25** annealed to 0.1, **550,000 nominal draws
per epoch**, batch **6,250**, reconstruction weight **72.31601255189986**, two accepted
candidates and the eight-attempt cap. The effective quantizer configuration was checked
field-by-field against the completed S record. The original O used quantizer tau 0.5.

This changes only O's quantizer temperature. O still trains from the complete target,
with no 3M pool restriction, acquisition or sample replacement. It uses the same saved
seed-43 starting tensors, target, full 150-column SI, RTX 3090, TF32/high policy and AMP-off
setting as the completed controls. The separate full-target prior fit retains the same
30 epochs, three accepted candidates, warm LR and prior temperature as the previous O/S
fits; S's quantizer-only override never changed those prior settings.

Fit seeds **43, 44, 45** are paired with the retained O and S runs. Primary comparisons
are `O_tau025` versus O (optimizer effect), and S versus `O_tau025` (remaining sampling
effect at matched optimizer settings), on medians of cumulative rates and raw WMAPE,
and separately on individually paired seed curves. The comparison remains a single
client-0/round-5 gradient, with no frozen-round or population claim.

Plan: `Experiments/Protocol_test/plans/matched_o_tau025_svhn_seed43base_20261008.json`.
Output: `_/experiments/matched_o_tau025_svhn_seed43base_20261008`.
Supervisor unit: `fl-wz-matched-o-tau025-svhn-seed43base-20261008`.

**Completed: three new O fits; twelve matched fits including retained controls.**
The new worker and live plotter both exited with code **0**. All new native streams have
zero symbol errors; all stages decrease distortion. No stream-growth or full-target
joint-prior/marginal failures were flagged. No clients were excluded (one target retained).
The original O/S/T records are reused without further fitting, and the new O control
shares their exact starting prefix, target, full SI and numerical policy.

On this fixed gradient, S retains its advantage after matching the optimizer settings:
S versus `O_tau025` is **−4.024879%** on the median full-SI rANS curves, with paired gaps
**−3.676251%, −5.188801%, −3.474724%** (three wins out of three), median **−3.676251%**.
The original S/O median-curve gap was **−2.567006%**. Applying tau 0.25 to full-target O
therefore does not account for that observed S advantage; the matched-settings result
supports a sampling-related effect in this single-gradient diagnostic.

`O_tau025` versus original O has a median-curve gap of **+1.953427%**, but its individual
paired gaps are **+7.670517%, −4.659207%, −1.564142%**, median **−1.564142%** (two wins
out of three). The difference between those summaries is retained rather than averaging
gap percentages or treating the median curve as a single trained machine. It does not
establish that tau 0.5 is generally optimal for O, or that sampling wins across clients.

All percentages above are directed equal-distortion rate changes; negative favors the
named row. Using the historical symmetric convention instead gives S/`O_tau025`
**-4.109274%** on median curves and **-3.746404%** as the paired-seed median.

Expanded raw tables, all five-method matrices, per-seed comparisons and PNG/PDF curves:
`_/experiments/matched_o_tau025_svhn_seed43base_20261008_analysis`.
The new worker runs serially; earlier control timings include concurrent GPU use, so the
mixed timing columns are not a matched throughput benchmark. The prior-fit checks confirm
that encoder/decoder weights stay fixed during the subsequent entropy-prior fitting.

### Matched median stage RD measurements

Cumulative bits/float; medians over fit seeds 43–45. F* is one fixed seed-43 starting model.
Fine-tuned priors are directly measured at every stage; marginal is measured only as a final total.
F*'s joint-prior column is frozen CE, and its refitted-prior columns are unavailable.

| Method | Stage | Marginal | Joint/pre-fit prior | Temporal prior | Refitted full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O | 1 | — | 0.363550 | — | 0.357779 | 0.363216 | — | 0.363216 | -3.601309 (66.059390%) |
| O | 2 | — | 0.452905 | — | 0.437813 | 0.443594 | — | 0.443594 | -5.070811 (55.777553%) |
| O | 3 | 1.170209 | 0.733659 | — | 0.599014 | 0.604181 | — | 0.604181 | -7.032092 (44.503628%) |
| O_tau025 | 1 | — | 0.366170 | — | 0.353358 | 0.358548 | — | 0.358548 | -3.786906 (64.662827%) |
| O_tau025 | 2 | — | 0.509794 | — | 0.426559 | 0.431863 | — | 0.431863 | -4.513285 (59.475175%) |
| O_tau025 | 3 | 1.165070 | 0.643560 | — | 0.598527 | 0.604500 | — | 0.604500 | -7.153676 (43.885010%) |
| F* | 1 | — | 0.366520 | — | — | 0.372292 | — | 0.372292 | -3.093789 (70.034263%) |
| F* | 2 | — | 0.490296 | — | — | 0.496130 | — | 0.496130 | -4.631368 (58.672093%) |
| F* | 3 | 1.784724 | 0.765840 | — | — | 0.771315 | — | 0.771315 | -8.779112 (36.395226%) |
| S_tau025 | 1 | — | 0.283551 | — | 0.277417 | 0.282259 | — | 0.282259 | -3.203784 (69.152961%) |
| S_tau025 | 2 | — | 0.425918 | — | 0.399176 | 0.404137 | — | 0.404137 | -4.458321 (59.852731%) |
| S_tau025 | 3 | 1.498782 | 0.715245 | — | 0.614682 | 0.620495 | — | 0.620495 | -7.603128 (41.671929%) |
| T | 1 | — | 0.345456 | 0.342410 | 0.335445 | 0.347759 | 0.347759 | 0.340718 | -3.015186 (70.670910%) |
| T | 2 | — | 0.453614 | 0.438287 | 0.413998 | 0.443515 | 0.443515 | 0.419533 | -4.274996 (61.129411%) |
| T | 3 | 1.505803 | 0.764601 | 0.745363 | 0.713126 | 0.751724 | 0.751724 | 0.719646 | -8.443663 (37.828303%) |

### Equal-distortion rate changes (%)

Row versus column; negative favors the row. Both directions are recomputed.
Coverage: one client/round, all three fit seeds for O/O_tau025/S_tau025/T; F* fixed. No clients excluded.
With/without symmetric whole-window client exclusions is identical (1/1 target retained).

Full-SI rANS, all measured clients (client 0):

```text
                     O   O_tau025         F*   S_tau025          T
          O     +0.000     -1.916    -12.444     +2.635     -4.619
   O_tau025     +1.953     +0.000    -10.656     +4.194     -2.820
         F*    +14.212    +11.926     +0.000    +17.233     +7.600
   S_tau025     -2.567     -4.025    -14.700     +0.000     -7.552
          T     +4.843     +2.902     -7.063     +8.169     +0.000
```

Full-SI rANS, with symmetric client exclusions (none):

```text
                     O   O_tau025         F*   S_tau025          T
          O     +0.000     -1.916    -12.444     +2.635     -4.619
   O_tau025     +1.953     +0.000    -10.656     +4.194     -2.820
         F*    +14.212    +11.926     +0.000    +17.233     +7.600
   S_tau025     -2.567     -4.025    -14.700     +0.000     -7.552
          T     +4.843     +2.902     -7.063     +8.169     +0.000
```

Native transport (T uses temporal SI), all clients and no-exclusion retained population:

```text
                     O   O_tau025         F*   S_tau025          T
          O     +0.000     -1.916    -12.444     +2.635     -9.246
   O_tau025     +1.953     +0.000    -10.656     +4.194     -7.553
         F*    +14.212    +11.926     +0.000    +17.233     +2.667
   S_tau025     -2.567     -4.025    -14.700     +0.000    -11.883
          T    +10.188     +8.170     -2.598    +13.486     +0.000
```

### Individual fit-seed raw metrics

| Method | Seed | Marginal | Joint prior | Temporal prior | Full-SI prior | Native rANS | Full-SI rANS | WMAPE % | Quantizer fit s | Prior-fit/encode s | Decode s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O | 43 | 1.519806 | 0.733659 | — | 0.700296 | 0.706208 | 0.706208 | 37.495509 | 129.885737 | 86.494417 | 18.770758 |
| O | 44 | 1.170209 | 0.743691 | — | 0.599014 | 0.604181 | 0.604181 | 44.503628 | 132.678187 | 81.312300 | 17.459393 |
| O | 45 | 1.048250 | 0.588008 | — | 0.565294 | 0.571412 | 0.571412 | 47.404830 | 115.487408 | 64.316204 | 15.891779 |
| O_tau025 | 43 | 1.709384 | 0.694298 | — | 0.662048 | 0.667742 | 0.667742 | 40.205969 | 167.780231 | 63.765543 | 16.048104 |
| O_tau025 | 44 | 1.165070 | 0.629252 | — | 0.598527 | 0.604500 | 0.604500 | 43.885010 | 221.618731 | 64.206242 | 16.183633 |
| O_tau025 | 45 | 0.876503 | 0.643560 | — | 0.484138 | 0.489693 | 0.489693 | 52.483861 | 113.116712 | 63.920530 | 15.887553 |
| S_tau025 | 43 | 1.498782 | 0.832966 | — | 0.614682 | 0.620495 | 0.620495 | 41.671929 | 168.489269 | 82.214165 | 18.690400 |
| S_tau025 | 44 | 1.580731 | 0.715245 | — | 0.679272 | 0.684919 | 0.684919 | 37.096467 | 102.918102 | 64.415337 | 16.104390 |
| S_tau025 | 45 | 1.256194 | 0.569899 | — | 0.526909 | 0.532379 | 0.532379 | 49.298637 | 152.869870 | 64.412850 | 15.928421 |
| T | 43 | 1.778716 | 0.764601 | 0.745363 | 0.713126 | 0.751724 | 0.719646 | 37.828303 | 108.811647 | 128.194044 | 10.593880 |
| T | 44 | 1.171906 | 0.651254 | 0.637731 | 0.612115 | 0.643614 | 0.617768 | 46.852198 | 107.229120 | 125.467291 | 10.779940 |
| T | 45 | 1.505803 | 0.779626 | 0.752168 | 0.719362 | 0.758278 | 0.726067 | 36.661303 | 108.354681 | 124.876563 | 11.219671 |

### Paired-seed comparisons

Directed full-SI rANS gaps versus the same-seed reference; seed-gap medians are separate from median-curve gaps.

| Method | Reference | Seed 43 | Seed 44 | Seed 45 | Valid pairs | Median paired gap % | Directed median-curve gap % |
|---|---|---:|---:|---:|---:|---:|---:|
| O_tau025 | O | +7.670517 | -4.659207 | -1.564142 | 3/3 | -1.564142 | +1.953427 |
| S_tau025 | O | +4.570148 | -9.963724 | -3.497871 | 3/3 | -3.497871 | -2.567006 |
| S_tau025 | O_tau025 | -3.676251 | -5.188801 | -3.474724 | 3/3 | -3.676251 | -4.024879 |
| T | O | +13.392703 | +3.326231 | -2.321965 | 3/3 | +3.326231 | +4.842531 |

Diagnostics (no measurement is silently removed):

- T / seed 43: monotone stages; no flagged rate growth or full-target prior failure.
- T / seed 44: monotone stages; no flagged rate growth or full-target prior failure.
- T / seed 45: monotone stages; no flagged rate growth or full-target prior failure.
- O / seed 43: monotone stages; no flagged rate growth or full-target prior failure.
- O / seed 44: monotone stages; no flagged rate growth or full-target prior failure.
- O / seed 45: monotone stages; no flagged rate growth or full-target prior failure.
- S_tau025 / seed 43: monotone stages; no flagged rate growth or full-target prior failure.
- S_tau025 / seed 44: monotone stages; no flagged rate growth or full-target prior failure.
- S_tau025 / seed 45: monotone stages; no flagged rate growth or full-target prior failure.
- O_tau025 / seed 43: monotone stages; no flagged rate growth or full-target prior failure.
- O_tau025 / seed 44: monotone stages; no flagged rate growth or full-target prior failure.
- O_tau025 / seed 45: monotone stages; no flagged rate growth or full-target prior failure.

All native rANS symbol errors are zero. Per-stage marginals were not measured; symbol-stage death is not inferred.


## Search inventory and setting changes

| Study | Sample counts | Extra planes | Additional settings / control | Repeats and setup |
|---|---|---|---|---|
| Initial Cartesian grid | 1k, 10k, 100k | 1, 2, 3 | Ordinary 3-plane pass; six-plane baseline; inherited 35 epochs / LR / tau | 9 S + O; fit seed 43; codec CUDA 1 / TITAN RTX, trace CUDA 0 |
| Expanded Cartesian grid | 1k, 10k, 100k, 300k, 1M, 3M | 0–6 inclusive | Nine-plane baseline; other defaults unchanged | 42 S + O; seed 43; RTX 3090 |
| Targeted acquisition/objective controls | 1k, 10k, 100k | 6 | All-head versus deployed-prefix fit; LWZQ versus compressed normalized FP16; 10 versus 35 epochs; smoothing multiplier 0 versus 0.5 | Six completed cases; earlier startup failed import, no result |
| Long FP16 budget | 1k, 10k, 40k, 100k | 6 in plan; FP16 has no LWZQ refinement stream | Prefix objective, **175 epochs** | Four cases; seed 43; capped row budget |
| Decoder adaptation | 1k, 10k, 100k FP16; 100k/e6 and 300k/e1 LWZQ | 1 or 6 for LWZQ | Encoder frozen; batch **125 versus 6,250** at 10k; 35 epochs | Six cases; seed 43 |
| Frozen joint prior | 10k and 100k FP16; 100k/e6 and 300k/e1 LWZQ | 1 or 6 for LWZQ | Joint prior frozen during quantizer fit; ordinary post-symbol entropy-prior refit still runs | Four cases; seed 43 |
| Output-head-only adaptation | Same four sample/format combinations as frozen prior | 1 or 6 for LWZQ | Encoder, decoder and conditional-prior recurrent backbones frozen; output heads train | Four cases; seed 43 |
| Hard decoder/prior training inputs | 300k/e1 all or prefix; 100k/e6 prefix; 10k FP16 prefix | 1 or 6 for LWZQ | Straight-through hard inputs; **10/35 epochs** | Five S completed; two nonmonotone; full-target hard O failed acceptance |
| Reconstruction weight | 300k/e1, 100k/e6 LWZQ; 10k FP16 | 1 or 6 for LWZQ | λ × **0.5 / 0.25** = 36.15800627594993 / 18.079003137974965; paired changed-λ O | Eight cases; seed 43 |
| Temporal masked control | Full target | None | T-shaped control keeps 150 columns / same baseline but only five own-client columns nonzero during quantizer fitting; full-SI prior uses real 150 columns | One T control; not standard five-column T |
| Fresh O / archived O reproduction | Full target | None | Explicit TF32 on versus old implicit off; identical data / baseline | Fresh on reproduces archived on at seed 43 |
| O TF32 repeat study | Full target | None | TF32 **on/off**, same RTX 3090; otherwise inherited warm settings | Five paired fit seeds **43–47**, including existing seed-43 controls |
| Historical raw-label range | 1k, 10k, 100k, 300k, 1M, 3M | **0** | Exact raw labels; fixed 550k draws; training-only and free exact decoder overlay evaluated separately | Six S fits + fresh O, seed 43; implicit TF32 off; earlier capped-budget jobs excluded |
| Precision-fixed raw-label range | Same six counts | **0** | Explicit `high` / TF32 on, AMP off; same baseline bytes; 35 epochs / inherited LR / tau | **21 fits**, **39 curves**, fresh O + each count at fit seeds **43–45**; RTX 3090 |
| Raw 3M epoch sweep | **3M** | **0** | **10, 20, 35, 70 epochs**, other factors fixed | Three paired seeds 43–45; reference 35 shared with other factors |
| Raw 3M LR sweep | **3M** | **0** | Warm LR × **0.1, 0.3, 1, 3** = 0.00008509229529398079, 0.00025527688588194236, 0.0008509229529398079, 0.0025527688588194235 | Three paired seeds; 35 epochs / tau 0.5 |
| Raw 3M tau sweep | **3M** | **0** | Initial tau **0.1, 0.25, 0.5, 1**; anneals to 0.1 (constant at 0.1) | Three paired seeds; 35 epochs / inherited LR; no combined-factor tuning |
| Coordinate/prior audit | **3M raw** | **0** | Fresh O, reference S and tau-0.25 S; capture per-element errors, post-refit prior bits and symbols | Nine reruns, three seeds; reproduce earlier scalar measurements |
| Prior swap controls | Same captured 3M raw cases | **0** | Own symbols with other model's prior; direct swap then XOR-aligned per-plane alphabet; no decoder change / further prior fit | 21 evaluations per successful direct/aligned study; two failed startups have no completed measurements |
| Worsening-coordinate histograms | Same audit arrays | **0** | x = raw gradient value; y = coordinate count; color = bin mean positive extra conditional-prior cost relative to O at equal distortion | No new training; final view retains only worsening coordinates, improvements omitted |

The three independent 3M optimizer factors share the O and reference-S controls: **33 fits /
63 primary-plus-free-overlay curves**, rather than treating their overlapping settings as
independent extra fits. All reliable precision-fixed fits use RTX 3090 / CUDA 0, PyTorch
2.5.1+cu121, CUDA 12.1, cuDNN 90100, explicit TF32 and AMP disabled. Earlier interventions
ran on both RTX 3090 and TITAN RTX; their wall times are not matched hardware benchmarks.

A separate protocol launch plan proposed all 30 clients at start 12, one trained plus nine
frozen rounds, 1k/10k/50k samples and one/three extras (six SR, one S, one OR case). It is a
planned coverage study, not evidence of completed multi-client results; the plan is removed.
The exact historical per-case configurations are recorded at the end of this report.

## Historical practical outcomes and precision caveat

The first grid's best case was **100k / one extra**, sample WMAPE **31.26%**, MAPE **2744.11%**,
full-vector WMAPE **41.87%**, rate **0.63099 bpf**, and symmetric RD penalty **+4.12%** against
its matched six-plane-baseline O. Three extras lowered input sample WMAPE to **14.19%** but
raised full-vector WMAPE to **42.24%** and the RD penalty to **+8.06%**. Better acquisition
error alone therefore did not ensure better final RD.

The expanded nine-plane-baseline grid's best case was **300k / one extra**, sample WMAPE
**28.19%**, MAPE **2426.49%**, acquisition **0.024083 bpf** including **0.003441 bpf** extra
planes, final WMAPE **35.23%**, rate **0.748942 bpf**, and RD penalty **+10.20%** against
expanded O (**37.50%**, **0.706208 bpf**). The original and expanded baselines differ; these
best scores do not isolate the effect of sample size. Expanded-grid RD penalties (%):

| Samples / extras | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1k | 18.79 | 17.57 | 19.13 | 21.53 | 21.94 | 23.95 | 19.43 |
| 10k | 19.49 | 15.89 | 14.35 | 19.43 | 19.44 | 19.49 | 18.70 |
| 100k | 15.14 | 20.78 | 24.64 | 24.62 | 23.53 | 25.49 | 26.75 |
| 300k | 14.26 | 10.20 | 21.98 | 26.40 | 27.49 | 28.18 | 28.26 |
| 1M | 24.56 | 12.41 | 16.20 | 25.01 | 28.47 | 39.88 | 41.25 |
| 3M | 20.39 | 12.81 | 17.12 | 35.09 | 51.31 | 61.25 | 79.65 |

Later practical ablations used cached inputs without repeating an FL constructor's hidden
TF32 enablement. Thus their cross-comparisons to imported original O/S curves mix numerical
policies and **cannot establish causal intervention gains**. Within-setting changes under a
matched policy remain informative: on 10k FP16 labels (~**0.018%** input WMAPE), 35 → 175 epochs
reduced sampled quantizer error **28.12% → 23.07%** but raised unsampled error
**34.26% → 34.51%**. Historical gaps against the imported O were **20.00% → 26.78%**;
at 100k those gaps were **23.38% → 31.63%**. Near-exact labels alone did not prevent poor
sample generalization. Head-only adaptation reduced sampled/unsampled error separation
without establishing a matched-policy practical RD advantage. The historical best apparent
+10.17% practical ablation result is not a validated improvement over +10.20%.

The hidden global behavior came from FL model construction enabling CUDA TF32. A cached
problem skipped construction, leaving a different training policy. Explicit TF32-on fresh O
exactly reproduced the archived seed-43 O; off versus on differed by **+7.03% RD**. Five
paired seeds 43–47 had off-versus-on gaps **+7.03, −0.50, −2.56, +8.42, +13.00%**. TF32-on
won three of five; pooled curves (mean cumulative rate and raw WMAPE first) favored on by
**5.07% RD**, with equal-rate WMAPE change **3.88%**. The policy mattered without winning
every seed. Explicit scoped precision now restores caller state, including on exceptions;
historical baseline producer precision remains unknown, with identical baseline bytes held
fixed in new comparisons. Earlier raw or ablation results are not silently relabeled as fixed.

## Coordinate, rate and histogram diagnostics

Nine precision-matched reruns captured O, reference raw S and tau-0.25 S for seeds 43–45.
Four layer-3 convolution tensors contain **2,064,384 coordinates (18.4623%)**, but account
for median **38.119%** of reference-S gross positive error regression and **40.115%** for
tau-0.25 S. Sampled and unsampled relative error increases were similar, so the observed
worsening there was not confined to coordinates omitted from training. This is a descriptive
concentration, not a causal fraction of the total net RD gap: improving coordinates elsewhere
can cancel regressions. A matched-distortion rate attribution alone does not show how much
an element caused inferior quantizer training.

Histogram x is raw parameter/gradient value, y is count, and bin color summarizes extra
conditional-prior rate versus O over the shared distortion span. The final worsening-only
view filters to increased reconstruction error, increased final conditional-prior bits, and
positive matched-distortion extra cost, and uses only the positive color scale. Alternative
per-bin RD curves were flagged unavailable where nonmonotone or nonoverlapping. **62.127%**
of target entries are zero; the 99th percentile of absolute target value is **0.00380619**.
A high count at zero can therefore have a small net rate/error contribution. Histograms are
removed with the diagnostic arrays; these observed setup facts are retained here.

Decoder reconstruction determines error; conditional-prior probabilities determine rate for
the produced symbols. O and S may use opposite binary labels for the same plane. A direct
prior swap is a valid cross-entropy evaluation but confounds prior quality with symbol-label
convention. The aligned control translates labels with a fixed-pool per-plane majority XOR;
it never changes decoder symbols or decoded values. Raw-S flip masks were [1,1,0], [0,1,0],
[0,0,0] for seeds 43/44/45; tau-0.25 masks were [0,0,0], [0,1,0], [0,0,0].
Even after alignment, O's prior on S symbols increased post-refit cross-entropy versus S's
own prior: reference S **+243.71%, +506.47%, +15.42%** and tau-0.25 S
**+31.59%, +417.42%, +196.38%** across those seeds. These are CE diagnostics, not measured
cross-model rANS or costs of transferring a selector/model. They neither establish prior
optimality nor prove quantizer training is the sole cause. No additional fitting occurred.

## Reliable precision-fixed raw-sample results

The following retained tables compare newly trained O with raw-label S. Raw samples have
**zero input WMAPE/MAPE and zero charged acquisition bits by assumption**. They test sample
training, not practical sample transmission. `X` additionally grants exact decoder samples
for free; this stronger oracle bound remains separate. All rates use the entropy prior fitted
again after quantizer fitting. The production S/SR acquisition path does not gain raw access.

## Primary comparison: raw training, full-vector quantization

Median curves take medians of **cumulative** rates and **raw** stage WMAPE before dB conversion
and bidirectional integrated equal-distortion comparison. They are aggregate curves, not one
trained machine. The paired-gap column instead takes the median of the three individually
matched S/O seed gaps. The knot column uses two-direction stage-knot interpolation.
Positive rate/equal-rate WMAPE changes favor O. No extrapolation or monotonic repair.

| Curve | Sample % | Final WM % | rANS bpf | Median-curve Δrate % | Knot Δrate % | Paired-gap median % | Valid pairs | Equal-rate ΔWM % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F* | 0.000 | 36.395 | 0.771315 | ~+13.305 | — | — | — | — |
| O | 0.000 | 44.504 | 0.604 | — | — | 0.000 | 3/3 | — |
| S1k | 0.009 | 81.208 | 0.668 | — | — | — | 0/3 | 48.236 |
| S10k | 0.089 | 75.367 | 0.853 | — | — | — | 0/3 | 48.485 |
| S100k | 0.894 | 32.807 | 0.970 | 30.364 | 28.862 | 34.702 | 3/3 | 21.815 |
| S300k | 2.683 | 31.147 | 0.897 | 8.857 | 7.764 | 8.803 | 3/3 | 7.041 |
| S1M | 8.943 | 35.409 | 0.716 | 4.747 | 2.996 | 3.949 | 3/3 | 3.785 |
| S3M | 26.830 | 42.690 | 0.626 | 2.854 | 2.204 | 0.733 | 3/3 | 2.280 |

**3M is best in this tested raw-training range:** +2.854% rate penalty on median curves,
+0.733% median paired penalty. Its paired range is **−7.835% to +7.607%**, winning one of
three seeds; it uses 26.83% of the target as privileged training data. This does not establish
a universal optimum or practical acquisition cost. The 1k/10k curves have no distortion
overlap with O; their equal-rate diagnostics show roughly 48% higher WMAPE.

| Curve | Sampled quantizer WM % | Unsampled quantizer WM % |
| --- | --- | --- |
| S1k | 0.926 | 81.216 |
| S10k | 3.253 | 75.434 |
| S100k | 17.759 | 32.949 |
| S300k | 24.357 | 31.334 |
| S1M | 34.316 | 35.570 |
| S3M | 42.401 | 42.797 |

These are post-training quantizer errors before replacement, using each group's own absolute
value denominator; they are different from the **0% error of the raw training labels**.
The 1k pool fits at 0.926% WMAPE but unsampled coordinates reconstruct at 81.216%, supporting
the sample-overfitting concern. At 1M/3M the sampled/unsampled errors become much closer.

The full-vector **joint/pre-finetuning prior exceeds the measured marginal** in every 1k,
10k and 100k primary/overlay result. Median primary final rates: **5.764 vs 1.628**, **5.926 vs
2.375**, and **2.931 vs 1.903** bpf. These are flagged prior failures on full/unseen data,
despite acceptance on training data. Native prior refitting reduces the mismatch (actual
rANS 0.668 / 0.853 / 0.970 bpf), but leaves the reconstruction problem. All measured curves
are monotone and symbol errors are zero. Stage marginal rates were not separately measured,
so dead symbol stages cannot be inferred from those missing values.

## Separate free exact decoder-sample bound

X uses the same S weights and additionally gives the decoder exact sampled values for free,
omitting their symbols. It has **no raw acquisition charge** and is a stronger oracle bound.

| Curve | Sample % | Final WM % | rANS bpf | Median-curve Δrate % | Knot Δrate % | Paired-gap median % | Valid pairs | Equal-rate ΔWM % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X1k | 0.009 | 81.208 | 0.668 | — | — | — | 0/3 | 48.233 |
| X10k | 0.089 | 75.364 | 0.853 | — | — | — | 0/3 | 48.441 |
| X100k | 0.894 | 32.656 | 0.962 | 28.885 | 27.333 | 33.238 | 3/3 | 20.703 |
| X300k | 2.683 | 30.490 | 0.873 | 3.394 | 2.068 | 3.521 | 3/3 | 2.662 |
| X1M | 8.943 | 32.400 | 0.653 | -15.125 | -16.881 | -18.231 | 3/3 | -13.407 |
| X3M | 26.830 | 31.304 | 0.460 | -74.012 | -74.074 | -77.168 | 2/3 | -55.872 |

X3M's paired median has **2/3** valid comparisons; the unavailable pair is `[{'seed': 45, 'reason': 'non_overlapping_distortion'}]`.
Its median-curve gap is valid. Large symmetric improvements use different directional
denominators and are not literal bandwidth-saving percentages. These free decoder values
cannot be interpreted as practical communication costs.


## Full square equal-distortion rate changes

S = raw training/full-vector quantization. X = additional free exact decoder samples.
Negative favors the row. Reverse entries are independently recomputed. No client exclusions,
so with/without exclusions give the same full square table below.

```text
Equal-distortion rate change (%), row vs column; negative favors row. Reverse entries recomputed.
            O   S1k  S10k   S100k  S300k    S1M     S3M   X1k  X10k   X100k  X300k     X1M      X3M      F*
   F* ~+14.19     —     — ~-15.18 ~+2.69 ~+9.06 ~+10.69     —     — ~-14.05 ~+8.11 ~+30.43 ~+105.89    0.00
    O   +0.00     —     —  -25.86  -8.47  -4.63   -2.81     —     —  -24.80  -3.34  +16.26   +98.42 ~-12.42
  S1k       — +0.00 +0.36       —      —      —       — +0.00 +0.50       —      —       —        —       —
 S10k       — -0.36 +0.00       —      —      —       — -0.36 +0.12       —      —       —        —       —
S100k  +34.87     —     —   +0.00 +19.74 +28.71  +30.88     —     —   +1.36 +25.93  +53.46  +137.24 ~+17.90
S300k   +9.25     —     —  -16.49  +0.00  +6.65   +6.34     —     —  -15.34  +5.15  +28.73  +101.44  ~-2.62
  S1M   +4.86     —     —  -22.31  -6.23  +0.00   +1.77     —     —  -21.27  -1.27  +19.91   +87.64  ~-8.31
  S3M   +2.89     —     —  -23.60  -5.96  -1.74   +0.00     —     —  -22.53  -0.77  +19.42   +97.03  ~-9.65
  X1k       — -0.00 +0.36       —      —      —       — +0.00 +0.50       —      —       —        —       —
 X10k       — -0.50 -0.12       —      —      —       — -0.50 +0.00       —      —       —        —       —
X100k  +32.97     —     —   -1.34 +18.11 +27.02  +29.08     —     —   +0.00 +24.22  +51.48  +134.15 ~+16.35
X300k   +3.45     —     —  -20.59  -4.89  +1.29   +0.77     —     —  -19.50  +0.00  +22.51   +92.13  ~-7.50
  X1M  -13.99     —     —  -34.83 -22.32 -16.60  -16.26     —     —  -33.99 -18.38   +0.00   +56.51 ~-23.33
  X3M  -49.60     —     —  -57.85 -50.36 -46.71  -49.25     —     —  -57.29 -47.95  -36.11    +0.00 ~-51.43
```

## Stage RD table

Rates are cumulative bits/full-vector float. Scalar-only marginal/fine-tuned-prior totals
appear at the final stage; unavailable measurements are `—`. Full-SI rANS is primary;
native transport is separate although it uses the same SI here. Temporal priors/transport
were not measured. WMAPE dB = `20*log10(median_raw_wmape/100)`.

| Curve | Stage | Marginal | Joint prior | Temporal prior | Full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WM dB | WM % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F* | 1 | — | 0.366520 | — | — | 0.372292 | — | 0.372292 | -3.094 | 70.034 |
| F* | 2 | — | 0.490296 | — | — | 0.496130 | — | 0.496130 | -4.631 | 58.672 |
| F* | 3 | 1.784724 | 0.765840 | — | 0.765840 | 0.771315 | — | 0.771315 | -8.779 | 36.395 |
| O | 1 | — | 0.364 | — | — | 0.363 | — | 0.363 | -3.601 | 66.059 |
| O | 2 | — | 0.453 | — | — | 0.444 | — | 0.444 | -5.071 | 55.778 |
| O | 3 | 1.170 | 0.734 | — | 0.599 | 0.604 | — | 0.604 | -7.032 | 44.504 |
| S1M | 1 | — | 0.369 | — | — | 0.368 | — | 0.368 | -3.668 | 65.556 |
| S1M | 2 | — | 0.546 | — | — | 0.515 | — | 0.515 | -5.446 | 53.418 |
| S1M | 3 | 1.621 | 0.810 | — | 0.711 | 0.716 | — | 0.716 | -9.018 | 35.409 |
| X1M | 1 | — | 0.337 | — | — | 0.336 | — | 0.336 | -4.439 | 59.985 |
| X1M | 2 | — | 0.499 | — | — | 0.470 | — | 0.470 | -6.227 | 48.824 |
| X1M | 3 | 1.476 | 0.740 | — | 0.648 | 0.653 | — | 0.653 | -9.789 | 32.400 |
| S100k | 1 | — | 0.925 | — | — | 0.381 | — | 0.381 | -2.540 | 74.648 |
| S100k | 2 | — | 1.985 | — | — | 0.702 | — | 0.702 | -5.980 | 50.236 |
| S100k | 3 | 1.903 | 2.931 | — | 0.964 | 0.970 | — | 0.970 | -9.681 | 32.807 |
| X100k | 1 | — | 0.922 | — | — | 0.378 | — | 0.378 | -2.582 | 74.283 |
| X100k | 2 | — | 1.980 | — | — | 0.696 | — | 0.696 | -6.018 | 50.016 |
| X100k | 3 | 1.886 | 2.925 | — | 0.955 | 0.962 | — | 0.962 | -9.721 | 32.656 |
| S10k | 1 | — | 2.671 | — | — | 0.384 | — | 0.384 | -1.501 | 84.129 |
| S10k | 2 | — | 4.528 | — | — | 0.666 | — | 0.666 | -1.705 | 82.181 |
| S10k | 3 | 2.375 | 5.926 | — | 0.849 | 0.853 | — | 0.853 | -2.456 | 75.367 |
| X10k | 1 | — | 2.671 | — | — | 0.384 | — | 0.384 | -1.501 | 84.125 |
| X10k | 2 | — | 4.527 | — | — | 0.665 | — | 0.665 | -1.705 | 82.178 |
| X10k | 3 | 2.373 | 5.926 | — | 0.848 | 0.853 | — | 0.853 | -2.457 | 75.364 |
| S1k | 1 | — | 2.291 | — | — | 0.375 | — | 0.375 | -1.409 | 85.026 |
| S1k | 2 | — | 4.907 | — | — | 0.500 | — | 0.500 | -1.518 | 83.969 |
| S1k | 3 | 1.628 | 5.764 | — | 0.662 | 0.668 | — | 0.668 | -1.808 | 81.208 |
| X1k | 1 | — | 2.291 | — | — | 0.375 | — | 0.375 | -1.409 | 85.026 |
| X1k | 2 | — | 4.907 | — | — | 0.500 | — | 0.500 | -1.518 | 83.969 |
| X1k | 3 | 1.628 | 5.764 | — | 0.662 | 0.668 | — | 0.668 | -1.808 | 81.208 |
| S3M | 1 | — | 0.379 | — | — | 0.378 | — | 0.378 | -3.767 | 64.812 |
| S3M | 2 | — | 0.544 | — | — | 0.483 | — | 0.483 | -5.250 | 54.638 |
| S3M | 3 | 1.258 | 0.698 | — | 0.621 | 0.626 | — | 0.626 | -7.393 | 42.690 |
| X3M | 1 | — | 0.278 | — | — | 0.278 | — | 0.278 | -6.455 | 47.559 |
| X3M | 2 | — | 0.399 | — | — | 0.354 | — | 0.354 | -7.941 | 40.084 |
| X3M | 3 | 0.920 | 0.511 | — | 0.454 | 0.460 | — | 0.460 | -10.088 | 31.304 |
| S300k | 1 | — | 0.421 | — | — | 0.370 | — | 0.370 | -3.304 | 68.357 |
| S300k | 2 | — | 0.737 | — | — | 0.608 | — | 0.608 | -6.596 | 46.793 |
| S300k | 3 | 1.975 | 1.046 | — | 0.890 | 0.897 | — | 0.897 | -10.132 | 31.147 |
| X300k | 1 | — | 0.411 | — | — | 0.361 | — | 0.361 | -3.501 | 66.823 |
| X300k | 2 | — | 0.720 | — | — | 0.592 | — | 0.592 | -6.789 | 45.765 |
| X300k | 3 | 1.922 | 1.021 | — | 0.867 | 0.873 | — | 0.873 | -10.317 | 30.490 |

## Reliable independent 3M optimizer sweeps

Default reference settings are 35 epochs, warm LR 0.0008509229529398079, initial tau 0.5.
The controls and fixed-pool draw budget match the precision-fixed range study. Each factor
changes alone; the following three-factor report includes shared controls, not a combined
hyperparameter search. Tau 0.25 is the most consistent tested raw-label option; it has not
been promoted to the practical S/SR default. Three million samples is the chosen default
count for now, not a universal optimum or a claim that practical transmission matches O.

## Independent factors and results

Unchanged values for each factor: 35 epochs, warm LR **0.0008509229529398079**, initial tau **0.5**. Sweeps: epochs 10/20/35/70; LR factors 0.1/0.3/1/3; initial tau 0.1/0.25/0.5/1. Tau anneals toward 0.1 over the chosen epoch count; tau=0.1 stays constant. Lambda is 72.31601255189986. Each factor varies alone; combined LR/tau changes were not tested.

Median curves take componentwise medians of cumulative stage rates and raw WMAPE before dB conversion and piecewise-linear RD comparison. They are aggregate curves, not a single machine. Paired gap medians compare each fit with its fresh same-seed O control separately. Rate gaps are the sign-correct mean of both directed integrations at equal distortion; the equal-rate column is the analogous raw-WMAPE percentage change. Negative favors S. Bidirectional stage-knot diagnostics were recorded separately in the removed CSV outputs. Final rate/error pairs need not lie at equal distortion.

| Setting | Final WMAPE % | Refitted prior bpf | Full-SI rANS bpf | Median-curve rate gap vs O % | Paired median gap % (valid) | Equal-rate WMAPE change % |
|---|---:|---:|---:|---:|---:|---:|
| F* (prior frozen) | 36.40 | — | 0.771315 | ~+13.33 | — | — |
| Fresh O | 44.50 | 0.599014 | 0.604181 | 0.00 | 0.00 (3/3) | 0.00 |
| S reference | 42.69 | 0.620871 | 0.626339 | +2.85 | +0.73 (3/3) | +2.28 |
| Epochs 10 | 38.12 | 0.685859 | 0.691533 | +3.00 | +0.00 (3/3) | +2.36 |
| Epochs 20 | 36.52 | 0.720144 | 0.726089 | +3.42 | +1.38 (3/3) | +2.69 |
| Epochs 70 | 36.89 | 0.693304 | 0.699526 | +3.33 | -1.00 (3/3) | +2.66 |
| LR 0.1× | 38.20 | 0.684734 | 0.690769 | +2.54 | +1.56 (3/3) | +1.99 |
| LR 0.3× | 35.78 | 0.707897 | 0.714126 | +0.26 | +1.54 (3/3) | +0.08 |
| LR 3×† | 60.15 | 0.395499 | 0.400627 | -0.67 | +13.54 (1/3) | -0.50 |
| Tau 0.1 | 41.17 | 0.625553 | 0.631036 | -2.83 | -2.76 (3/3) | -2.04 |
| Tau 0.25 | 41.67 | 0.614682 | 0.620495 | -2.60 | -3.56 (3/3) | -1.73 |
| Tau 1 | 37.46 | 0.684048 | 0.689917 | -1.02 | +0.48 (3/3) | -0.81 |

† LR 3× is unstable: nine rejected quantizer attempts and non-monotone primary curves for seeds 44 and 45. Its apparently favorable median curve hides two invalid individual curves. The sole valid paired gap is +13.54%; it is not a recommended setting. Individual stage points and flags were checked before their raw outputs were removed.

For epoch count, 35 gives the smallest aggregate RD penalty among tested values. Ten epochs is a training-cost option with nearly the same aggregate curve, not an RD improvement. For LR, 0.3× (**0.00025527688588194236**) reduces the median-curve gap to +0.26%, although its paired median does not improve. For tau, 0.1 gives the best aggregate result and 0.25 gives the better paired median and consistent improvement over reference S. Prefer **3M / 35 epochs / warm LR / tau=0.25** for the next validation; use tau=0.1 when selecting strictly by the median-curve score. No combined-setting gain is claimed.

## Paired comparison with reference S and fit cost

Negative rate changes favor the tuned setting. Wins count valid paired full-SI RD comparisons. Quantizer attempts include rejected attempts. Training seconds are observed medians under shared-GPU execution, not isolated timing benchmarks.

| Setting | Median-curve gap vs S % | Paired median gap vs S % | Wins vs S | Quantizer attempts | Median quantizer-fit seconds |
|---|---:|---:|---:|---:|---:|
| S reference | +0.00 | +0.00 | 0/3 | 6 | 115.2 |
| Epochs 10 | +0.19 | +0.82 | 1/3 | 7 | 43.9 |
| Epochs 20 | +0.61 | +0.62 | 1/3 | 6 | 74.0 |
| Epochs 70 | +0.33 | +0.20 | 1/3 | 6 | 222.3 |
| LR 0.1× | -0.15 | +0.50 | 0/3 | 6 | 111.4 |
| LR 0.3× | -2.55 | +0.53 | 1/3 | 6 | 112.9 |
| LR 3×† | -3.32 | +1.09 | 0/1 | 15 | 286.7 |
| Tau 0.1 | -5.20 | -3.09 | 2/3 | 10 | 169.3 |
| Tau 0.25 | -4.93 | -4.55 | 3/3 | 8 | 168.4 |
| Tau 1 | -3.68 | +0.65 | 1/3 | 7 | 117.0 |

## Full directed square comparison

Equal-distortion rate changes (%) on median primary full-SI rANS curves, row versus column; negative favors row. Reverse entries are recomputed. `S35` is reference S; `E` labels epochs, `L` labels LR multipliers, and `T` labels initial tau. `L3†` retains the aggregate curve but has only one valid individual comparison; see the warning above. Auxiliary and individual-repeat matrices were recorded separately and are now removed.

```text
            O     S35    E10    E20     E70    L0.1    L0.3     L3†    T0.1   T0.25      T1      F*
   F* ~+14.21 ~+10.72 ~+9.95 ~+8.85 ~+10.05 ~+10.27 ~+12.44 ~+16.66 ~+16.99 ~+17.24 ~+13.45    0.00
    O   +0.00   -2.81  -2.96  -3.36   -3.28   -2.50   -0.26   +0.68   +2.87   +2.63   +1.02 ~-12.44
  S35   +2.89   +0.00  -0.19  -0.61   -0.33   +0.15   +2.59   +3.37   +5.33   +5.05   +3.75  ~-9.68
  E10   +3.05   +0.19  +0.00  -0.63   +0.20   +0.31   +2.60   +3.62   +5.41   +5.40   +3.42  ~-9.05
  E20   +3.48   +0.61  +0.64  +0.00   +1.00   +0.85   +3.26   +2.56   +5.73   +5.53   +4.00  ~-8.13
  E70   +3.39   +0.33  -0.20  -0.99   +0.00   +0.09   +2.25   +4.50   +5.34   +5.27   +3.08  ~-9.13
 L0.1   +2.57   -0.15  -0.31  -0.84   -0.09   +0.00   +2.36   +1.26   +4.94   +4.85   +3.18  ~-9.31
 L0.3   +0.26   -2.52  -2.54  -3.16   -2.20   -2.31   +0.00   +0.05   +2.46   +2.30   +0.75 ~-11.06
  L3†   -0.67   -3.26  -3.50  -2.50   -4.30   -1.25   -0.05   +0.00   +4.70   +5.64   +3.95 ~-14.28
 T0.1   -2.79   -5.06  -5.13  -5.42   -5.07   -4.70   -2.40   -4.49   +0.00   +0.18   -1.41 ~-14.52
T0.25   -2.57   -4.80  -5.12  -5.24   -5.01   -4.63   -2.25   -5.34   -0.18   +0.00   -1.21 ~-14.71
   T1   -1.01   -3.61  -3.31  -3.84   -2.99   -3.09   -0.74   -3.80   +1.43   +1.23   +0.00 ~-11.86
```

## Stage RD points

Cumulative bits/float. Scalar totals appear only at stage 3; unavailable measurements are `—`. Joint prior is measured before the post-quantizer entropy-prior refit. Native and full-SI rANS are the same measured transport here. Raw WMAPE medians are converted to dB after aggregation.

| Setting | Stage | Marginal | Joint prior | Temporal prior | Refitted full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F* | 1 | — | 0.366520 | — | — | 0.372292 | — | 0.372292 | -3.09 (70.034%) |
| F* | 2 | — | 0.490296 | — | — | 0.496130 | — | 0.496130 | -4.63 (58.672%) |
| F* | 3 | 1.784724 | 0.765840 | — | — | 0.771315 | — | 0.771315 | -8.78 (36.395%) |
| O | 1 | — | 0.3635 | — | — | 0.3632 | — | 0.3632 | -3.60 (66.06%) |
| O | 2 | — | 0.4529 | — | — | 0.4436 | — | 0.4436 | -5.07 (55.78%) |
| O | 3 | 1.1702 | 0.7337 | — | 0.5990 | 0.6042 | — | 0.6042 | -7.03 (44.50%) |
| S reference | 1 | — | 0.3795 | — | — | 0.3778 | — | 0.3778 | -3.77 (64.81%) |
| S reference | 2 | — | 0.5438 | — | — | 0.4826 | — | 0.4826 | -5.25 (54.64%) |
| S reference | 3 | 1.2582 | 0.6976 | — | 0.6209 | 0.6263 | — | 0.6263 | -7.39 (42.69%) |
| Epochs 10 | 1 | — | 0.3637 | — | — | 0.3633 | — | 0.3633 | -3.56 (66.38%) |
| Epochs 10 | 2 | — | 0.4588 | — | — | 0.4526 | — | 0.4526 | -4.78 (57.65%) |
| Epochs 10 | 3 | 1.7164 | 0.7188 | — | 0.6859 | 0.6915 | — | 0.6915 | -8.38 (38.12%) |
| Epochs 20 | 1 | — | 0.3772 | — | — | 0.3759 | — | 0.3759 | -3.82 (64.39%) |
| Epochs 20 | 2 | — | 0.4858 | — | — | 0.4716 | — | 0.4716 | -5.04 (56.01%) |
| Epochs 20 | 3 | 1.7662 | 0.7508 | — | 0.7201 | 0.7261 | — | 0.7261 | -8.75 (36.52%) |
| Epochs 70 | 1 | — | 0.3775 | — | — | 0.3757 | — | 0.3757 | -3.70 (65.30%) |
| Epochs 70 | 2 | — | 0.4613 | — | — | 0.4526 | — | 0.4526 | -4.71 (58.15%) |
| Epochs 70 | 3 | 1.7878 | 0.7182 | — | 0.6933 | 0.6995 | — | 0.6995 | -8.66 (36.89%) |
| LR 0.1× | 1 | — | 0.3557 | — | — | 0.3540 | — | 0.3540 | -3.65 (65.72%) |
| LR 0.1× | 2 | — | 0.4547 | — | — | 0.4428 | — | 0.4428 | -4.64 (58.58%) |
| LR 0.1× | 3 | 1.6694 | 0.7091 | — | 0.6847 | 0.6908 | — | 0.6908 | -8.36 (38.20%) |
| LR 0.3× | 1 | — | 0.3627 | — | — | 0.3645 | — | 0.3645 | -3.77 (64.79%) |
| LR 0.3× | 2 | — | 0.4723 | — | — | 0.4607 | — | 0.4607 | -5.10 (55.57%) |
| LR 0.3× | 3 | 1.7979 | 0.7292 | — | 0.7079 | 0.7141 | — | 0.7141 | -8.93 (35.78%) |
| LR 3×† | 1 | — | 0.4593 | — | — | 0.3636 | — | 0.3636 | -3.69 (65.39%) |
| LR 3×† | 2 | — | 0.6076 | — | — | 0.4006 | — | 0.4006 | -4.29 (61.03%) |
| LR 3×† | 3 | 0.6731 | 0.6076 | — | 0.3955 | 0.4006 | — | 0.4006 | -4.42 (60.15%) |
| Tau 0.1 | 1 | — | 0.2962 | — | — | 0.2877 | — | 0.2877 | -3.04 (70.51%) |
| Tau 0.1 | 2 | — | 0.4013 | — | — | 0.3637 | — | 0.3637 | -3.99 (63.20%) |
| Tau 0.1 | 3 | 1.5114 | 0.6840 | — | 0.6256 | 0.6310 | — | 0.6310 | -7.71 (41.17%) |
| Tau 0.25 | 1 | — | 0.2836 | — | — | 0.2823 | — | 0.2823 | -3.20 (69.15%) |
| Tau 0.25 | 2 | — | 0.4259 | — | — | 0.4041 | — | 0.4041 | -4.46 (59.85%) |
| Tau 0.25 | 3 | 1.4988 | 0.7152 | — | 0.6147 | 0.6205 | — | 0.6205 | -7.60 (41.67%) |
| Tau 1 | 1 | — | 0.3465 | — | — | 0.3470 | — | 0.3470 | -3.78 (64.74%) |
| Tau 1 | 2 | — | 0.4819 | — | — | 0.4636 | — | 0.4636 | -5.20 (54.94%) |
| Tau 1 | 3 | 1.6378 | 0.7483 | — | 0.6840 | 0.6899 | — | 0.6899 | -8.53 (37.46%) |

## Historical plan identities and exact variant settings

These plan identifiers distinguish runs whose target/baseline setup was held fixed. Counts
in this table identify declared cases, not automatically completed fits: failed/import-only
jobs and superseded capped raw-budget plans are described above. Values absent from a plan
inherit the common setup or imported reference. FP16 `e` values in identifiers do not create
LWZQ tail streams. Historical raw labels were renamed in fixed-budget execution; the reliable
completed-count summaries above come from their completed reports, not a filename guess.

| Plan identifier | Explicit setup / variants |
|---|---|
| `svhn_budget.json` | `{"name": "sampled_budget_svhn_seed43_r5_20261007", "fit_device": "cuda:1", "cases": [{"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "fp16_epochs175", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 175}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "fp16_epochs175", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 175}}, {"method": "S", "sample_size": 1000, "extra_planes": 6, "variant": "fp16_epochs175", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 175}}, {"method": "S", "sample_size": 40000, "extra_planes": 6, "variant": "fp16_epochs175", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 175}}]}` |
| `svhn_decoder.json` | `{"name": "sampled_decoder_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "cases": [{"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "decoder_fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 35, "sample_fit_batch_size": null, "sample_train_encoder": false}}, {"method": "S", "sample_size": 1000, "extra_planes": 6, "variant": "decoder_fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 35, "sample_fit_batch_size": null, "sample_train_encoder": false}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "decoder_fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 35, "sample_fit_batch_size": null, "sample_train_encoder": false}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "decoder_fp16_batch125", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 35, "sample_fit_batch_size": 125, "sample_train_encoder": false}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "decoder_lwzq", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_fit_batch_size": null, "sample_train_encoder": false}}, {"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "decoder_lwzq", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_fit_batch_size": null, "sample_train_encoder": false}}]}` |
| `svhn_element_audit_part1.json` | `{"name": "sampled_element_audit_part1_svhn_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": true, "reference_cases": [], "cases": [{"method": "O", "variant": "raw_seed43", "fit_seed": 43, "capture_elements": true}, {"method": "S", "variant": "tau0p25_seed43", "fit_seed": 43, "capture_elements": true, "sample_size": 3000000, "extra_planes": 0, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.25}}, {"method": "S", "variant": "tau0p25_seed44", "fit_seed": 44, "capture_elements": true, "sample_size": 3000000, "extra_planes": 0, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.25}}, {"method": "S", "variant": "raw_seed45", "fit_seed": 45, "capture_elements": true, "sample_size": 3000000, "extra_planes": 0, "raw_sample_oracle": true}]}` |
| `svhn_element_audit_part2.json` | `{"name": "sampled_element_audit_part2_svhn_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": true, "reference_cases": [], "cases": [{"method": "S", "variant": "raw_seed43", "fit_seed": 43, "capture_elements": true, "sample_size": 3000000, "extra_planes": 0, "raw_sample_oracle": true}, {"method": "O", "variant": "raw_seed44", "fit_seed": 44, "capture_elements": true}, {"method": "S", "variant": "raw_seed44", "fit_seed": 44, "capture_elements": true, "sample_size": 3000000, "extra_planes": 0, "raw_sample_oracle": true}, {"method": "O", "variant": "raw_seed45", "fit_seed": 45, "capture_elements": true}, {"method": "S", "variant": "tau0p25_seed45", "fit_seed": 45, "capture_elements": true, "sample_size": 3000000, "extra_planes": 0, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.25}}]}` |
| `svhn_fixedprior.json` | `{"name": "sampled_fixedprior_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "cases": [{"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "fixedprior_fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 35, "sample_train_prior": false}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "fixedprior_fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 35, "sample_train_prior": false}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "fixedprior_lwzq", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_train_prior": false}}, {"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "fixedprior_lwzq", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_train_prior": false}}]}` |
| `svhn_grid.json` | `{"name": "sampled_grid_svhn_seed43_r5_20261005", "trace_device": "cuda:0", "fit_device": "cuda:1", "seed": 43, "target_round": 5, "target_client": 0, "baseline_client": 1, "sample_sizes": [1000, 10000, 100000], "extra_planes": [1, 2, 3], "quantizer": {"bins_per_plane": 2, "num_planes": 3, "training_log_every_epochs": 10}}` |
| `svhn_grid_expanded.json` | `{"name": "sampled_grid_svhn_expanded_seed43_r5_20261006", "trace_device": "cuda:0", "fit_device": "cuda:0", "seed": 43, "target_round": 5, "target_client": 0, "baseline_client": 1, "sample_sizes": [1000, 10000, 100000, 300000, 1000000, 3000000], "extra_planes": [0, 1, 2, 3, 4, 5, 6], "quantizer": {"bins_per_plane": 2, "num_planes": 3, "training_log_every_epochs": 10}}` |
| `svhn_hard.json` | `{"name": "sampled_hard_svhn_seed43_r5_20261007", "fit_device": "cuda:1", "cases": [{"method": "O", "variant": "hard", "hard_inputs": true}, {"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "hard_lwzq_all_epochs35", "sampling_options": {"sample_training": "all", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_hard_inputs": true}}, {"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "hard_lwzq_prefix_epochs35", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_hard_inputs": true}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "hard_lwzq_prefix_epochs35", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_hard_inputs": true}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "hard_fp16_prefix_epochs35", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 35, "sample_hard_inputs": true}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "hard_fp16_prefix_epochs10", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 10, "sample_hard_inputs": true}}]}` |
| `svhn_hard_s.json` | `{"name": "sampled_hard_s_svhn_seed43_r5_20261007", "fit_device": "cuda:1", "cases": [{"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "hard_lwzq_all_epochs35", "sampling_options": {"sample_training": "all", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_hard_inputs": true}}, {"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "hard_lwzq_prefix_epochs35", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_hard_inputs": true}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "hard_lwzq_prefix_epochs35", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": 35, "sample_hard_inputs": true}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "hard_fp16_prefix_epochs35", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 35, "sample_hard_inputs": true}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "hard_fp16_prefix_epochs10", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 10, "sample_hard_inputs": true}}]}` |
| `svhn_heads.json` | `{"name": "sampled_heads_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "cases": [{"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "heads_lwzq", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_train_backbones": false}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "heads_lwzq", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_train_backbones": false}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "heads_fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_train_backbones": false}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "heads_fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_train_backbones": false}}]}` |
| `svhn_lambda.json` | `{"name": "sampled_lambda_svhn_seed43_r5_20261007", "fit_device": "cuda:1", "cases": [{"method": "O", "variant": "lambda0p5", "reconst_ld": 36.15800627594993}, {"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "lwzq_lambda0p5", "reconst_ld": 36.15800627594993, "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq"}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "lwzq_lambda0p5", "reconst_ld": 36.15800627594993, "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq"}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "fp16_lambda0p5", "reconst_ld": 36.15800627594993, "sampling_options": {"sample_training": "prefix", "sample_format": "fp16"}}, {"method": "O", "variant": "lambda0p25", "reconst_ld": 18.079003137974965}, {"method": "S", "sample_size": 300000, "extra_planes": 1, "variant": "lwzq_lambda0p25", "reconst_ld": 18.079003137974965, "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq"}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "lwzq_lambda0p25", "reconst_ld": 18.079003137974965, "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq"}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "fp16_lambda0p25", "reconst_ld": 18.079003137974965, "sampling_options": {"sample_training": "prefix", "sample_format": "fp16"}}]}` |
| `svhn_o_tf32_reproduction.json` | `{"name": "sampled_o_tf32_reproduction_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": true, "reference_cases": ["O"], "cases": [{"method": "O", "variant": "tf32_reproduction"}]}` |
| `svhn_o_tf32_seeds_off.json` | `{"name": "sampled_o_tf32_seeds_off_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": false, "reference_cases": ["O"], "cases": [{"method": "O", "variant": "tf32_off_seed44", "fit_seed": 44}, {"method": "O", "variant": "tf32_off_seed45", "fit_seed": 45}, {"method": "O", "variant": "tf32_off_seed46", "fit_seed": 46}, {"method": "O", "variant": "tf32_off_seed47", "fit_seed": 47}]}` |
| `svhn_o_tf32_seeds_on.json` | `{"name": "sampled_o_tf32_seeds_on_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": true, "reference_cases": ["O"], "cases": [{"method": "O", "variant": "tf32_on_seed44", "fit_seed": 44}, {"method": "O", "variant": "tf32_on_seed45", "fit_seed": 45}, {"method": "O", "variant": "tf32_on_seed46", "fit_seed": 46}, {"method": "O", "variant": "tf32_on_seed47", "fit_seed": 47}]}` |
| `svhn_prior_swap_aligned_median3.json` | `{"name": "sampled_prior_swap_aligned_median3_svhn_r5_20261007", "input_roots": ["../../_/experiments/sampled_element_audit_part1_svhn_r5_20261007", "../../_/experiments/sampled_element_audit_part2_svhn_r5_20261007"], "fit_device": "cuda:0", "align_symbols": true}` |
| `svhn_prior_swap_median3.json` | `{"name": "sampled_prior_swap_median3_svhn_r5_20261007", "input_roots": ["../../_/experiments/sampled_element_audit_part1_svhn_r5_20261007", "../../_/experiments/sampled_element_audit_part2_svhn_r5_20261007"], "fit_device": "cuda:0"}` |
| `svhn_prior_swap_median3_v2.json` | `{"name": "sampled_prior_swap_median3_svhn_r5_20261007_v2", "input_roots": ["../../_/experiments/sampled_element_audit_part1_svhn_r5_20261007", "../../_/experiments/sampled_element_audit_part2_svhn_r5_20261007"], "fit_device": "cuda:0"}` |
| `svhn_prior_swap_median3_v3.json` | `{"name": "sampled_prior_swap_median3_svhn_r5_20261007_v3", "input_roots": ["../../_/experiments/sampled_element_audit_part1_svhn_r5_20261007", "../../_/experiments/sampled_element_audit_part2_svhn_r5_20261007"], "fit_device": "cuda:0"}` |
| `svhn_raw_optimizer_median3_part1.json` | `{"name": "sampled_raw_optimizer_median3_part1_svhn_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": true, "reference_cases": [], "cases": [{"method": "O", "variant": "raw_seed43", "fit_seed": 43}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs10_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 10}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs70_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 70}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr3_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 0.002552768858819424}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau0p25_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.25}}, {"method": "O", "variant": "raw_seed44", "fit_seed": 44}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs10_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 10}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs20_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 20}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr0p1_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 8.50922952939808e-05}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr0p3_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 0.0002552768858819424}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau0p1_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.1}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau1_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 1.0}}, {"method": "O", "variant": "raw_seed45", "fit_seed": 45}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs70_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 70}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr3_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 0.002552768858819424}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau0p25_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.25}}]}` |
| `svhn_raw_optimizer_median3_part2.json` | `{"name": "sampled_raw_optimizer_median3_part2_svhn_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": true, "reference_cases": [], "cases": [{"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "raw_seed43", "fit_seed": 43, "raw_sample_oracle": true}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs20_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 20}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr0p1_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 8.50922952939808e-05}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr0p3_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 0.0002552768858819424}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau0p1_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.1}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau1_seed43", "fit_seed": 43, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 1.0}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "raw_seed44", "fit_seed": 44, "raw_sample_oracle": true}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs70_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 70}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr3_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 0.002552768858819424}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau0p25_seed44", "fit_seed": 44, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.25}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "raw_seed45", "fit_seed": 45, "raw_sample_oracle": true}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs10_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 10}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "epochs20_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_epochs": 20}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr0p1_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 8.50922952939808e-05}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "lr0p3_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_lr": 0.0002552768858819424}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau0p1_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 0.1}}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "tau1_seed45", "fit_seed": 45, "raw_sample_oracle": true, "sampling_options": {"sample_fit_tau": 1.0}}]}` |
| `svhn_raw_oracle_large.json` | `{"name": "sampled_raw_oracle_large_svhn_seed43_r5_20261007", "fit_device": "cuda:1", "cases": [{"method": "S", "sample_size": 1000000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}]}` |
| `svhn_raw_oracle_large_fixed_budget.json` | `{"name": "sampled_raw_oracle_large_svhn_seed43_r5_20261007_fixed_budget", "fit_device": "cuda:1", "cases": [{"method": "S", "sample_size": 1000000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}]}` |
| `svhn_raw_oracle_small.json` | `{"name": "sampled_raw_oracle_small_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "cases": [{"method": "O", "variant": "matched_raw"}, {"method": "S", "sample_size": 1000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}, {"method": "S", "sample_size": 10000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}, {"method": "S", "sample_size": 100000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}, {"method": "S", "sample_size": 300000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}]}` |
| `svhn_raw_oracle_small_fixed_budget.json` | `{"name": "sampled_raw_oracle_small_svhn_seed43_r5_20261007_fixed_budget", "fit_device": "cuda:0", "cases": [{"method": "O", "variant": "matched_raw"}, {"method": "S", "sample_size": 1000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}, {"method": "S", "sample_size": 10000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}, {"method": "S", "sample_size": 100000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}, {"method": "S", "sample_size": 300000, "extra_planes": 0, "variant": "raw", "raw_sample_oracle": true}]}` |
| `svhn_raw_precision_large_median3.json` | `{"name": "sampled_raw_precision_large_median3_svhn_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": true, "reference_cases": [], "cases": [{"method": "S", "sample_size": 300000, "extra_planes": 0, "variant": "raw_seed43", "fit_seed": 43, "raw_sample_oracle": true}, {"method": "S", "sample_size": 1000000, "extra_planes": 0, "variant": "raw_seed43", "fit_seed": 43, "raw_sample_oracle": true}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "raw_seed43", "fit_seed": 43, "raw_sample_oracle": true}, {"method": "S", "sample_size": 300000, "extra_planes": 0, "variant": "raw_seed44", "fit_seed": 44, "raw_sample_oracle": true}, {"method": "S", "sample_size": 1000000, "extra_planes": 0, "variant": "raw_seed44", "fit_seed": 44, "raw_sample_oracle": true}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "raw_seed44", "fit_seed": 44, "raw_sample_oracle": true}, {"method": "S", "sample_size": 300000, "extra_planes": 0, "variant": "raw_seed45", "fit_seed": 45, "raw_sample_oracle": true}, {"method": "S", "sample_size": 1000000, "extra_planes": 0, "variant": "raw_seed45", "fit_seed": 45, "raw_sample_oracle": true}, {"method": "S", "sample_size": 3000000, "extra_planes": 0, "variant": "raw_seed45", "fit_seed": 45, "raw_sample_oracle": true}]}` |
| `svhn_raw_precision_small_median3.json` | `{"name": "sampled_raw_precision_small_median3_svhn_r5_20261007", "fit_device": "cuda:0", "matmul_allow_tf32": true, "reference_cases": [], "cases": [{"method": "O", "variant": "raw_seed43", "fit_seed": 43}, {"method": "S", "sample_size": 1000, "extra_planes": 0, "variant": "raw_seed43", "fit_seed": 43, "raw_sample_oracle": true}, {"method": "S", "sample_size": 10000, "extra_planes": 0, "variant": "raw_seed43", "fit_seed": 43, "raw_sample_oracle": true}, {"method": "S", "sample_size": 100000, "extra_planes": 0, "variant": "raw_seed43", "fit_seed": 43, "raw_sample_oracle": true}, {"method": "O", "variant": "raw_seed44", "fit_seed": 44}, {"method": "S", "sample_size": 1000, "extra_planes": 0, "variant": "raw_seed44", "fit_seed": 44, "raw_sample_oracle": true}, {"method": "S", "sample_size": 10000, "extra_planes": 0, "variant": "raw_seed44", "fit_seed": 44, "raw_sample_oracle": true}, {"method": "S", "sample_size": 100000, "extra_planes": 0, "variant": "raw_seed44", "fit_seed": 44, "raw_sample_oracle": true}, {"method": "O", "variant": "raw_seed45", "fit_seed": 45}, {"method": "S", "sample_size": 1000, "extra_planes": 0, "variant": "raw_seed45", "fit_seed": 45, "raw_sample_oracle": true}, {"method": "S", "sample_size": 10000, "extra_planes": 0, "variant": "raw_seed45", "fit_seed": 45, "raw_sample_oracle": true}, {"method": "S", "sample_size": 100000, "extra_planes": 0, "variant": "raw_seed45", "fit_seed": 45, "raw_sample_oracle": true}]}` |
| `svhn_targeted.json` | `{"name": "sampled_targeted_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "cases": [{"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "prefix", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": null, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": null, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": null, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "fp16_short", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 10, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 1000, "extra_planes": 6, "variant": "fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": null, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "prefix_smooth", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": null, "sample_noise_multiplier": 0.5}}]}` |
| `svhn_targeted_v2.json` | `{"name": "sampled_targeted_svhn_seed43_r5_20261007_v2", "fit_device": "cuda:0", "cases": [{"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "prefix", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": null, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": null, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": null, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 10000, "extra_planes": 6, "variant": "fp16_short", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": 10, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 1000, "extra_planes": 6, "variant": "fp16", "sampling_options": {"sample_training": "prefix", "sample_format": "fp16", "sample_fit_epochs": null, "sample_noise_multiplier": 0}}, {"method": "S", "sample_size": 100000, "extra_planes": 6, "variant": "prefix_smooth", "sampling_options": {"sample_training": "prefix", "sample_format": "lwzq", "sample_fit_epochs": null, "sample_noise_multiplier": 0.5}}]}` |
| `svhn_temporal_control.json` | `{"name": "sampled_temporal_control_svhn_seed43_r5_20261007", "fit_device": "cuda:0", "cases": [{"method": "T", "variant": "masked"}]}` |

## Cleanup and retained verification

Only S/SR implementation, its codec tests and this report are retained from the sampled
investigation. Shared FL data/traces, the Python environment, numerical-policy commits,
protocol infrastructure, separate launch plans and the history-distortion work are preserved.
Dedicated sampled result/control directories, saved model and coordinate tensors, plots,
CSV/JSON measurements, problem caches, experiment modules/plans and diagnostic temporary
files are deleted. No sampled experiment worker, supervisor or plotter was running at the
cleanup audit; no research CUDA allocations remained to release. Unrelated system memory,
GPU desktop allocations and swap are not forcibly cleared.

Codec regression checks exercise received sample-only fitting, current normalization,
preceding-model acquisition, prefix heads, decoder sample replacement, rate/overhead
accounting, S/SR transport and protocol-owned reuse. Historical experiment/report tests and
measured figures were checked when those studies completed; their raw artifacts are now
intentionally unavailable. The retained test suite is rerun after cleanup.

Former report SHA-256 identities (text provenance only; no claim to preserve raw data):

- `RESULTS.md`: `0aad39ad200f0740bdafdc07c2bdfa002898cd632097985a3a7bef7047da7d18`
- `RESULTS_ELEMENT_AUDIT.md`: `be3ff10bb1a5521e79f10a7bcd49535c38e38dbab0347d95bafc8a2c7f709f10`
- `RESULTS_EXPANDED.md`: `b436f00d28dff6f6e0749b5d64e7b0c9b30a54e6dc1511740e4d8ec0a8496fbc`
- `RESULTS_O_REPRODUCTION.md`: `c882b778c1b7b96baa733706659a94f33dffcdc7c32a810020fe13a0e09afad2`
- `RESULTS_O_TF32_REPEATS.md`: `08de8fe9eb82cf8a44ed43b552e19a8efe9fe49b5bc18b3d5c4f84cd0bf758a6`
- `RESULTS_PRACTICAL.md`: `69984da4db81412201625d4e1e00416445085ae86216a156e235baf0eefb5b41`
- `RESULTS_RAW_OPTIMIZER_MEDIANS.md`: `1593d3df0c18044db56c85761a6cbed59644422b2b0a3a9580f1dca05434a73d`
- `RESULTS_RAW_ORACLE.md`: `f253257f8d7c48e559ebd2456ba4cbd0bf5097a76ea1f55c9441293ffcba038a`
- `RESULTS_RAW_PRECISION_MEDIANS.md`: `364d2c3a957c443d590e8795c6cd28677f058c1be31101ab0fe3bf5c17ae28a1`

Cleanup removed **1,962 dedicated files** across **71 selected paths**, releasing approximately **15.34 GiB** of allocated file blocks, plus **17 temporary files**. The mixed pre-fix Git stash was dropped after verifying the TF32 fix is committed on master and identically applied on the sampled branch. Retained S/SR regression suite: **34 passed**.

Post-cleanup validation: **63 passed** across retained S/SR, replay-memory and training-attempt-budget suites; **16 passed** for numerical-policy behavior (**79 total**). No dedicated sampled result directories, problem caches or experiment modules remain.
