# Round-shared quantizer normalization

The proposed schedule was tested on the same seed-44 ResNet18/SVHN trajectory
and measurement cohort as the original experiment: clients 0, 10, and 20 in
rounds 0–5, 201,269,556 raw local-update values. All 30 clients were admitted
to the experimental ledger in ascending order. The first admission of each
30-item group fits one scale and center per parameter tensor, using
`normalization_params(raw_layer)` directly. The other 29 admissions reuse those
factors. Every stored vector owns the factors needed for its inverse transform,
including after the next refresh. Existing production history is the baseline;
the new schedule is implemented in the experiment's `RoundNormalizedHistory`.

**The scheme improves global MAPE, but does not solve the small-value distortion.**
The existing seven variants reproduce the original measurements exactly.
The complete [per-layer MAPE table](PER_LAYER_MAPE.md) gives parameter tensors
equal weight and lists tiny-error tensors separately at the bottom.

| Storage | Global nonzero MAPE (%) | Global WMAPE (%) | Smallest-bin MAPE (%) | Worst layer-observation MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| Current per-client five-round factors | 0.119519 | 0.071690 | 227.787369 | 44.095190 |
| Proposed round-shared factors | 0.088735 | 0.070948 | 79.605429 | 14.140224 |
| Factors fitted freshly to each item | 0.088809 | 0.070915 | 83.218921 | 44.095190 |
| Uncentered 18-bit prefix | 0.069443 | 0.070233 | 0.000753 | 0.094834 |
| Cheap layer-scaled float16 | 0.017547 | 0.017610 | 0.271590 | 0.037072 |

The smallest bin is `0 < abs(value)/layer_RMS < 0.0001`, containing 8,809 values.
Global nonzero MAPE improves by 25.76%; WMAPE improves only about 1.04%.
The proposed scheme still changes all 125,042,809 exact zeros into nonzero
values, with maximum absolute error about 2.27e-7, and causes 760 sign flips.
Memory use is essentially unchanged: 1.7948 versus 1.7946 payload bytes/float.

Normalization and its inverse alone, with no prefix truncation, give global
WMAPE about 8e-7% and nonzero MAPE about 1e-6%. Smallest-bin MAPE is 0.001149%
in that control. It changes some zeros through float32 round-off, but the
maximum zero error is only 7.28e-12 and there are no sign reversals. This
isolates the main loss to truncation **after centering**: reversing the transform
cannot recover the bits discarded from a value near `-center/scale`.

MAPE here is averaged across strictly nonzero source values, without an epsilon.
It excludes exact zeros because ordinary percentage error is undefined there;
zero errors are counted separately. The protocol recorder instead averages
`abs(error)/(abs(source)+1e-8)` across every coordinate. These are different
metrics, so its MAPE should not be compared directly with this table.

The three-client distortion run measures about 2.26 seconds to admit and retrieve
client 0, versus about 0.50 seconds for measured later clients. Its mean of
1.09 seconds over three clients overrepresents the refresh by a factor of ten;
one complete round has one fit among 30 clients. The all-client timing pass
below provides the appropriate cost comparison. Timing excludes FL training.

Four focused tests pass, including exact quantizer-factor reuse, refreshing on
the submission count, and decoding an old payload after factors are refreshed.
Baseline equality, complete coverage, and magnitude-bin partitions were verified.
Results are storage diagnostics on raw updates; end-to-end codec RD and FL
convergence remain unmeasured. If this shared scheme is deployed across workers,
client 0's factors must reach any worker that needs an identical temporal history;
the first client's decoded reconstruction is not automatically available to
other workers. Server-only history can derive them locally. Retaining factors
inside each payload ensures retrieval but does not itself supply that information
to a separate worker.

Evidence: [summary](results/svhn_seed44_round_shared_20261005/summary.csv),
[layer measurements](results/svhn_seed44_round_shared_20261005/layers.csv),
[magnitude bins](results/svhn_seed44_round_shared_20261005/magnitude_bins.csv),
[selected-client timing](results/svhn_seed44_round_shared_20261005/storage.csv),
[configuration](results/svhn_seed44_round_shared_20261005/metadata.json).

![Matched round-shared comparison](results/svhn_seed44_round_shared_20261005/distortion.png)

## Time cost and refresh spike

The separate timing pass admits **all 30 clients** under both policies over all
six rounds, recording 360 admissions and 360 full-vector retrievals. It uses
the same trace identity and two Torch CPU threads; existing history uses up to
four layer-fitting/compression workers. Measurements are one-run wall latency,
excluding FL training and a separate CPU operator-profile pass.

| Policy / admission | Mean admission (seconds) | Mean full retrieval (seconds) |
| --- | ---: | ---: |
| Current history, cached factors | 0.3956 | 0.0767 |
| Current history, refreshing factors | 0.8918 | 0.0803 |
| Shared history, reused factors | 0.3935 | 0.0784 |
| Shared history, first item / refresh | 2.1487 | 0.0843 |

**The first shared item has a real latency spike:** about 1.73 seconds are spent
fitting its factors. For each of 62 layers, the quantizer estimator draws and
sorts five samples to estimate the center, then five more to estimate the scale.
Samples have up to 200,000 elements. That is **620 quantile calls and 620 sorts**
per fit. Sorting accounts for 71.77% of the profile's attributed self CPU operator
time, random sampling 12.04%, and indexing 5.27%. The profile's 1.244 seconds
of sort self time is a profiler measurement, separate from the unprofiled
1.734-second mean complete fit latency.

The proposed experiment fits the layers serially, following quantizer
preprocessing. Current history estimates layers concurrently with up to four
workers. This scheduling difference explains the larger one-item shared refresh
spike. Reusing factors across the other 29 submissions amortizes that cost.

The arithmetic transform itself averages only **12.97 ms per item** with shared
factors. Routine packing and ledger work average **0.3806 seconds per item**;
this phase includes float-prefix truncation, `np.unique` dictionary/rank creation,
adaptive rank packing, constructor and ledger overhead. These operations run on
every admission regardless of whether factors are refreshed. Over the shared
window, fitting totals 10.40 seconds, normalization 2.33 seconds, and packing plus
ledger work 68.63 seconds: packing is about 84% of admission time after fitting
has been amortized. Full retrieval averages 78.6 ms versus 77.9 ms for current
history, so inverse normalization does not introduce a comparable retrieval spike.

| Round | Current all-client admissions (seconds) | Shared all-client admissions (seconds) |
| --- | ---: | ---: |
| 0 | 26.37 | 13.36 |
| 1 | 11.83 | 13.84 |
| 2 | 11.98 | 13.55 |
| 3 | 11.59 | 13.45 |
| 4 | 12.07 | 13.81 |
| 5 | 27.14 | 13.36 |
| Total | 100.98 | 81.37 |

Shared admissions average **13.56 seconds per round**, versus **16.83 seconds**
for current history over this window, a 19.42% reduction. The window contains
two current-policy refresh rounds, with 60 per-client fits versus six shared
fits. Current non-refresh rounds take about 11.87 seconds; the shared policy
adds its one fit each round instead of concentrating fits in rounds 0 and 5.
Consequently the observed six-round speedup is not a steady-state estimate.
Totals including one full retrieval after each admission are 95.52 seconds
shared versus 115.01 seconds current. These are storage costs, not codec or FL
training runtime. Production's separate fitting latency is unmeasured because
its parallel fitting happens inside the normalization phase; missing timing
cells in the CSV remain missing.

The scheme therefore offers a modest distortion improvement and reduces
whole-window admission cost, while retaining a first-item pause and the centering
problem near zero. Uncentered prefix storage remains the simpler candidate for
both tiny-value fidelity and avoiding robust factor-fitting work. The existing
scale-only alternative similarly avoids centering cancellation and quantile sorts.

Timing evidence: [individual admissions](results/svhn_seed44_round_shared_timing_20261005/admissions.csv),
[round totals](results/svhn_seed44_round_shared_timing_20261005/rounds.csv),
[estimator CPU profile](results/svhn_seed44_round_shared_timing_20261005/fit_profile.csv),
[timing configuration](results/svhn_seed44_round_shared_timing_20261005/metadata.json).
