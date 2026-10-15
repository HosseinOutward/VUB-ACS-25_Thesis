# History storage distortion

Run from the repository root:

```bash
.venv/bin/python -m Experiments.history_distortion.run \
  --output Experiments/history_distortion/results/svhn_seed44_round0_5_complete
.venv/bin/python -m pytest -q Experiments/history_distortion/test_distortion.py
```

Use a fresh output path on each run. Defaults regenerate the full seed-44 identity
ResNet18/SVHN trajectory on `cuda:0`, with all 30 clients contributing to every
server update. Measure clients 0, 10, and 20 in rounds 0–5. These are raw local
model deltas after local SGD, the vectors sent to the FL compressor. They are not
individual minibatch gradients. No trajectory checkpoint is read or written.
The six-round window covers factors fitted in round 0, reused in rounds 1–4,
and refreshed in round 5. History retention is one entry per client to bound
memory; its factor refresh policy is the production policy and does not depend
on retention depth.

The production WZ protocol constructs `CompressedReconstructionHistory` with
the model's trainable parameter slices. Each vector is layer-normalized with a
robust fitted scale and center, then its float32 bit pattern is truncated to
18 leading bits: sign, eight exponent bits, and nine mantissa bits. Adaptive
dictionary/rank packing preserves those truncated values exactly. Even a chunk
stored in its `raw` layout has already undergone this truncation. The plain
`ReconstructionHistory`/`HistoryVector` implementation copies float32 exactly.
This experiment measures the additional history storage error relative to its
input; quantizer reconstruction error is a separate source of distortion.

| Variant | Transform before storage | Representation |
| --- | --- | --- |
| `float32` | None | Exact `HistoryVector` |
| `prefix_raw` | Identity factors | Actual adaptive history packing, 18-bit prefix |
| `current_5round` | Production robust scale + center, refreshed every five rounds | Actual compressed ledger |
| `current_fresh` | Production robust scale + center, fitted every vector | Actual compressed vector |
| `prefix_pow2` | Per-layer power-of-two scale; zero center | Actual adaptive history packing, 18-bit prefix |
| `float16_raw` | None | Float16 round trip |
| `float16_pow2` | Per-layer power-of-two scale; zero center | Float16 round trip + one float32 scale per layer |
| `round_shared` | Exact quantizer estimator on the first submission of every 30, shared across that round | Actual compressed history vector |
| `round_shared_float32` | Same shared factors | Normalize and invert in float32, without prefix truncation (diagnostic) |

The round-shared experiment admits **all** clients in their normal ascending
order, including clients not selected for measurement. Its admission counter
fits before submissions 0, 30, 60, and so on; client 0 supplies the new factors
in each round. It uses `normalization_params(raw_layer)` directly, just as
quantizer preprocessing does, then stores `(value-center)/scale`. Each compressed
payload retains its own factors, so a later refresh does not change retrieval
of an earlier item. Factors are fitted from the first current submission, not
from future client submissions or a pooled round average. This is an experiment
implementation in `RoundNormalizedHistory`; the active production ledger remains
the baseline under comparison.

The original seven-variant results are in `svhn_seed44_round0_5_complete`.
The follow-up is a separate matched output:

```bash
.venv/bin/python -m Experiments.history_distortion.run \
  --output Experiments/history_distortion/results/svhn_seed44_round_shared_20261005
```

Its float32 transform-only control isolates normalization arithmetic from the
loss caused by the 18-bit prefix. Unlike the exact float32-history baseline,
normalizing and reversing in float32 can itself introduce small round-off.

For matched **all-30-client** timing, including the refresh spikes:

```bash
.venv/bin/python -m Experiments.history_distortion.timing \
  --output Experiments/history_distortion/results/svhn_seed44_round_shared_timing_20261005
```

This second pass records admission and retrieval separately for every client,
splits shared-factor fitting from normalization and packing, and profiles the
CPU operators used by the first fit. Production normalization's phase includes
its concurrent fitting; its separate fit latency is left missing rather than
inferred. All latency measurements exclude FL training and the operator-profile
pass. `packing_and_ledger_seconds` is the remainder of admission latency after
the measured transform/fitting phases, so it includes constructor and ledger
overhead. `rounds.csv` sums admissions in their sequential client order. The
three-client distortion report's timing mean overrepresents the shared-factor
refresh: it includes one refresh per three measured clients, whereas a complete
round includes one per 30. Use the all-client timing tables for cost comparisons.

The cheap scale is `2**floor(log2(max(abs(layer))))`, or one for an all-zero
layer. A decoder needs the stored scale to invert this transform. It can be
computed from a decoder-visible reconstruction without worker-only knowledge.
Power-of-two scaling preserves float32 prefix truncation for ordinary normal
values, while shifting tiny values into float16's representable range. Centering
can increase the absolute quantization error near raw zero: the stored value is
near `-center/scale`, and restoring it subtracts nearly equal quantities.
Downstream normalization cannot recover precision already discarded by storage.

`layers.csv` has every parameter tensor, client, round, and variant.
`per_layer_mape.csv` pools each parameter tensor's nonzero percentage errors
across the full measurement window for current and round-shared storage. Rows
are sorted by current MAPE, with tensors below 0.1% in both methods listed last.
The mean of these per-tensor MAPEs gives each tensor equal weight.
`magnitude_bins.csv` additionally splits each tensor by `abs(value)/layer_RMS`;
exact zeros have their own bin. `summary.csv` pools additive statistics before
forming ratios. It also includes mean and worst layer-observation WMAPE, keeping
small parameter tensors visible. `storage.csv` contains resident tensor payload
bytes and single-run round-trip time, including fitting and restoration. These
bytes exclude Python object overhead and are **local memory, not communication
rate**. Timing is diagnostic, not a repeated performance benchmark.

WMAPE is `100*sum(abs(error))/sum(abs(source))`. MAPE is the mean
`100*abs(error)/abs(source)` over strictly nonzero source values, with no epsilon
or clipping. For all-zero sources relative metrics are missing. Errors on exact
zeros remain in the WMAPE numerator and are counted separately. Also record
nonzero values lost to zero, sign reversals, relative L2 error, maximum error,
and each observation's 99th percentile absolute percentage error. Percentiles
in CSV are per observation; they are not pooled by averaging them.
This MAPE definition differs from the protocol recorder's full-vector
`abs(error)/(abs(source)+1e-8)` metric; the epsilon and inclusion of exact zeros
make its reported MAPE unsuitable for direct comparison with these numbers.

`distortion.png` and `distortion.pdf` show per-layer WMAPE against raw layer RMS,
and nonzero MAPE by within-layer magnitude. This is a storage diagnostic, not
an RD comparison or an FL convergence experiment: no learned codec or entropy
coder is trained, and no storage variant feeds back into the trajectory.

Completed measurements and interpretation are in [REPORT.md](REPORT.md).

For two capped histograms—one history entry and full server SI before downstream
normalization—run:

```bash
.venv/bin/python -m Experiments.history_distortion.histogram \
  --output Experiments/history_distortion/results/svhn_seed44_raw_full_si_histogram_20261006
```

The single entry is client 10, round 4. Full SI contains all 30 clients' five
entries from rounds 0–4, the server snapshot at the start of round 5. Compare
exact raw updates against current and round-shared compressed storage. The SI
histogram reads the actual `SideInfoMatrix` with identity target factors, exposing
restored raw values before downstream target normalization. It streams eight
columns and 200,000 rows per batch, covering all 150 entries without allocating
the complete dense matrix on the accelerator.

`summary.csv` counts coordinates that differ numerically, including changed exact
zeros. It also reports the fraction of strictly nonzero coordinates with absolute
percentage error above 0.01%, 0.1%, 1%, 10%, and 100%. `histogram_bins.csv` contains
every bin's exact count and percentage of nonzero reference coordinates, including
zero-error/underflow and overflow bins. No epsilon is used. Histograms use a log
x-axis and a y-axis capped at 0.5% per bin by default (`--y-max` changes that
limit). Clipped peak heights and the proportion of exact zeros are annotated.
Zero-reference errors are reported separately instead of assigned an artificial
percentage. Full-SI coordinates are matrix cells: each model parameter has 150
history occurrences, rather than being counted once. These are storage errors on
real identity-FL updates, separate from codec reconstruction error.
