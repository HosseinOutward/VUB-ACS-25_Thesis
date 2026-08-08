# Which integration and training changes made it stable?

Setup and caveats: [00_setup.md](00_setup.md). Partial-probe gaps are regenerated in
[archive_rd_gaps.csv](results/analysis/archive_rd_gaps.csv); incomplete probes are not promoted to
final comparisons.

## T7 — Initial expanded router
[`v2_initial_probe`](results/archive/index_aware_ablations/v2_initial_probe/),
[`v2_initial_partial`](results/archive/index_aware_ablations/v2_initial_partial/)

**Idea.** Increase hashes, routes, embedding width, RNN capacity and epochs together, following the
original router proposal, before adding conservative integration constraints.

**How.** Initial V2 uses 8×24 hashes, 64 routes, 24-wide embeddings and 112×8 RNNs for 180 epochs,
but allows a large learned routed scale, early sparse routing, content-dependent routing and equal
current/three-history exposure. Its coordinate transport is exact, while downstream conditioning is
still too strong.

**Result.** On matched client-0 rounds, OMI is `+0.01966 bit` vs O and TMI `+0.12924 bit` vs T;
the two-client training partial gives TMI `+0.06511 bit` (10.67× floor).

**Verdict.** Capacity and exact identity alone make the instability worse, especially for TMI.

## T8 — Raw index bits in the downstream RNN
[`v2_raw_bits_downstream`](results/archive/index_aware_ablations/v2_raw_bits_downstream/)

**Idea.** Guarantee precision by appending all 24 identity bits directly beside routed features.

**How.** The raw identity is available to encoder, decoder and in-model prior, creating a direct
coordinate-to-symbol path that bypasses the small residual router. OMI and TMI share the same path;
the run is stopped as soon as memorization is visible.

**Result.** OMI's single training row is `−0.00068 bit` vs O (0.11× floor); TMI's in-model training
prior collapses to `(0,0,0)` before a compression row is emitted.

**Verdict.** Direct raw bits are precise but unsafe: they let the prior memorize coordinate symbols.

## T9 — Bounded router-only lag-1 probes
[`OMI client-0 probe`](results/archive/index_aware_ablations/v2_router_only_winner_probe/index_aware_OMI/compression_records.csv),
[`TMI high-gate probe`](results/archive/index_aware_ablations/v2_router_only_high_gate/index_aware_TMI/compression_records.csv),
[`TMI near-T probe`](results/archive/index_aware_ablations/v2_near_t_winner_probe/index_aware_TMI/compression_records.csv)

**Idea.** Preserve exact identity inside the router while making the ordinary codec a recoverable
path, and reduce historical drift to lag 1.

**How.** Raw bits feed only the router; the downstream SI grows by the routed embedding width, not
by `24+embedding`. Routing stays dense, content routing is disabled, index dropout is 0.1, and a
sigmoid gate bounds the residual. TMI progressively reduces the cap from 0.05 to 0.001 and raises
current-task sampling to 95%.

**Result.** The nine-row OMI client-0 probe is `+0.00413 bit` vs O (0.68× floor). High-gate TMI is
`+0.04013 bit`; the three-row near-T probe improves to `+0.00133 bit` vs T (0.22× floor).

**Verdict.** Conservative routing creates plausible probes, but one client is not a generalization test.

## T10 — Two attempts plus constant-plane rejection
[`final OMI`](results/records/OMI/compression_records.csv),
[`final TMI`](results/records/TMI/compression_records.csv),
[`failed repeat-1 partial`](results/archive/index_aware_v2/index_aware_TMI/compression_records.csv)

**Idea.** Prevent a finite but useless all-constant quantizer from winning candidate selection.

**How.** Final indexed runs train two candidates and retry any candidate for which a refinement
plane emits only one symbol over the full nonconstant vector. The failed repeat-1 full attempt is
retained: client 1 emitted zero rate on every plane despite finite parameters and reconstruction.

**Result.** Failed partial: 1 zero-rate row out of 2. Final OMI/TMI: 0 zero-rate rows out of 45 each;
TMI nevertheless remains `+0.02417 bit` vs T (3.96× floor).

**Verdict.** The gate fixes collapse, not the frozen-generalization problem.

## M2 — Replacement O/T reproducibility
[`new O/T`](results/records/), [`previous O/T`](results/archive/baselines/),
[`baseline_reproduction.csv`](results/analysis/baseline_reproduction.csv)

**Idea.** Replace inherited baseline files with fresh runs under the exact final replay workflow and
measure whether that changes conclusions.

**How.** O and T are rerun for the same 45 keys, seed, raw-delta history and ordinary 74×6 setup in
fresh cache namespaces. Their conventional Protocol_test paths are compatibility symlinks to these
new records; prior files are retained under `results/archive/baselines`.

**Result.** New O is `+0.00261 bit` vs previous O (0.43× floor); new T is `−0.00772 bit` vs previous T
(1.27× floor).

**Verdict.** O reproduces within noise; T moves slightly beyond the O-derived floor, so all final claims use the new pair.
