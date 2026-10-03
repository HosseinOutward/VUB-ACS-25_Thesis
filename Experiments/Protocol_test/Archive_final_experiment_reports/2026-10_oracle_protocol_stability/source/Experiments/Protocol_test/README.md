# Sparse protocol replay

This experiment measures explicit protocol windows (including optional retraining within a window)
placed at selected points in one deterministic, uncompressed FL trajectory. It does not
run compressed FL: the full client population always trains and advances the trajectory,
independent of which clients are measured for compression.

## Prior and coder columns

`quantizer_prior_rate` and `quantizer_prior_stage_rates` measure the prior produced by joint
quantizer training, before fixed-symbol fine-tuning. Prior weights are never part of the payload:
the prior stands in for the conditional entropy coder and is treated as decoder-available. The
fine-tuned model is identified by its SI:
`temporal_si_prior_rate` or `server_si_prior_rate`, both scalar averages. `marginal_rate` is also a
scalar average. Rounds with rANS retain the selected transport in `coder_rate` and
`coder_stage_rates`; SI-specific comparisons are recorded separately as `temporal_si_rans_*` and
`server_si_rans_*`. O/OR use server SI natively, while T/TR use temporal SI natively; T/TR may also
report the separately trained server-SI comparison.

## Flow

1. **Trace.** `Trace` (`FL_code/FL_core/trace.py`) regenerates the raw, uncompressed client deltas
   for every round a case needs, on one explicitly selected device. Cases run in order of their
   first trace round, and each leaves one continuation state (`_/data/trace_states/<trace identity>.pt`,
   overwritten in place) where the next case begins, so that case resumes instead of retraining
   from round zero. A state later than a case's first round is ignored.
2. **Cases.** `sparse_replay.py` seeds each case's preceding raw history, then replays its window:
   every client of the trace is measured, and its decoded reconstruction advances decoder-visible
   history in normal server order. A case's `round_codecs` is one line of space-separated tokens,
   where `token*n` repeats a token a positive integer `n` times, e.g.
   `"TR|bins_per_plane=2|num_planes=3 F*9"`. Records land under
   `<output_root>/cases/<codec_id>/<case_id>/`. Before each measured client-round the global RNG is
   reseeded from the plan seed, start round, round, and client, so fits are reproducible yet
   differ across clients and rounds.
3. **Records.** `records.py` is the shared completion contract: `validate_case_records` checks a
   case's CSV holds exactly its scheduled rows, and `read_completed_case` reads a finished case's
   `complete.json`, checks its identity, and enriches its rows with placement/phase metadata;
   `read_available_case` reads the validated prefix of an unfinished case. The runner uses them
   to decide reuse; the analysis path uses them to read cases.
4. **Analysis.** `plot.py` reads every finished case, plus the prefix of unfinished ones (figures
   then carry an `_INCOMPLETE` suffix and `coverage.csv` lists what is missing), and writes
   `analysis/stage_points.csv`, `comparison_gaps.csv`, `timing_summary.csv`, and one figure
   directory per rate source (`analysis/quantizer_prior_stage_rates/`,
   `analysis/coder_stage_rates/`) plus
   `analysis/combined_rate_sources/`, which draws both staged sources against one shared x axis.
   Every directory holds the same `rd_overall`, `rd_by_start`, `rd_by_phase` (trained round beside
   the frozen-round average) and, for multi-cycle plans, `rd_cycle_<n>` views. Workers count equally: stage WMAPE is the mean worker ratio (records store
   percentages), then `20*log10(mean)` dB; stage rates are mean worker rates, then cumulative. Rate sources
   (`quantizer_prior_stage_rates`, `coder_stage_rates`, `final_bytes`) are never substituted for one
   another; a case lacking the quantizer-prior rate or payload size fails, so records from an older
   column layout cannot silently drop out of the figures. The gap is the existing interpolation-based `mean_interpolated_equal_distortion_gap`;
   single-stage, non-overlapping,
   or non-monotone curves are reported as unavailable rather than aborting.

Run a plan with:

```bash
.venv/bin/python -m Experiments.Protocol_test.sparse_replay Experiments/Protocol_test/plans/tr_svhn_30c_f9.json
```

## Global view

`plot.py --global <container>` discovers every experiment output root under `<container>` that has
written case records and draws them all into `<container>/_global_analysis`: the same figure
directories and views as one experiment's `analysis/`, plus the pooled `stage_points.csv`, with
every experiment on one axis. Coverage, timing and gap tables stay with the experiment that owns
them. Codec is the line hue, sparse start an ordinal shade of that hue (also written at each curve
end), experiment the dash pattern, and rate source the marker; `initial_si` is named in the legend
only. O, OR, T, and TR use a stable color lexicon; other codec names use the remaining palette in
sorted order.

```bash
.venv/bin/python -m Experiments.Protocol_test.plot Experiments/Protocol_test/Runs_exp --global
```

Keep it live with its own watcher, launched independently of any run, since experiments under the
container start and finish at different times. It refreshes on change and runs until stopped:

```bash
systemd-run --user --collect --unit=fl-wz-global-plot --service-type=exec \
  --working-directory=$PWD .venv/bin/python -u -m Experiments.Protocol_test.live_plot \
  Experiments/Protocol_test/Runs_exp --global
```

## Tracked plans

- `plans/or_svhn_30c_f9.json` and `plans/tr_svhn_30c_f9.json` share the seed-43
  SVHN/ResNet-18 trace. Each measures all 30 clients at starts 5, 12, and 20 with one trained round
  followed by nine frozen rounds.
- `plans/o_svhn_30c_f9_warm.json`, `plans/t_svhn_30c_f9_warm.json`, `plans/or_svhn_30c_f9_warm.json`,
  and `plans/tr_svhn_30c_f9_warm.json` use those placements with the warm-pool protocol. Within each trained round, the first five workers build the shared cold pool
  and the remaining workers use warm starts; worker curves remain intentionally aggregated.
- `plans/or_tr_svhn_30c_start12_seed44.json` (+ `plans/svhn_30c_fl_config_seed44.json`) regenerates
  an independent seed-44 FL trajectory and compares OR with TR at start 12.

These plans train real 30-client quantizers and are not smoke tests.

## Experimental multi-round codecs

`experimental_codecs.py` registers OM/TM and their rANS variants OMR/TMR. The experimental
`wz_cancer_multi` and `wz_cancer_multi_warm` protocols train on the current raw delta plus
the same client's raw deltas at `train_offsets=(1,2)` by default. The corresponding
`train_weights=(0.5,0.3,0.2)` sample current/previous/second-previous pairs at 50/30/20 shares.
Each historical target is
paired with a shallow, immutable reconstruction-history snapshot captured immediately before
that client was encoded. OMR reads the snapshot's server SI; TMR reads its temporal SI. Current
SI remains live, including earlier clients' decoded updates in server order. Each training
problem is normalized using its own target, and problems are sampled with the declared shares and the
ordinary epoch/sample budget. SI blocks remain lazy instead of allocating a dense stacked matrix.

The sparse runner supplies the raw targets; these protocols are intended for sparse replay.
It seeds `5 + max(train_offsets)` preceding rounds, so lag 2 needs a start of at least 7.
`plans/omr_svhn_30c_f9_w503020_gate.json` and `plans/tmr_svhn_30c_f9_w503020_gate.json` use starts 7, 12, and 20
on the unchanged seed-43 `cuda:0` trace, all 30 clients, 2 bins x 3 planes, 2048 rANS lanes,
and nine frozen rounds. The existing five-parent warm pool and stabilized quantizer defaults
apply. OMR transports server-SI rANS; TMR transports temporal-SI rANS and separately records
the full-server-SI rANS comparison. Starts 12 and 20 also align with the earlier OR/TR plans.

All cold/warm quantizer and fixed-symbol prior candidates must have finite loss and total
hard-symbol prior rate no greater than the empirical marginal rate of those same symbols
before entering the existing lowest-loss argmin. Quantizers are checked on the full current
target (the leading block); priors on all symbols supplied to their fit. The existing bounded
retry budget still applies, and exhaustion fails with rejected rates instead of silently
deploying a marginal or invalid model. Frozen priors are not refit or gated on future data.
Joint OM/TM training and all trained-round prior fits use the three weighted pairs. Historical
symbols are encoded once by the final encoder, and both native fine-tuning and TMR's independent
full-SI prior fit use each target's capture-time SI, normalized for that target. Prior acceptance
compares the 50/30/20 weighted conditional rate with the same weighted per-round marginal rates.
Reported rates and distortion still measure only the current transmitted target.
The acceptance policy and mixture are recorded in semantic identity; old results cannot be
reused under the changed behavior. New plans use separate output directories, preserving all
stopped lag-1 outputs. Historical lag-1 plans need explicit `train_weights=(0.5,0.5)` if used
with the new protocol, and a new output directory for the changed acceptance policy.

```bash
.venv/bin/python -m pytest Experiments/Protocol_test/test_experimental_codecs.py -q
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch Experiments/Protocol_test/plans/omr_svhn_30c_f9_w503020_gate.json
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch Experiments/Protocol_test/plans/tmr_svhn_30c_f9_w503020_gate.json
```

`smoke_multi_round.py <plan>` checks one full-size real-trace client with the plan's actual
training schedule, mixture, and marginal acceptance gate. Its output under
`<experiment>/smoke/` verifies execution and memory before a full sweep.
Run this probe through a transient user service too. Production records and supervisor health
remain under each plan's separate `runs/` and `runs_control/` directories.

## Reruns and recovery

### Full-vector fallback confirmation

`full_vector_confirmation.py` follows the sampled diagnostic with the unchanged full-vector
training budget: cold 140 epochs/best of three, 550,000 sampled positions per epoch and 6,250-row
joint batches; embedded-T warm fits use the default 35 epochs/two accepted attempts. Targets
7/6/5 and their capture-time SI use 50/30/20 sampling in quantizer and fixed-symbol prior fits.
The baseline is a real all-30-client TMR replay with a five-slot/two-parent warm pool; its decoded
updates advance one common history for all diagnostic candidates. Clients 0/5/11/21 retain T,
the exact T-in-O embedding, warm O-from-T, and cold O. Every joint prior remains a candidate;
fine-tuning cannot overwrite it. The embedding also retains T's fine-tuned temporal prior,
zero-extended to full SI, so the trained temporal baseline is representable too.

Round8 is deliberately chronological validation, not training or future evaluation. For each
fixed encoder choose its prior by validation hard-symbol NLL; then choose O's quantizer by the
stage-averaged normalized-MSE-plus-native-NLL objective. Rounds9–16 are frozen evaluation only,
with selections persisted before evaluating them. This diagnostic selection uses raw validation
targets and oracle full SI; it is not yet a deployable worker-only selection protocol. All models
and prior policies are evaluated, not just winners. Actual 2048-lane rANS verifies decoded symbols.

This reproduces a decoded warm-pool trajectory and full production update budget, but it is a
common-history mechanism confirmation, not independently evolving O/T protocols or a sweep
over all clients/seeds. The candidate comparison adds cold and warm fitting costs; report them
separately from baseline replay cost. Every-round prior fine-tuning is NOT enabled in this test.
Invalid candidate budgets still fail loudly; previously retained tensor checkpoints survive.

Run **alone**, not concurrently with another quantizer job, through:

```bash
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch Experiments/Protocol_test/plans/full_vector_confirmation_retry8.json \
  --worker-module Experiments.Protocol_test.full_vector_confirmation \
  --plotter-module Experiments.Protocol_test.full_vector_confirmation
```

Outputs are `Runs_exp/full_vector_confirmation_retry8/runs/`: manifest/provenance, all baseline
client-round CSVs, tensor-only candidate checkpoints, round8 selection files, every stage RD
point and failure, equal-distortion gaps, full rate tables, and live plots. The custom worker
refuses an existing output root: preserve any stopped run and use a new root before restarting.
The shared supervisor retains its normal health checks. Stop only with
`systemctl --user stop fl-wz-full-vector-confirmation-retry8`.

The stopped first confirmation exhausted its warm quota at client21: one of four attempts
passed for parent1, whereas two accepted candidates are required. The revised
`plans/full_vector_confirmation_retry8.json` uses the experiment-only `wz_cancer_confirmation`
protocol and raises quantizer rejection slots from two to six: warm maximum attempts per
parent rise from four to eight, cold from five to nine. Fits still stop immediately once
their original accepted quota is met. Prior retries, acceptance checks, optimizer settings,
and production defaults are unchanged. Its fresh output root preserves the first run's
checkpoints and partial records under `Runs_exp/full_vector_confirmation/runs/`. The command
above launches the revised plan; its unit is `fl-wz-full-vector-confirmation-retry8`.

### Controlled training diagnostic

`training_diagnostic.py` screens clients 0 (control), 5, and 11 on common compressed **raw**
history, so T's five SI columns are an exact subset of O's 150. This is deliberately not an
exact reproduction of warm-pool failures or a substitute for full-vector protocol results.
Full-vector target normalization precedes fixed sampling (16,384 training coordinates and
4,096 held-out coordinates per round). The unchanged model, learning rate, epoch defaults,
and batch size apply, but the smaller dataset means fewer optimizer updates per epoch.
Every seed (43/44/45) is retained for T, O, and O initialized by exact T-in-O embedding;
the latter then uses the default 35-epoch warm schedule. Each learns targets at rounds 6 and 5.
Prior comparisons preserve the joint prior, fine-tune on current symbols, or fine-tune on both
targets encoded by the **final encoder**. Both temporal and full-SI priors/rANS are evaluated;
current-only rows are repeated to match the two-round fit's optimizer-update budget in this sample.
an alternate-SI prior is a scratch fit, as in the protocol, not the preserved joint prior.
Historical held-out coordinates at rounds 5/6 support selection auditing; rounds 7/8/12/13 are
evaluation only. Sample rANS includes its large 2048-lane flush, recorded separately, and must
not be compared numerically to full-vector payload rates without accounting for that overhead.
Models, every attempt, stage tables, equal-distortion gaps, selection scores, and live RD plots
are retained under `Runs_exp/training_diagnostic/runs/`.

```bash
.venv/bin/python -m Experiments.Protocol_test.detached_replay launch Experiments/Protocol_test/plans/training_diagnostic.json \
  --worker-module Experiments.Protocol_test.training_diagnostic \
  --plotter-module Experiments.Protocol_test.training_diagnostic
```

The shared detached supervisor provides the normal scheduled health checks. Stop with
`systemctl --user stop fl-wz-training-diagnostic`; original stopped OMR/TMR outputs are preserved.

A repeated `sparse_replay` invocation reuses every completed named case artifact after an identity
match, without recompression. Quantizer and conditional-prior training always run. An interrupted
case is recovered only when its pre-run identity matches and its full CSV validates; other partial
output requires inspection or a new output directory. Seeded reruns reproduce trained weights,
symbols, rates, and WMAPE exactly, but
`encoder_size`, `decoder_size`, and `final_bytes` jitter by ~0.1%: they are gzip'd pickles, and
pickling bitwise-equal torch tensors does not yield identical bytes.

## Identity versus provenance

Every `complete.json` (per case) and `test_config.json` (per experiment) stores two separate
parts:

- **`identity`**: the trace identity (`Trace.identity()`), the experiment seed, the codec-seed
  policy, the case's schedule (`protocol`, `start_round`, `client_ids` (every trace client),
  `round_codecs`), the
  JSON-round-tripped `protocol.config_snapshot()`, `history_representation`, and
  `oracle_assumptions`. This is the *only* part compared for reuse: a case is reused iff its
  stored identity exactly equals the freshly computed one. A mismatch fails loudly, naming the
  case and the differing top-level keys - it never silently retrains or overwrites.
- **`provenance`**: git `HEAD`, a fingerprint (and file list) of relevant uncommitted source under
  `FL_code/` and `Experiments/Protocol_test/`, torch/CUDA versions, the GPU names actually used,
  the plan path, output root, `gpu_ids`, and the original plan JSON. This half is never compared
  for reuse: changing `gpu_ids` or output location
  does not alter codec semantics.

Each case also writes a small atomic `pre_run_identity.json` before replay starts. If a replay is
interrupted after its CSV is complete but before `complete.json`, recovery compares this sidecar
with the freshly computed identity before marking the case complete.

## Oracle assumptions

Every case records `oracle_assumptions` explicitly, rather than inferring them from a codec's
display label:

- `initial_history_source`: the five-round seed history is either raw updates or the declared
  n-split reconstruction; a worker's own seeded history remains locally available.
- `history_replacement = "decoded_updates_after_start"`: history begins with raw deltas before the
  sparse start, then clients retain their decoded reconstruction so later rounds see the
  protocol trajectory being measured.
- `client_order = "normal"`: clients are processed in ascending id order each round; a client sees
  its own preceding rounds and earlier clients' current-round updates, never a later client's or
  its own future update.
- `frozen_prior_refit`: whether frozen rounds refit their priors (`protocol.refit_priors_on_frozen`).
  Frozen rounds always code with the most recent side information; only parameters stay frozen.
- `server_history = "decoder_visible_reconstructions"`: after the raw prefix, the server-visible
  ledger holds the decoded values the protocol actually produced.

Quantizer SI access, prior-weight transmission, and SW oracle search are per-token
facts and already live in `protocol_snapshot`'s `round_codecs[token]["codec_provenance"]`
(supplied by the codec/transport package); they are not duplicated here.
