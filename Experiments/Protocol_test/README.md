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
   every selected client's decoded reconstruction advances decoder-visible history in normal
   server order, while unselected clients advance it with raw deltas. Records land under
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
- `plans/o_svhn_30c_f9_warm.json` and `plans/t_svhn_30c_f9_warm.json` use those placements with the
  warm-pool protocol. Within each trained round, the first five workers build the shared cold pool
  and the remaining workers use warm starts; worker curves remain intentionally aggregated.
- `plans/or_tr_svhn_30c_start12_seed44.json` (+ `plans/svhn_30c_fl_config_seed44.json`) regenerates
  an independent seed-44 FL trajectory and compares OR with TR at start 12.

These plans train real 30-client quantizers and are not smoke tests.

## Reruns and recovery

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
  policy, the case's schedule (`protocol`, `start_round`, `client_ids`, `round_codecs`), the
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
  sparse start, then selected clients retain their decoded reconstruction so later rounds see the
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
