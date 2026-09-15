# Sparse protocol replay

This experiment measures short protocol windows (one trained round, then several frozen-reuse
rounds) placed at selected points in one deterministic, uncompressed FL trajectory. It does not
run compressed FL: the full client population always trains and advances the trajectory,
independent of which clients are measured for compression.

## Flow

1. **Trace.** `Trace` (`FL_code/FL_core/trace.py`) generates or resumes the raw, uncompressed
   client deltas for every round any case needs, on one explicitly selected device. Generation
   happens once per run, before any case executes.
2. **Cases.** `sparse_replay.py` seeds each case's preceding raw history, then replays its window:
   every client's raw delta advances the decoder-visible history in normal server order; only the
   case's selected clients are actually encoded/decoded. Records land under
   `<output_root>/cases/<codec_id>/<case_id>/`.
3. **Records.** `records.py` is the shared completion contract: `validate_case_records` checks a
   case's CSV holds exactly its scheduled rows, and `read_completed_case` reads a finished case's
   `complete.json`, checks its identity, and enriches its rows with placement/phase metadata. The
   runner uses it to decide reuse; the analysis path uses it to read finished cases.
4. **Analysis.** `plot.py` reads every finished case through `records.read_completed_case` and
   writes `analysis/stage_points.csv`, `rate_distortion.png`, `rates_by_round.png`, and
   `comparison_gaps.csv`. Workers count equally: stage WMAPE is the mean worker percentage, then
   `20*log10(mean/100)`; stage rates are mean worker rates, then cumulative. Rate sources
   (`prior_stage_rates`, `coder_stage_rates`, `final_bytes`) are never substituted for one
   another. The gap is the existing interpolation-based `mean_interpolated_equal_distortion_gap`,
   not the repository's average-slope/perpendicular specification; single-stage, non-overlapping,
   or non-monotone curves are reported as unavailable rather than aborting.

Run a plan with:

```bash
.venv/bin/python -m Experiments.Protocol_test.sparse_replay Experiments/Protocol_test/plans/smoke.json
```

## Tracked plans

- `plans/smoke.json` (+ `plans/smoke_fl_config.json`): a tiny SVHN/resnet56/3-client plan for
  validating the pipeline end to end. Two windows (starts 5 and 8), workers `[0, 2]`, codecs `T`
  and `O` only, tiny quantizer settings. Output under
  `Experiments/Protocol_test/exp_records/refactor_smoke`.
- `plans/frozen_coder_rounds.json` (+ `plans/frozen_coder_rounds_fl_config.json`): the retained
  study (ResNet-56/CIFAR-10, 30 clients, starts 5 and 12, all 30 workers, codecs `T`, `TR`, `TRR`,
  `O`, `OSW`, `OSWR`). Its settings were recovered from
  `exp_records/binary_frozen/r56_cifar10/test_config.json` per the coordinator's decision
  (`REFACTOR_PLAN.md`, "Coordinator decisions", item 1). **This is a new study, not a reproduction
  of `binary_frozen`**: schedule placement (`protocol.set_schedule`), record identity
  (`complete.json`'s identity/provenance split), and trace generation (`Trace`, a new directory
  `r56_cifar10_30c_frozen_v2`) all differ from that run. Do not run this plan casually - it
  trains real ResNet-56 quantizers over 30 clients and two 13-round windows.

## Validation commands

```bash
.venv/bin/python -m pytest -q --import-mode=importlib tests/test_codec_lifecycle.py tests/test_history.py tests/test_trace.py tests/test_staged_transport.py tests/test_oracle_sw_mixin.py Experiments/Protocol_test/tests/test_plot.py Experiments/Protocol_test/tests/test_sparse_replay.py Experiments/slepian_wolf/test_oracle_sw_mixin.py Experiments/slepian_wolf/test_practical_sw.py
.venv/bin/python -m Experiments.Protocol_test.sparse_replay Experiments/Protocol_test/plans/smoke.json
.venv/bin/python -m Experiments.Protocol_test.sparse_replay Experiments/Protocol_test/plans/smoke.json
.venv/bin/python -m Experiments.Protocol_test.plot Experiments/Protocol_test/exp_records/refactor_smoke
```

The second `sparse_replay` invocation must reuse every completed case (identity match) without
retraining any quantizer. Reruns reproduce trained weights, symbols, rates, and WMAPE exactly, but
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
  the plan path, output root, cache root, `gpu_ids`, `cache_overwrite`, and the original plan JSON.
  This half is never compared for reuse: changing `gpu_ids`, the output root, the cache root, or
  `cache_overwrite` - or a source-only documentation commit - must not force retraining.

## Oracle assumptions

Every case records `oracle_assumptions` explicitly, rather than inferring them from a codec's
display label:

- `history_source = "raw_updates"`: every client's own raw delta is available to seed and advance
  history - a worker's own raw history is locally available.
- `history_replacement = "raw_after_measurement"`: a selected client's measured reconstruction is
  replaced by its raw delta immediately after compression is simulated, so later rounds see exact
  raw history rather than a lossy reconstruction.
- `client_order = "normal"`: clients are processed in ascending id order each round; a client sees
  its own preceding rounds and earlier clients' current-round updates, never a later client's or
  its own future update.
- `frozen_si_refresh`: whether a frozen round's side information is refreshed from advancing raw
  history (`protocol.refresh_frozen_si`); the quantizer's parameters stay frozen either way.
- `server_history = "exact_raw_granted"`: the server-visible ledger holds exact raw history for
  this study - not generally available after lossy transmission - which is why it is recorded as
  an explicit oracle grant rather than assumed.

Quantizer-training oracle access, source-fitted prior grants, and SW oracle search are per-token
facts and already live in `protocol_snapshot`'s `round_codecs[token]["codec_provenance"]`
(supplied by the codec/transport package); they are not duplicated here.
