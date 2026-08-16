# Protocol round-range experiment

This experiment replays one protocol over a consecutive range of cached client-delta rounds. It
does not rerun federated learning. Decoder history is seeded from saved reconstructions, then each
client is encoded and decoded in the normal server order.

```bash
.venv/bin/python -m Experiments.Protocol_test.replay \
  --name wz_rounds_12_20 \
  --protocol wz_cancer \
  --start-round 12 \
  --end-round 20
```

Records are written beneath `exp_records/`; rerunning the command creates another numbered run
instead of overwriting an existing one. Plot that replay's per-round rate-distortion curves with:

```bash
.venv/bin/python Experiments/Protocol_test/plot.py \
  Experiments/Protocol_test/exp_records/run1_wz_rounds_12_20/compression_records.csv
```

The existing `exp_records/` and `analysis_plots/` contents are retained. `--history-source
raw_deltas` and `--replace-commits-with-raw` are explicit oracle controls; neither represents the
normal decoder state.

## Sparse placement variant

`sparse_replay.py` is the flexible variant for testing a protocol at selected locations in an FL
trace. A `ReplayPlan` contains any number of `ReplayCase` objects. Each case independently declares:

- its absolute `start_round`;
- the clients that are actually compressed;
- the exact ordered `round_codecs` schedule;
- its protocol and comparison identity.

Unselected clients still advance the decoder-visible history in normal within-round order, so a
sparse case observes the same surrounding FL trace rather than an artificial client subset.
`run_sparse_plan` distributes cases over GPUs, resumes only completed cases, and refreshes partial
aggregates, RD plots, and declared comparisons in the parent process as cases finish.

The command accepts one JSON `ReplayPlan`, keeping the placements and the runner separate:

```bash
.venv/bin/python -m Experiments.Protocol_test.sparse_replay sparse_plan.json
```

Paths in the JSON are resolved relative to that file. `context` names the saved tensors used for
the surrounding FL trace; `context_source` says whether they are decoder reconstructions or raw
deltas. `protocol_factory` may name an importable `module:callable` when a case needs more setup than
aligning its declared `round_codecs` with `start_round`.

The previous detached watcher implementation, multi-run comparison plotter, fixed multi-start
runner, experimental codecs, and unrelated regression scaffolding remain under
`Experiments/review_before_simplification/Protocol_test/` for review.
