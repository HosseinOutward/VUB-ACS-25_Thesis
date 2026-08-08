# Index-aware multi-round replay manifest

This is the portable record of the six primary GPU replays, the sparse encoder-history extension,
and every retained exploratory run.
The JSON snapshots beside each CSV remain unchanged; absolute paths and implementation module names
describe where the run was produced before this investigation was consolidated.

Run from the repository root. To reproduce without replacing retained evidence, choose a new
`--records-root` containing neither the V1 archive nor these final records:

```bash
.venv/bin/python -u \
  Experiments/Final_experiment_reports/2026-08_index_aware_multi_round_codec/experiment/replay.py \
  --profile v2 --mode MODE --start-round 12 --end-round 20 --train-offset 1 \
  --deltas _/data/federated_learning/raw_client_deltas \
  --fl-config Experiments/Final_experiment_reports/2026-08_index_aware_multi_round_codec/experiment/replay_fl_config.json \
  --records-root /tmp/index_aware_multi_round_reproduction \
  --cache-root _/data/protocol_cache/flagship_comparison
```

`--train-offset 1` is ignored by O/T, selects the one lagged problem for OM/TM/OMI/TMI, and is
recorded explicitly to keep all commands uniform. O/T/OM/TM use ordinary WZ defaults even under the
V2 experiment profile. OMI/TMI use the pinned indexed profile in their manifests.

| evidence | MODE | protocol | distinguishing setup |
|---|---|---|---|
| [O](records/O/compression_records.csv) | `O` | `wz_cancer\|train_at_round=12` | ordinary oracle, no lag |
| [T](records/T/compression_records.csv) | `T` | `wz_cancer\|replace_O_with='T'\|train_at_round=12` | ordinary temporal, no lag |
| [OM](records/OM/compression_records.csv) | `OM` | `wz_cancer\|replace_O_with='OM'\|train_at_round=12\|train_offsets=(1,)` | ordinary oracle, lag 1 |
| [TM](records/TM/compression_records.csv) | `TM` | `wz_cancer\|replace_O_with='TM'\|train_at_round=12\|train_offsets=(1,)` | ordinary temporal, lag 1 |
| [OMI](records/OMI/compression_records.csv) | `OMI` | `wz_cancer\|replace_O_with='OMI'\|train_at_round=12\|train_offsets=(1,)` | V2 index, 75/25, gate ≤0.05 |
| [TMI](records/TMI/compression_records.csv) | `TMI` | `wz_cancer\|replace_O_with='TMI'\|train_at_round=12\|train_offsets=(1,)` | V2 index, 95/5, gate ≤0.001 |

## Sparse encoder-history extension

The exact case/configuration manifest is
[test_config.json](sparse_encoder_history/records/test_config.json). The queue is resumable and
skips every complete 13-row case:

```bash
.venv/bin/python -u \
  Experiments/Final_experiment_reports/2026-08_index_aware_multi_round_codec/experiment/index_aware_codec/sparse_encoder_history_queue.py
```

| evidence | role |
|---|---|
| [OM](sparse_encoder_history/records/OM/compression_records.csv) | oracle multi-round control |
| [OM_FE](sparse_encoder_history/records/OM_FE/compression_records.csv) | OM with local history also at encoder |
| [TM](sparse_encoder_history/records/TM/compression_records.csv) | temporal multi-round control |
| [TM_FE](sparse_encoder_history/records/TM_FE/compression_records.csv) | TM with local history also at encoder |

O, T, OMI and TMI sparse records are retained beside these four because they were produced by the
same queue and manifest. Per-case records, resolved configurations and worker logs remain under
`sparse_encoder_history/records/`; the reusable quantizer cache remains external at
`_/data/protocol_cache/sparse_codec_queue`.

## Retained exploratory evidence

| evidence | purpose | status |
|---|---|---|
| [index_aware_v1](archive/index_aware_v1/) | OI/TI and dense-history OMI/TMI | complete 45-row runs |
| [v2_initial_partial](archive/index_aware_ablations/v2_initial_partial/) | first expanded two-client training check | partial |
| [v2_initial_probe](archive/index_aware_ablations/v2_initial_probe/) | first expanded client-0 frozen check | partial |
| [v2_raw_bits_downstream](archive/index_aware_ablations/v2_raw_bits_downstream/) | direct raw identity downstream | stopped on TMI collapse |
| [v2_router_only_high_gate](archive/index_aware_ablations/v2_router_only_high_gate/) | router-only residual ≤0.05 | partial |
| [v2_conservative_gate](archive/index_aware_ablations/v2_conservative_gate/) | TMI residual ≤0.01 | partial |
| [v2_near_t_winner_probe](archive/index_aware_ablations/v2_near_t_winner_probe/) | TMI residual ≤0.001, 95/5 | three-row probe |
| [v2_router_only_winner_probe](archive/index_aware_ablations/v2_router_only_winner_probe/) | OMI residual ≤0.05, 75/25 | client-0 rounds 12-20 |
| [interrupted V2 partial](archive/index_aware_v2/) | first repeat-1 five-client attempt | OMI partial; TMI zero-rate row |
| [previous baselines](archive/baselines/) | O/T files superseded by final reruns | complete 45-row controls |

## Derived evidence

Run:

```bash
.venv/bin/python \
  Experiments/Final_experiment_reports/2026-08_index_aware_multi_round_codec/experiment/analyse_rd.py
```

This validates identical 45-row final coverage and regenerates:

- [rd_gaps.csv](analysis/rd_gaps.csv): training, frozen and full matched-curve gaps.
- [rd_curves.csv](analysis/rd_curves.csv): each averaged three-stage curve.
- [archive_rd_gaps.csv](analysis/archive_rd_gaps.csv): every retained partial probe on matched keys.
- [baseline_reproduction.csv](analysis/baseline_reproduction.csv): new O/T against the superseded pair.

For the sparse extension, run:

```bash
.venv/bin/python \
  Experiments/Final_experiment_reports/2026-08_index_aware_multi_round_codec/experiment/analyse_sparse_encoder_history.py
```

This regenerates the RD plots plus
[encoder_history_gaps.csv](sparse_encoder_history/analysis/encoder_history_gaps.csv),
[repeat_gaps.csv](sparse_encoder_history/analysis/repeat_gaps.csv), and the O/T-family frozen-age
comparison table under `sparse_encoder_history/analysis/`.
