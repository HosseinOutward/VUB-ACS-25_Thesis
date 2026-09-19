# Cross-worker replay manifest

This is the portable record of the GPU replays used by the reports. Historical
`fl_config.json` and `protocol_config.json` files remain unchanged beside each records CSV; their
absolute output paths describe the original run and are not current rerun instructions.

Run commands from the repository root. Every run shares:

```bash
.venv/bin/python -u \
  Experiments/Final_experiment_reports/2026-08_cross_worker_side_information/experiment/replay.py \
  --deltas _/data/federated_learning/raw_client_deltas \
  --history-source raw_deltas --start-round 12 --end-round 20 --clients 5 \
  --fl-config Experiments/Final_experiment_reports/2026-08_cross_worker_side_information/experiment/replay_fl_config.json \
  --records-root Experiments/Final_experiment_reports/2026-08_cross_worker_side_information/results/records \
  --name NAME --protocol 'PROTOCOL' EXTRA_ARGS
```

The runner assigns a numeric `run` prefix to a new output directory. `NAME` below is the part after
that prefix in the preserved evidence directory.

| evidence | NAME | PROTOCOL | EXTRA_ARGS |
|---|---|---|---|
| [run5](records/run5_longf_o_elem/compression_records.csv) | `longf_o_elem` | `wz_cancer\|replace_O_with='O'\|extra_train_tokens=('dump_element_metrics',)\|train_at_round=12` | — |
| [run7](records/run7_longf_t_elem/compression_records.csv) | `longf_t_elem` | `wz_cancer\|replace_O_with='T'\|extra_train_tokens=('dump_element_metrics',)\|train_at_round=12` | — |
| [run8](records/run8_longf_o_seed44/compression_records.csv) | `longf_o_seed44` | `wz_cancer\|replace_O_with='O'\|extra_train_tokens=('dump_element_metrics',)\|train_at_round=12` | `--seed 44 --protocol-cache-dir _/data/protocol_cache_seed44` |
| [run9](records/run9_longf_n_nosi/compression_records.csv) | `longf_n_nosi` | `wz_cancer\|replace_O_with='N'\|extra_train_tokens=('dump_element_metrics',)\|train_at_round=12\|refresh_frozen_si=False\|retrained_prior_uses_server_si=False` | — |
| [run10](records/run10_longf_o_rawsi/compression_records.csv) | `longf_o_rawsi` | `wz_cancer\|replace_O_with='O'\|extra_train_tokens=('dump_element_metrics',)\|train_at_round=12` | `--replace-commits-with-raw` |
| [run11](records/run11_longf_om_gap1357/compression_records.csv) | `longf_om_gap1357` | `wz_cancer\|replace_O_with='OM'\|extra_train_tokens=('dump_element_metrics',)\|train_at_round=12\|train_offsets=(1,3,5,7)` | — |
| [run12](records/run12_longf_om_dense123/compression_records.csv) | `longf_om_dense123` | `wz_cancer\|replace_O_with='OM'\|extra_train_tokens=('dump_element_metrics',)\|train_at_round=12\|train_offsets=3` | — |
| [run15](records/run15_longf_od30/compression_records.csv) | `longf_od30` | `wz_cancer\|replace_O_with='OD'\|extra_train_tokens=('own_si_dropout=0.3',)\|train_at_round=12` | — |
| [run16](records/run16_longf_od60/compression_records.csv) | `longf_od60` | `wz_cancer\|replace_O_with='OD'\|extra_train_tokens=('own_si_dropout=0.6',)\|train_at_round=12` | — |
| [run17](records/run17_longf_od30_fo20/compression_records.csv) | `longf_od30_fo20` | `wz_cancer\|replace_O_with='OD'\|extra_train_tokens=('own_si_dropout=0.3','own_si_flag_only=0.2')\|train_at_round=12` | — |
| [run22](records/run22_longf_od_scr70/compression_records.csv) | `longf_od_scr70` | `wz_cancer\|replace_O_with='OD'\|extra_train_tokens=('own_si_scramble=0.7',)\|train_at_round=12` | — |
| [run23](records/run23_longf_od_scr40/compression_records.csv) | `longf_od_scr40` | `wz_cancer\|replace_O_with='OD'\|extra_train_tokens=('own_si_scramble=0.4',)\|train_at_round=12` | — |
| [run24](records/run24_longf_sparse10_o/compression_records.csv) | `longf_sparse10_o` | `wz_cancer\|replace_O_with='O'\|train_at_round=12` | `--sparsify-keep 0.1` |
| [run25](records/run25_longf_sparse10_t/compression_records.csv) | `longf_sparse10_t` | `wz_cancer\|replace_O_with='T'\|train_at_round=12` | `--sparsify-keep 0.1` |
| [run26](records/run26_longf_shared10_o/compression_records.csv) | `longf_shared10_o` | `wz_cancer\|replace_O_with='O'\|train_at_round=12` | `--sparsify-keep 0.025 --sparsify-shared-support` |
| [run27](records/run27_longf_shared10_t/compression_records.csv) | `longf_shared10_t` | `wz_cancer\|replace_O_with='T'\|train_at_round=12` | `--sparsify-keep 0.025 --sparsify-shared-support` |
| [run28](records/run28_longf_op3db/compression_records.csv) | `longf_op3db` | `wz_cancer\|replace_O_with='OP'\|extra_train_tokens=("split_mask='Experiments/Final_experiment_reports/2026-08_cross_worker_side_information/results/partitions/three_way_db.npy'",)\|train_at_round=12` | — |
| [run29](records/run29_longf_op3_crosseval/compression_records.csv) | `longf_op3_crosseval` | `wz_cancer\|replace_O_with='OP'\|extra_train_tokens=("split_mask='Experiments/Final_experiment_reports/2026-08_cross_worker_side_information/results/partitions/three_way_db.npy'",'cross_evaluate=True','dump_element_metrics')\|train_at_round=12` | — |
| [run30](records/run30_longf_op3_iter2/compression_records.csv) | `longf_op3_iter2` | `wz_cancer\|replace_O_with='OP'\|extra_train_tokens=("split_mask='Experiments/Final_experiment_reports/2026-08_cross_worker_side_information/results/partitions/three_way_reassigned.npy'",)\|train_at_round=12` | — |

The OD commands are historical provenance. Their own-SI dropout/flag implementation was removed
from the shared model; the experiment-local codec documents what must be restored before rerunning
them.

## Derived evidence

- M1's compact tables and plot are in [analysis/elements_o_vs_t](analysis/elements_o_vs_t/). The
  retained raw dumps live under the ignored `raw_elements/` directory and can be read with
  `../experiment/compare_elements.py`.
- T13 and T14 consume [three_way_db.npy](partitions/three_way_db.npy) and
  [three_way_reassigned.npy](partitions/three_way_reassigned.npy), respectively. The latter's
  retained summary is [three_way_reassigned.json](partitions/three_way_reassigned.json).
- `../experiment/build_wz_split.py`, `../experiment/reassign_split.py`, and
  `../experiment/analyse_split_structure.py` retain the analysis that is still supported by the
  available artifacts. The report README lists the portions that cannot currently be regenerated.
