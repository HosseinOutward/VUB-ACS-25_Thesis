# Federated-learning gradient compression

This repository studies two-step compression: a learned Wyner–Ziv quantizer followed by a conditional entropy coder. Decoder-side side information (SI) determines which priors and operations are legitimate.

- `FL_code/` — reusable FL, quantization, and codec implementation.
- `Experiments/` — studies and analysis; `Protocol_test/Archive_final_experiment_reports/`, `Archive/` (including the former `hyper_param_search/`), and `statistical_relation_boosting/` are frozen.
- `archived_tests/` — git-ignored tests written while building features; run them by path, e.g. `.venv/bin/python -m pytest -q archived_tests`.
- `_` — local datasets and reference material.
- `Thesis_doc/` — thesis sources.
- `Paper_doc/` — paper sources.
- `.claude/` and `.vscode/` — local agent and editor configuration.

The `.venv` entry is a shared environment symlink. From the repository root, invoke FL with `python -m FL_code.run_fl`; it defaults to the warm-pool `wz_cancer_warm` protocol, while `--protocol wz_cancer` explicitly selects cold quantizer training. Use `python -m FL_code.run_fl --help` to inspect the remaining options. Experiment modules use the same `python -m FL_code...` form. Run the sparse study with:

```bash
.venv/bin/python -m Experiments.Protocol_test.sparse_replay Experiments/Protocol_test/plans/tr_svhn_30c_f9.json
.venv/bin/python -m Experiments.Protocol_test.plot Experiments/Protocol_test/Runs_exp/tr_svhn_30c_f9/runs
```

The sparse study writes records and figures under its output root; broader reports live under `Experiments/Protocol_test/Archive_final_experiment_reports/`.

Use `get_into_node.sh` to enter the configured compute node.
