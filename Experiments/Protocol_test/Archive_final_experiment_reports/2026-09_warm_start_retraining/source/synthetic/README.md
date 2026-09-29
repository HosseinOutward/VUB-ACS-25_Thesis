# Warm-start retraining of the learned Wyner–Ziv quantizer

This experiment preserves the full synthetic and production-protocol setup in
`Experiments/brent_quantizer_check_paper/quantizer_check.py`. The declared deviations are the
requested shifted side information, warm-start variants, additional independent repeats, disk
caching, and failure/timing/RD analysis.

The additive SI shift is `+0.1`: one synthetic noise standard deviation and one tenth of the SI
standard deviation. The cached parent is evaluated on the same held-out target with both unshifted
and shifted SI before any conclusion is drawn.

Historical commands from the study's original location; they do not run from this archive.

```bash
.venv/bin/python -m Experiments.warm_start_retraining.experiment \
  Experiments/warm_start_retraining/config.json --validate-only
.venv/bin/python -m Experiments.warm_start_retraining.supervisor launch \
  Experiments/warm_start_retraining/config.json
```

The supervisor records health at 1, 5, and 30 minutes and every 90 minutes thereafter. Raw
per-repeat CSVs and parent checkpoints are under `results/full_run/cases`; combined CSVs, plots,
and the report are under `results/full_run/analysis`. Partial reports are explicitly marked
incomplete and must not be used for conclusions.
