# Warm-start retraining studies

This archive closes the two warm-start development studies that lived under
`Experiments/warm_start_retraining/` and
`Experiments/warm_start_gradient_retraining/`. They were implementation studies
for the warm-start mechanism, not production protocol reports. The active study
directories were removed after their useful evidence was copied here.

## Contents

- `source/synthetic/` — synthetic shifted-side-information study, including the
  runner and analysis source, configuration, and verification notes.
- `source/gradient/` — SVHN/ResNet-18 reduced-epoch gradient study, including
  the base and warm-policy-grid configurations and runner source.
- `evidence/synthetic/` — final 24-case report, plots, summaries, manifests, and
  compact per-repeat CSV records with completion markers.
- `evidence/gradient/v3/` — final 30-epoch warm-versus-scratch study.
- `evidence/gradient/v4/` — final LR/initial-temperature policy grid.

The analysis source is retained for historical interpretation. Its mean-curve construction now
uses the central `Experiments/Protocol_test/rd_metrics.py` conversion so linear WMAPE is averaged
before converting to dB.

## Gradient-study parent setting

The production round-12 parent was trained with
`quantizer_train_repeats=2`. Every round-22 scratch or warm candidate used one
candidate. This distinction is part of the historical experiment design and is
recorded here even though the generic configuration files do not expose it as a
top-level field. The retained `parent_raw_stage_results.csv` files are the compact
parent evidence; the large gradient trace, model checkpoint, and cache artifacts
were intentionally excluded.

## Evidence policy

Retained files are sufficient to inspect the reported summaries and recompute
compact analyses without rerunning the expensive jobs. Removed artifacts include
traces, checkpoints, transparent caches, supervisor/control logs, health files,
plot refresh logs, bytecode, and abandoned/incomplete run versions (`full_run`,
`full_run_v2`). The original manifests' provenance fields may mention those
generated files because they describe the dirty working tree at run time; that is
historical metadata, not a claim that those files are part of this archive.

The synthetic study's historical configuration used one quantizer candidate for
its parent; the gradient study's round-12 production parent used two as noted
above. No training was rerun while making this archive.
