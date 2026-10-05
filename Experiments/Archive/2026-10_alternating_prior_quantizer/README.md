# Does a poorly fitted joint prior constrain the quantizer?

**The quantizer seems not to be constrained by the poorly fitted jointly trained prior in
this experiment.** The hypothesis was that the prior's remaining room to fit the emitted
symbols signaled ineffective joint training and might steer the quantizer toward a worse RD
solution. Separately fitting the prior, freezing it, and continuing quantizer training did
not improve RD after another prior fit. This weakens that hypothesis in the tested setting,
even though prior-only fitting clearly improves the rate estimate.

Completed 2026-10-05: seed 43, SVHN/ResNet-18, five clients, target round ID 7, and fixed full
decoder SI from all five clients in rounds 5 and 6 (ten compressed-history vectors).
Round IDs are zero-based. Raw past updates were explicitly available as decoder seed history;
no target or future update entered SI. The quantizer used three binary stages.

The sequence was initial joint training → prior only → quantizer with frozen prior → prior
only → joint fine-tuning → final prior only. Initial training used 140 epochs and the best of
three accepted candidates; each prior fit used 30 epochs and the best of three accepted
candidates. Quantizer/joint continuations each used one 35-epoch warm fit. The existing RD
objective, Gumbel behavior, and detached earlier-symbol prior context were preserved.

The first prior fit saved **38.78% at equal distortion** versus the jointly trained prior.
Quantizer-only continuation then worsened RD by **1.80%**; another prior fit recovered
approximately the original fitted-prior curve. The final joint-plus-prior continuation
remained **2.10% worse** than the first prior fit. These are one-seed, one-target training
diagnostics with fixed SI, not evidence that joint training is generally optimal.

## Retained files

- [Report](report.md): complete phase matrix and all stage RD points; [RD plot](rd_phases.png).
- [Measurements](measurements.csv): all 90 client/phase/stage observations, including symbol
  counts and failure flags. No clients triggered exclusions; all comparisons retain 5/5.
- [Configuration](config.json): the original effective settings and historical run identity.
- `code/`: experiment training, plan, and analysis code plus the experiment and FL configs.
  Shared training libraries are reused from the repository. Only file-location paths were
  adjusted in the training code/configs; the metadata records the original run locations.

Checkpoints, logs, copied shared modules, tests, completion markers, and duplicate per-phase
records were removed. Conditional-prior rates were measured; entropy coding was skipped.
The retained CSV is sufficient to recompute both directional percentage matrices. Rates and
raw WMAPE are averaged over matched clients before dB conversion and RD interpolation.

Regenerate comparisons and the plot without training:

```bash
.venv/bin/python -c 'import runpy; runpy.run_path("Experiments/Archive/2026-10_alternating_prior_quantizer/code/plot.py", run_name="__main__")' \
  Experiments/Archive/2026-10_alternating_prior_quantizer --out /tmp/alternating-prior-report
```
