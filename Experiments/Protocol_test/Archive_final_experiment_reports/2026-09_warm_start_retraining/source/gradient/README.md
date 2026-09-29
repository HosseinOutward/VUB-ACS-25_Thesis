# Reduced-epoch warm start on federated gradients

This study uses one deterministic seed-42069 SVHN/ResNet-18 identity-FL trace with five clients.
It trains the production Oracle-rANS codec for client 0 at round 12 with raw rounds 10–11 as its
ten decoder-visible SI vectors. Rounds 13–21 advance only in the raw FL trace. At round 22, raw
rounds 20–21 supply the ten refreshed SI vectors.

Round 22 compares the frozen parent, 30-epoch scratch, 30-epoch naïve warm start, and a 120-epoch
scratch reference. Twelve paired stochastic repeats use training seeds `42069 + repeat`. Rate is
cumulative realized rANS rate; distortion is hard-symbol stage WMAPE in dB.

The production round-12 parent retains its default two-candidate selection. Every round-22 fit uses
one candidate so scratch and warm conditions differ only in initialization, not in access to an
extra cold-start candidate.

Historical commands from the study's original location; they do not run from this archive.

```bash
.venv/bin/python -m Experiments.warm_start_gradient_retraining.experiment \
  Experiments/warm_start_gradient_retraining/config.json --validate-only
.venv/bin/python -m Experiments.warm_start_gradient_retraining.supervisor launch \
  Experiments/warm_start_gradient_retraining/config.json
```
