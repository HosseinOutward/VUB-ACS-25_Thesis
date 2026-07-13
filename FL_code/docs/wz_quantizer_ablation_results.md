# Gaussian Wyner–Ziv quantizer ablation results

## Decision summary

The primary metric is distance to the Gaussian Wyner–Ziv bound,

```text
gap_db = 10 log10(MSE / (0.01 * 2^(-2R))).
```

The strongest robust improvement is **doubling the distortion multiplier while retaining
shared encoder/decoder heads**. A final factorial control showed that the earlier apparent
per-stage-head improvement was actually a lambda effect: at the same doubled lambda, shared
heads are equal or better in every tested case and use fewer parameters. The tiny source
perturbation helps both tested 2x3 seeds but hurts 4x2, so it is architecture-dependent. Hard
transmitted context is best for retraining the conditional prior; annealed Gumbel
probabilities have little effect once context is hard.

These are Gaussian paper-test conclusions, not recommendations for FL production defaults.
The production distribution and decoder-available side information require separate tests.

## Controlled setup

Every full run used independent 2,000,000-sample training and inference tensors,
`Y ~ N(0,1)`, `X = Y + N`, `Var(N)=0.01`, 180 epochs, 200,000 samples/epoch,
batch size 1,000, StepLR(40, 0.3), tau 1.0 to 0.1, FP32, and one training attempt.
The 2x3 model used lambda 60 and the 4x2 model lambda 80 unless explicitly noted.

For seed 0, the measured source scale was 0.98583984375 and the normalized distortion
denominator was 0.5195344. For seed 42 they were 0.9873046875 and 0.5181176.
GPU-sensitive comparisons use controls run on the same physical GPU.

## Quantizer results

### Optimizer and loss scale

| Choice | Final rate | MSE | Gap | Delta vs same-device control | Classification |
|---|---:|---:|---:|---:|---|
| Adam + summed loss | 1.5330 | 0.0022270 | 2.707 dB | — | baseline |
| AdamW, weight decay 1e-4 | 1.4245 | 0.0027487 | 2.967 dB | +0.124 dB | neutral/slightly harmful |
| Adam + loss/3 | 1.5305 | 0.0030176 | 4.011 dB | +1.304 dB | harmful |
| Adam + loss/3 + 3x LR | 1.3638 | 0.0027477 | 2.600 dB | -0.107 dB | neutral/suggestive |
| Adam + loss/3 + epsilon/3 | 1.5265 | 0.0022250 | 2.664 dB | -0.043 dB | neutral/equivalent |
| Adam + summed loss + 3x LR | 1.3699 | 0.0030222 | 3.051 dB | +0.344 dB | harmful |

One-step checks explain the loss-scale behavior. For 2x3, ordinary averaging changed the
Adam update by 2.18%, 3x LR changed it by 199%, and epsilon/3 restored it within 0.0011%.
For 4x2 the corresponding differences were 1.48%, 99.6%, and exactly zero at recorded
precision. Dividing the complete objective does not change its rate/distortion balance or
tau relaxation; it only changes optimizer scaling. The full run nevertheless matters because
small update differences amplify through Gumbel sampling.
The moving-forward implementation uses `loss/K` with Adam `epsilon/K`; this is the
update-equivalent form of the tested summed quantizer loss. The prior applies another `/K`
after its existing stage mean for the same reason.

### Source perturbation

The random perturbation used a dedicated generator, so it did not shift data-sampling or
Gumbel RNG. Draw-and-discard is the paired no-perturbation control.
The explicit no-draw and draw-discard runs were identical in every RD metric (2.843024 dB);
draw-discard only increased quantizer runtime from 588 s to 645 s (+9.6%).

| Architecture / seed | Control gap | Add perturbation gap | Delta | Classification |
|---|---:|---:|---:|---|
| 2x3 / 0 | 2.843 dB | 2.616 dB | -0.227 dB | beneficial |
| 2x3 / 42 | 3.089 dB | 2.747 dB | -0.342 dB | beneficial |
| 4x2 / 0 | 3.325 dB | 3.680 dB | +0.355 dB | harmful |

The perturbation is beneficial for 2x3 across the tested seeds but reverses on 4x2. Keep it
explicit and architecture-specific; it is not a safe global change.

### Shared versus per-stage heads

| Architecture / seed | Heads | Lambda | Rate | MSE | Gap | Delta vs original shared control |
|---|---|---:|---:|---:|---:|---:|
| 2x3 / 0 | shared | 60 | 1.5330 | 0.0022270 | 2.707 dB | — |
| 2x3 / 0 | per-stage | 60 | 1.6647 | 0.0046809 | 6.726 dB | +4.018 dB |
| 2x3 / 0 | shared | 120 | 1.0128 | 0.0041796 | 2.309 dB | -0.398 dB |
| 2x3 / 0 | per-stage | 120 | 1.1309 | 0.0035759 | 2.343 dB | -0.364 dB |
| 2x3 / 42 | shared | 60 | 1.5647 | 0.0023274 | 3.089 dB | — |
| 2x3 / 42 | shared | 120 | 1.0129 | 0.0040573 | 2.181 dB | -0.909 dB |
| 2x3 / 42 | per-stage | 120 | 1.0437 | 0.0040445 | 2.352 dB | -0.737 dB |
| 4x2 / 0 | shared | 80 | 2.0970 | 0.0011747 | 3.325 dB | — |
| 4x2 / 0 | shared | 160 | 1.1518 | 0.0034821 | 2.353 dB | -0.972 dB |
| 4x2 / 0 | per-stage | 160 | 1.4508 | 0.0024945 | 2.704 dB | -0.620 dB |

Per-stage heads add 606 parameters to 2x3 (154,015 versus 153,409). At the original lambda
they fail badly. Doubling lambda rescues them, but the missing factorial control shows that
shared heads at the same lambda are better by 0.034 dB (2x3 seed 0), 0.171 dB (2x3 seed 42),
and 0.352 dB (4x2 seed 0). The robust improvement is therefore caused by the higher lambda,
not separate heads. For the Gaussian setup, retain shared heads and use lambda 120 for 2x3
and lambda 160 for 4x2 as the current tested best choices.
This establishes the direction but not the globally optimal lambda. A future RD sweep should
vary lambda around these points while keeping shared heads fixed; separate heads no longer
need further testing for this Gaussian experiment.

### Kernel options (80-epoch paired screen)

| Choice | Gap | Quantizer time | Delta gap | Runtime change | Classification |
|---|---:|---:|---:|---:|---:|
| FP32, non-fused Adam | 3.288 dB | 301.6 s | — | — | control |
| Fused Adam | 3.117 dB | 255.4 s | -0.172 dB | -15.3% | speed-beneficial, RD inconclusive |
| TF32 | 3.344 dB | 307.8 s | +0.055 dB | +2.0% | neutral/unhelpful |

Fused Adam materially improves runtime on the RTX 3090 without evidence of worse RD. TF32
does not help these small RNN layers. Kernel-induced numerical differences can change a
single stochastic trajectory, so neither short-run gap difference is treated as a fundamental
RD change.

Moving-forward Gaussian runs use fused Adam for its measured runtime benefit. Source
perturbation remains disabled by default and should be tested on the final FL data rather than
selected from its architecture-dependent Gaussian behavior. Training uses the
update-equivalent `loss/K` and Adam `epsilon/K` form.

## Separately retrained conditional prior

All variants trained on the same frozen 80% of emitted symbols and side information, started
from the same seed, and were evaluated on the real hard-context/categorical coding path.

| Training context | Training probabilities | Full rate | Held-out rate | Delta held-out |
|---|---|---:|---:|---:|
| hard transmitted one-hot | categorical | 1.5217 | 1.5224 | — |
| hard transmitted one-hot | annealed Gumbel | 1.5293 | 1.5304 | +0.0079 |
| soft encoder codes | categorical | 1.5554 | 1.5553 | +0.0329 |
| soft encoder codes | annealed Gumbel | 1.5599 | 1.5598 | +0.0374 |

Soft previous-plane context is the main cause of the worse rate. With legitimate hard
decoder context, annealed Gumbel training is nearly neutral, although categorical remains
best and matches the coding path. Fixed-tau Gumbel diagnostic rows are preserved in the
result directory but are invalid for conclusion.

This prior comparison concerns conditional entropy coding only. It does not change quantizer
distortion or justify modifying the quantizer merely to improve the separately reported rate.

## Artifacts

`FL_code/experiments/results/wz_quantizer_ablation/` contains one JSON configuration, CSV,
and log per quantizer run; per-stage cumulative training-prior, inference-prior,
retrained-prior, and marginal rates; source scale, distortion denominator, parameter count,
GPU, and timings. It also contains `quantizer_results.csv`, `prior_results.csv`, the frozen
baseline emissions, one-step checks, and `rd_ablation.png`.
