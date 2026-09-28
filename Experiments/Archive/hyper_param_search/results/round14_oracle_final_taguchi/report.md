# Round-14 Oracle LWZQ hyperparameter search

## Result

The recommended default is:

| Parameter | Value |
|---|---:|
| `reconst_ld` | 72.31601255189986 |
| `tau` | 5.5 |
| `lr` | 0.0056728196862653865 |
| `train_epochs` | 140 |
| `train_sample_size` | 550,000 |
| `train_batch_size` | 6,250 |
| `lr_step_fraction` | 0.3075161942517416 |
| resolved `lr_step` | 43 |
| `hidden_dim` | 74 |
| `layers` | 6 |
| `source_perturbation` | `none` |

Its fresh-seed, five-client confirmation has a mean signed rate-equivalent
gap of **-0.07473 bits** relative to run 1. Negative is better. The
stage-specific gaps are **-0.02851**, **-0.04505**, and **-0.15064 bits**.
The gain therefore grows at the later, higher-quality stages rather than
coming from only one point on the RD curve.

## Experiment contract

- Round-14 raw client deltas were the targets.
- Raw rounds 9–13 from all five clients formed the decoder-visible history.
- Screening used client 0; finalists were re-run on clients 0–4 with fresh seeds.
- Normalization used the model's per-parameter standard-deviation slices from
  `StateDictManager.get_slices()`, matching the real protocol.
- The reference was the five-client round-14 curve from
  `run1_real_raw_o_nf4`.
- The response was the mean signed slope-adjusted rate-equivalent gap over
  all three stages.
- Codec structure was fixed at three bins per plane and three planes.
- Deduplication remained fixed at 26 training match bits. Match bits were not
  optimized as an RD parameter.
- Caches were disabled during screening, so each result was a fresh fit.

## Search

The corrected initial L27 treated only RD/training controls as factors. Its
best observed response was **-0.04074 bits**. Four viable, structurally
different rows initialized independent Optuna TPE studies.

Each study received 30 valid attempts. Ranges were not clipped to the
original three Taguchi values: a seed at a low or high endpoint caused the
range to extend in that direction. Across the studies, the outer limits
included:

| Parameter | Search extent |
|---|---:|
| `reconst_ld` | 30–2,700, log scale |
| `tau` | 0.1–10, log scale |
| `lr` | 0.0003–0.02, log scale |
| `train_epochs` | 20–200 |
| `train_sample_size` | 50,000–1,200,000 |
| `train_batch_size` | 12,500–200,000 |
| `lr_step_fraction` | 0.1–1.5 |
| `hidden_dim` | 26–250 |
| `layers` | 1–6 |

The basin results were:

| Basin | Complete | Pruned | Best gap (bits) |
|---|---:|---:|---:|
| low lambda, additive seed | 30 | 0 | **-0.08204** |
| middle lambda, draw/discard seed | 30 | 0 | -0.05498 |
| low lambda, no-perturbation seed | 29 | 1 | -0.05176 |
| high lambda, additive seed | 30 | 0 | -0.04938 |

The strongest basin consistently favored low reconstruction weight, high
temperature, 140–160 epochs, small batches, six layers, and usually
draw/discard perturbation. Because batch size and layer count still reached
search boundaries, the final L27 extended them to a 6,250 batch and seven
layers instead of treating the Optuna bounds as optima.

## Final Taguchi refinement

The final L27 embedded the exact Optuna winner as its first row and tested
local inward/outward levels. All 27 configurations completed. The strongest
client-0 rows were:

| Row | Gap (bits) | Main distinction |
|---|---:|---|
| 14 | **-0.08806** | 200 epochs, 6,250 batch, 7 layers |
| 11 | -0.08781 | 140 epochs, 12,500 batch, 96 hidden units |
| 13 | -0.08405 | 200 epochs, 25,000 batch |
| 24 | -0.08298 | 160 epochs, 6,250 batch, 7 layers |
| 2 | -0.07953 | 160 epochs, 25,000 batch |

The main-effect recommendation was not assumed to work. It was evaluated
separately because the earlier screening showed that independently favorable
levels can interact poorly.

## Parameter effectiveness and remaining bounds

Effectiveness is the range between the best and worst level-mean response for
each factor in the final orthogonal L27. This estimates how much changing that
factor moved the objective while balancing the other factors. Relative
effectiveness normalizes the largest observed effect to 100%. These are
Taguchi main effects, not proof that a parameter acts independently; the
five-client confirmation remains the test of a complete configuration.

| Rank | Parameter | Selected value | Effect range (bits) | Relative effectiveness | Search status | Suggested future search |
|---:|---|---:|---:|---:|---|---|
| 1 | `train_sample_size` | 550,000 | **0.01743** | **100%** | Interior local optimum between 350k and 700k | Narrow around 500k–650k |
| 2 | `lr` | 0.0056728 | **0.01332** | **76%** | Interior local optimum between 0.003 and 0.009 | Narrow around 0.0045–0.0075 |
| 3 | `lr_step_fraction` | 0.307516 | **0.01253** | **72%** | Interior local optimum between 0.25 and 0.4 | Narrow around 0.28–0.34 |
| 4 | `tau` | 5.5 | 0.00917 | 53% | **Lower final-L27 bound** | Test below 5.5 in the winning basin |
| 5 | `source_perturbation` | `none` | 0.00898 | 52% | Categorical winner; all three choices tested | Keep `none` unless other factors change materially |
| 6 | `reconst_ld` | 72.316 | 0.00726 | 42% | **Lower final-L27 bound** | Test lower than 72 while preserving the winning regime |
| 7 | `train_batch_size` | 6,250 | 0.00651 | 37% | **Lower final-L27 bound** | Test smaller batches; account for added runtime |
| 8 | `train_epochs` | 140 | 0.00566 | 33% | **Lower final-L27 bound** | Test 100–140 epochs; this may also recover speed |
| 9 | `layers` | 6 | 0.00538 | 31% | Interior local optimum between 5 and 7 | Treat six as resolved locally |
| 10 | `hidden_dim` | 74 | 0.00361 | 21% | Weak interior optimum between 62 and 96 | Low priority; 62 was nearly tied |

The underlying level means, ordered numerically where applicable, were:

| Parameter | Mean response by tested level (bits; lower is better) |
|---|---|
| `train_sample_size` | 350k: -0.05668; **550k: -0.07410**; 700k: -0.07151 |
| `lr` | 0.003: -0.05892; **0.0056728: -0.07224**; 0.009: -0.07114 |
| `lr_step_fraction` | 0.25: -0.06351; **0.307516: -0.07566**; 0.4: -0.06313 |
| `tau` | **5.5: -0.07106**; 8.1077: -0.06935; 11: -0.06189 |
| `source_perturbation` | draw/discard: -0.06601; **none: -0.07263**; add: -0.06365 |
| `reconst_ld` | **72.316: -0.07221**; 109.066: -0.06495; 150: -0.06514 |
| `train_batch_size` | **6,250: -0.07061**; 12,500: -0.06759; 25,000: -0.06410 |
| `train_epochs` | **140: -0.06960**; 160: -0.06876; 200: -0.06394 |
| `layers` | 5: -0.06645; **6: -0.07062**; 7: -0.06523 |
| `hidden_dim` | 62: -0.06844; **74: -0.06873**; 96: -0.06512 |

The four boundary selections are not established optima. They say only that,
within the final local design, lower temperature, reconstruction weight,
batch size, or epoch count was better. The next search should extend those
directions while keeping the three strongest interior factors close to their
selected values. In particular, lowering epochs and possibly reconstruction
weight is the most promising route to recover training speed without
discarding the RD gain.

## Five-client confirmation

| Candidate | Five-client gap (bits) | Runtime | Interpretation |
|---|---:|---:|---|
| Main-effect recommendation | **-0.07473** | 4,271 s | Best robust result |
| Final L27 row 13 | -0.06748 | 2,892 s | Faster alternative |
| Final L27 row 11 | -0.06611 | 3,884 s | Good, but dominated by the winner |
| Final L27 row 14 | -0.05615 | 5,664 s | Client-0 leader did not generalize best |

Client-0 screening overstated the absolute gains of some rows, but every
confirmed finalist remained better than run 1. The recommendation generalized
best and is the appropriate default. Row 13 is a possible speed-oriented
alternative: it was about 32% faster in this confirmation while giving up
about 0.00725 rate-equivalent bits.

## Interpretation

The most important result is not the client-0 minimum of -0.08806 bits, but
the five-client result of -0.07473 bits. It combines a meaningful RD
improvement with less sensitivity to the selected client and seed.

The selected 5.5 temperature and 72.32 reconstruction weight indicate that
the previous defaults over-weighted reconstruction during training. The
smaller batch and longer training schedule appear to give the optimizer
enough updates to learn the later-stage refinements. Six layers were more
reliable than the seven-layer boundary extension after averaging clients.
The lack of source perturbation in the confirmed winner also shows that the
draw/discard preference inside one Optuna basin was not robust enough to make
the final default.

The next test is the real round-14-to-18 protocol replay: train the O or T
quantizer on round 14, reuse it through four F rounds, refresh inference side
information, and compare the resulting run 3/run 4 curves against runs 1/2.
