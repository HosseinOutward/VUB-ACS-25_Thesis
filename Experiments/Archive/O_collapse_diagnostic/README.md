# O-round quantizer collapse diagnostic

Every ResNet-18/SVHN O and OR sparse run collapses: 7-11 of 30 trained-round quantizers emit a
constant symbol (WMAPE ~100%), and about two thirds have at least one stage that adds no
distortion reduction. T/TR on the same trace almost never do. The hypothesis is decoder-side
posterior collapse: O's decoder and prior see 150 SI vectors, including earlier clients' same-round
updates, and learn to rely on them early in training while the encoder's stages die.

`run_diagnostic.py` regenerates the seed-43 SVHN trace up to round 5 and trains single cold
attempts (`quantizer_train_repeats=1`, so collapse is observed instead of selected away) with the
protocol's default quantizer configuration, 2 bins x 3 planes, for clients 1, 10, and 29 and seeds
0-2, under four SI sets:

| SI set | Vectors | Content |
|---|---|---|
| `own5` | 5 | the client's own last five rounds (T's view) |
| `own5_same_round` | 5 + client id | `own5` plus earlier clients' same-round updates |
| `server_prev` | 150 | every client's rounds 0-4, no same-round updates |
| `server_actual` | 150 | O's real view at the client's turn |

Earlier clients' same-round updates are raw here, where the protocol would hold their decoded
reconstructions. Each attempt writes `results/<client>_<si>_<seed>.json` (full-vector stage WMAPE,
stage rates, distinct symbols per stage) and `results/curves/*.csv` (loss, hard stage MSE, and
cumulative stage rate on a 10k-row training sample every five epochs).

```bash
systemd-run --user --collect --unit=fl-wz-ocollapse-w0 --service-type=exec --working-directory=$PWD \
  .venv/bin/python -u -m Experiments.O_collapse_diagnostic.run_diagnostic --gpu 0 --worker 0
```

Run one worker per GPU (`--worker 0/1`); completed attempts are skipped on rerun. `--overrides`
takes JSON `WZcfgQuant` overrides on top of the protocol defaults:

| Folder | Overrides |
|---|---|
| `results/` | none, under the former defaults (6 layers, lr 5.67e-3, 120 epochs) |
| `results_layers4/` | `{"layers": 4}` over the former defaults |
| `results_layers4_halflr_140ep/` | `{"layers": 4, "lr": 2.836e-3, "train_epochs": 140}`, now the defaults |
| `rd_orig/`, `rd_l4_halflr_140ep/` | the first and last rows, rerun with `fine_tuned_stage_rates` and `final_training_loss` |

Attempts reproduce bit-exactly, so the `rd_*` reruns are the same models with the codec's
fine-tuned prior rate added. `compare_rd.py <candidate> <reference>` ranks each (client, SI set)'s
seeds by final training loss, the protocol's best-of-attempts rule, and compares the selected
candidates' three-stage RD curves with `rd_curve_gap`:

```bash
.venv/bin/python -m Experiments.O_collapse_diagnostic.compare_rd \
  Experiments/O_collapse_diagnostic/rd_l4_halflr_140ep Experiments/O_collapse_diagnostic/rd_orig
```

The 4-layer, half-lr, 140-epoch configuration won that comparison and became the `WZcfgQuant`
default on 2026-09-29; reproducing `results/` or `rd_orig/` now needs
`--overrides '{"layers": 6, "lr": 0.0056728196862653865, "train_epochs": 120}'`.

## Warm training under the new defaults

`warm_check.py --si-set <set>` builds the warm protocol's pool from clients 0-4 (cold, best of the
default three attempts), then warm-trains clients 10 and 29 from two parents each (the rotating
window: slots 0-1, then 2-3), for seeds 0-2. A warm fit derives 35 epochs at lr 8.5e-4 and tau 0.5
from the defaults (`WZcfgQuant.warm_retraining_config`). Results land in `warm_results/`; the cold
baseline for the same clients is `rd_l4_halflr_140ep/`.

## Results (2026-09-29, seed-43 SVHN/ResNet-18 trace, trained round 5)

Single cold attempts, 36 per configuration (3 clients x 4 SI sets x 3 seeds):

| Configuration | Full collapse | >=1 dead stage | Joint prior > 2 bit/stage |
|---|---|---|---|
| former defaults (6 layers, lr 5.67e-3, 120 epochs) | 10 | 16 | 5 |
| 4 layers | 1 | 7 | 0 |
| 4 layers, lr 2.836e-3, 140 epochs | 0 | 0 | 0 |

Collapse is a sudden whole-network death (loss jumps to the mean predictor's ~143 and never
recovers), 14 of 15 before the first lr drop; T's own-history SI collapses too, and 150-vector SI
more often. Selected candidates (lowest training loss of three, fine-tuned prior rate, `compare_rd.py`):
the new configuration is usable in 12/12 cases against 8/12, and where both are usable it wins 6,
ties 1, and loses 1 (+4.6% rate); client 29's O view improves from 1.84 to 0.59 bit.

Warm fits under the new defaults (`warm_check.py`, clients 10 and 29, seeds 0-2) are all usable and
average +0.9% (T view) and -0.4% (O view) rate at equal distortion against the cold selected
candidate, at ~3.5-4 min per client against ~10 min for a cold best-of-three.

Open lead, untested: joint training samples the prior through Gumbel-softmax (`get_priors`),
ignoring `prior_probabilities`, which rewards unbounded prior logits.
