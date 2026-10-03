# Final controlled O/T diagnostic analysis

Completed 2026-10-01 at 12:18 Europe/Brussels. Supervisor and plotter exited successfully.
All 27 candidates and 1,458 stage rows are present. Every evaluated rANS payload passed its
decoded-symbol equality assertion. No production code or quantizer defaults were changed.

## Conclusion

The controlled sampled test demonstrates severe **generalization failure**, not an intrinsic
advantage of temporal SI over oracle SI. There are two separable problems: the jointly trained
O encoder/decoder/prior generalizes poorly, and fixed-symbol full-SI prior fitting can make
future rates substantially worse even when the encoder and reconstruction are unchanged.
Initialization matters, but three random restarts did not solve the sampled O problem.

This is a mechanism diagnosis, not an exact reproduction of the original full-vector warm-pool
failures. It is not sufficient evidence to claim the original failure has been fixed, or to
draw an oracle-vs-temporal research conclusion from the stopped runs.

## Scope and limits

- Clients 0 (healthy control in the original run), 5, and 11; seeds 43, 44, 45.
- Joint training on rounds 6 and 5 with each target's correct capture-time SI.
- Common compressed **raw** history: T's five columns are an exact subset of O's 150.
  Unlike the original protocol replay, future history does not evolve through these models'
  decoded reconstructions, and no five-parent warm pool is reproduced.
- Full-vector normalization precedes sampling: 16,384 training coordinates and 4,096
  held-out coordinates per round. Future rounds 7, 8, 12, 13 are evaluation only; they include
  both coordinate subsets. All seeds are averaged equally below, not best-of-three winners.
- Cold models use 140 epochs; O-from-T uses the default 35-epoch warm schedule. However, the
  sampled joint fits have 6 batches/epoch versus 88 at the full 550,000-row sample cap. Their
  optimizer-update budget and number of distinct coordinates are much smaller than production.
- Current-only prior rows were repeated so current and two-round prior fits have equal
  sampling/update budgets. Historical symbols are produced by the **final encoder**.
- Native prior fits start from the joint model; alternate-SI priors are independent scratch fits.
  Thus full-SI T versus O-from-T comparisons include different prior initializations/schedules.
- Sampled rANS uses the requested 2,048 lanes and includes the payload's fixed header/states
  (3.211328 bits/float at 20,480 positions). These short-block payload rates cannot be
  extrapolated to full-vector protocol savings, nor corrected simply by subtracting the flush.

## 1. Hard-inference audit: overfitting is directly visible

These are the stage-averaged normalized-MSE-plus-native-NLL objective, in the training loss's
natural-log units. They are not the WMAPE equal-distortion rate gap. Scores were recomputed
from every saved joint model on the actual sampled training coordinates, and compared with
historical held-out coordinates. Each row averages nine client/seed models.

| Model | Stochastic training loss | Hard training RD | Hard held-out RD |
|---|---:|---:|---:|
| T | 29.56244 | 27.61497 | 62.63904 |
| O | 3.17849 | 2.11090 | 110.73738 |
| O initialized from T | 17.58863 | 17.24693 | 68.97102 |

O performs extremely well even under hard inference **on its training coordinates** and very
poorly on held-out coordinates. Therefore a soft-training/hard-inference mismatch alone cannot
explain this result. All three O seeds at every client have worse historical held-out RD than
all three corresponding T seeds. More retries against the same training-loss selector are
not a demonstrated remedy for this systematic generalization problem.

The exact T-in-O embedding was verified before warm training: symbols, reconstruction, and
priors match to numerical tolerance with extra SI weights zeroed. The model family can
represent T. Subsequent optimization can lose that baseline; it should not be discarded.

## 2. Full future-round RD table

Rates are cumulative bits/float at stages 1/2/3. Distortion averages raw WMAPE percentages
first, then converts to dB. Current = fixed-symbol prior fit on round 6; multi = fit on rounds
6 and 5. The encoder/decoder and all distortion values are identical across these two fits.

| Model | Prior fit | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| T | Current | 1 | 0.4448 | 0.3280 | 2.3106 | 3.2114 | 4.5536 | -2.107 (78.462%) |
| T | Current | 2 | 0.6407 | 0.5418 | 3.7579 | 3.2253 | 5.4880 | -4.297 (60.974%) |
| T | Current | 3 | 0.8744 | 0.8180 | 5.6723 | 3.3327 | 6.7663 | -7.234 (43.481%) |
| T | Multi | 1 | 0.4448 | 0.3235 | 2.1825 | 3.2113 | 4.4701 | -2.107 (78.462%) |
| T | Multi | 2 | 0.6407 | 0.5217 | 3.5831 | 3.2191 | 5.3769 | -4.297 (60.974%) |
| T | Multi | 3 | 0.8744 | 0.7601 | 5.4810 | 3.2816 | 6.6414 | -7.234 (43.481%) |
| O | Current | 1 | 2.2237 | 0.6709 | 3.2882 | 3.4232 | 5.1965 | -1.842 (80.895%) |
| O | Current | 2 | 3.6061 | 1.2292 | 5.6183 | 3.6933 | 6.8183 | -4.009 (63.029%) |
| O | Current | 3 | 4.8198 | 1.9337 | 8.0415 | 4.3058 | 8.5606 | -5.290 (54.388%) |
| O | Multi | 1 | 2.2237 | 0.6308 | 3.0091 | 3.4135 | 5.0179 | -1.842 (80.895%) |
| O | Multi | 2 | 3.6061 | 0.9868 | 5.0438 | 3.4816 | 6.4739 | -4.009 (63.029%) |
| O | Multi | 3 | 4.8198 | 1.3084 | 7.0711 | 3.7055 | 7.9931 | -5.290 (54.388%) |
| O-from-T | Current | 1 | 0.4688 | 0.3581 | 1.0900 | 3.2140 | 3.7002 | -2.248 (77.194%) |
| O-from-T | Current | 2 | 0.8317 | 0.7800 | 2.4816 | 3.3703 | 4.6755 | -4.927 (56.706%) |
| O-from-T | Current | 3 | 1.1827 | 1.4434 | 4.0723 | 3.8598 | 5.8068 | -7.400 (42.658%) |
| O-from-T | Multi | 1 | 0.4688 | 0.3289 | 0.6164 | 3.2116 | 3.3578 | -2.248 (77.194%) |
| O-from-T | Multi | 2 | 0.8317 | 0.5766 | 1.5648 | 3.2257 | 4.0149 | -4.927 (56.706%) |
| O-from-T | Multi | 3 | 1.1827 | 0.8651 | 2.6359 | 3.3534 | 4.8526 | -7.400 (42.658%) |

Preserving the joint prior yields final native prior/rANS rates of 0.8744/3.3265 for T,
4.8198/6.5382 for O, and 1.1827/3.6513 for O-from-T, at the same distortions shown above.
All preserved-prior stages and historical-validation stages are in report.md and stage_points.csv.

## 3. Equal-distortion gaps

Piecewise-linear shared-distortion-range metric from rd_metrics.py. Positive means worse.
Each candidate is compared with T using the same current/multi prior policy. The primary
transport comparison is O's full-SI rANS versus T's **temporal** rANS, not T's full-SI check.

| Candidate / fit | Native rANS gap bits (%) | Full-SI rANS check gap bits (%) | Native prior check gap bits (%) |
|---|---:|---:|---:|
| O / current | +3.51726 (+109.02%) | +1.50952 (+28.84%) | +5.01189 (+1038.10%) |
| O / multi | +3.17799 (+98.70%) | +1.26612 (+24.67%) | +4.45191 (+955.11%) |
| O-from-T / current | +1.40335 (+43.12%) | -1.02735 (-18.07%) | +1.86892 (+321.09%) |
| O-from-T / multi | +0.78314 (+24.20%) | -1.55593 (-27.91%) | +0.98726 (+178.25%) |

O-from-T improves against T's independently fitted **full-SI** prior/coder but still loses to
T's native temporal coder. Its slightly better final distortion does not make its RD curve
better. Preserved-prior O-from-T remains +0.14522 bits (+4.46%) worse in native sample rANS,
and +0.13810 bits (+20.55%) worse in native prior rate at equal distortion.

## 4. Prior fitting is an independent failure mechanism

For unchanged T symbols and reconstruction, current-only temporal prior rate is 0.8180,
whereas its independent full-SI prior rate is 5.6723. Quantizer reconstruction cannot explain
that rate difference: only the fitted prior changes. Full-SI fitting is failing to generalize.

At equal distortion, multi-round versus current-only native prior gaps are:

| Model | Prior gap bits (%) | Sample rANS gap bits (%) |
|---|---:|---:|
| T | -0.02754 (-4.79%) | -0.01779 (-0.55%) |
| O | -0.55517 (-10.40%) | -0.33370 (-5.03%) |
| O-from-T | -0.92623 (-37.03%) | -0.64833 (-13.81%) |

However, multi-round O/O-from-T priors remain worse than their preserved joint priors:
+1.38376 bits (+40.73%) and +0.75362 bits (+91.73%), respectively, at equal distortion.
Thus multi-round prior fitting helps, but blindly replacing the joint prior is still unsafe.

## 5. Selection and stage failures

Historical validation changed four of nine best-seed selections. For randomly initialized O,
validation-selected future native-prior RD was actually worse than training-loss selection:
+9.37% for current-only and +13.45% for multi-round fits. O-from-T improved by -19.95% and
-12.36%, respectively. Coordinate-held-out validation is therefore not a universal fix for
cross-round generalization; chronological round-held-out validation needs testing.

No evaluated prior stage had mean rate below 0.01 bits. In future evaluation, all three clients
(0, 5, 11) had >1.05-bit full-SI stages in O's current/multi fits, O-from-T's current/multi
fits, and T's independent full-SI fits. T's native temporal prior had no such stage in any
policy. O-from-T's preserved joint prior exceeded the threshold only at client 5.
These thresholds identify bad stage code lengths, not proof that a quantizer emits one symbol.
Individual rows, seeds, and stage rates are retained in suspect_stages.csv.

## 6. What the original full-vector experiment can support

Original runs remain incomplete/stopped: OMR 300/900 rows, TMR 225/900. Compare only the
seven complete matched rounds 6–12, 210 observations per codec, with no client exclusions.

| Phase | Native transported rANS gap bits (%) | Full-SI rANS check gap bits (%) |
|---|---:|---:|
| Trained | -0.04213 (-7.60%) | -0.01927 (-3.63%) |
| Frozen | +0.04384 (+7.58%) | +0.05147 (+9.01%) |
| Overall | +0.03154 (+5.48%) | +0.04135 (+7.31%) |

Frozen blown stages occur in OMR clients 5/11 and TMR client 1. The original control client 0
was healthy, while the sampled O test also fails there. That is further reason not to equate
this smaller-data diagnostic with reproduction of the original failures. Full original tables
are in ../baseline/report.md.

## Recommended next step, not an implemented fix

Run full-vector controlled confirmation with the production sample/update budget and paired
decoder-visible histories before restarting the full protocol sweep or changing defaults.
Test keeping the exact embedded-T candidate and the preserved joint prior as explicit fallback
candidates, alongside warm O training and multi-round fixed-symbol prior fitting. Select with
available chronological validation and verify that selection does not discard the stable
fallback; never select using future frozen-round results. More random retries alone, or the
current coordinate-validation selector alone, are not supported as sufficient remedies here.

No further experiments were launched by this final analysis, and nothing was committed.
