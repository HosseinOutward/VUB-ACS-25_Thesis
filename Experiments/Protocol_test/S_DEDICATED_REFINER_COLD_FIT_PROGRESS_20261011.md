# Dedicated refiner cold-fit progress — 11 October 2026

## Question and status

The corrected warm pilot showed that extra LWZQ stages were not all dead, but their later symbols had almost no conditional code cost and the acquired-sample WMAPE response was small and nonmonotone. This follow-up changes only the acquisition-refiner optimizer schedule. The deterministic frozen-prefix correction is a separate correctness fix; the cold schedule is an optimization experiment, not evidence that the fix alone improves rate-distortion.

The seed-43 cold pilot ran under `fl-wz-s-dedicated-refiner-coldfit-pilot-20261011` from [the cold pilot plan](plans/s_dedicated_refiner_coldfit_pilot_20261011.json), outputting to `_/experiments/s_dedicated_refiner_coldfit_pilot_20261011`. It compared O and S with 0, 1, and 3 extra planes using 200,000 sample positions. All four cases completed with exit code 0 and no error markers; the unit is inactive. The five-seed cold study is now launching from [the full cold-fit plan](plans/s_dedicated_refiner_coldfit_five_seed_20261011.json), output `_/experiments/s_dedicated_refiner_coldfit_five_seed_20261011`.

## What changed

During acquisition fitting, the retained three-plane prefix is copied from the original baseline state and held bitwise fixed. The model suppresses stochastic/Gumbel behavior for those inherited prefix encode/prior calls, while the secondary network remains independently initialized and trainable. The fit objective uses only the retained baseline client-1 gradient and its saved decoder-side SI; it does not use the current target. Hard decoder inputs are enabled. Existing candidate/repeat selection remains in effect, so this is not a set of independent cold restarts.

The cold acquisition settings are 140 epochs, LR `0.0028364098431326932`, tau `5.5`, `lr_step=43`, batch size 6,250. The replacement final-S fit is unchanged at 28 epochs, LR `0.0006807383623518464`, tau `0.3`, batch size 7,500. The reference target has 11,181,642 floats. Rates below are bits per full-gradient float unless a column says bits per sampled symbol.

## Completed cold-pilot cases

The sampled coordinates are paired: E0 and E1 both use coordinate seed `494155588` and index hash `a1976420316c448a2d567182eccdedbf81a9194c055f0ab97f703c4e68c4aaab`. E1 records `base_prefix_unchanged=true`; its refiner fit used the baseline target hash. Neither fact alone establishes a multi-seed benefit.

| Case | Acquired-sample WMAPE % | Extra-stage WMAPE % (stage order) | Extra marginal H (bits/symbol; stage order) | H(extra | prefix + previous extras) (bits/symbol; stage order) | Extra prior (bits/full float; stage order) | Final S WMAPE % | Final full-SI rANS (bits/full float) | Refiner fit (s) | Case time (s) |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| S_e0 | 36.298751 | — | — | — | — | 39.561218 | 0.743830647 | — | 112.0 |
| S_e1 | 35.054898 | [35.054898] | [0.586878] | [0.029421] | [0.000392065] | 37.506239 | 0.755624263 | 285.2 | 114.1 |
| S_e3 | 30.830558 | [36.605902, 35.004466, 30.830558] | [0.511924, 0.630022, 0.503405] | [0.000398, 0.029666, 0.186957] | [0.000012799, 0.000362663, 0.002100144] | 43.206843 | 0.766607981 | 513.2 | 115.8 |

For S_e1, adding the cold-fitted extra stage improves acquired-sample WMAPE by 1.243853 percentage points against E0 at the same coordinates. S_e3 improves its final acquired WMAPE by 5.468193 points against E0; stage 4 slightly worsens WMAPE before stages 5 and 6 improve it. Its final-extra-stage prefix-conditioned shuffle gives 37.132008% WMAPE; replacing the final transmitted symbol with the decoder prior argmax gives 35.088683%. Both counterfactuals affect only the final extra stage: the shuffle retains discrete prefix/earlier-symbol context counts but breaks alignment with continuous SI; prior-mode tests reliance on the final transmitted symbol while changing decoder inputs. Neither demonstrates RD optimality.

The extra prior estimate is `0.000392065` bits/full float, or approximately `0.02192` bits per transmitted extra symbol (`0.000392065 × 11,181,642 / 200,000`). Measured extra rANS stream growth is `0.000389209` bits/full float, approximately `0.02174` bits per transmitted extra symbol. Total acquisition sample rANS is `0.018516780` bits/full float, charged at the first full-SI stage; sample overhead is separately recorded. The sample fit retains its own LR/tau settings, and its final S curve must not be interpreted as a causal acquisition-only comparison.

On this one seed, the final S full-SI rANS curve gap versus O is +19.552750% for E0, +17.186220% for E1, and +32.596842% for E3. E1 reduces the one-seed curve gap by 2.366531 points; E3 worsens it despite much better acquired labels. Every variant remains worse than O. This exposes a sample-to-final-S transfer problem under the unchanged final-S fit and makes the matched full test necessary; it is not an overall benefit claim.

## Interpretation limits from code and data audit

A read-only implementation audit found no acquisition scaling, stage-order, frozen-prefix-flag, or demonstrated seed-wiring defect. The E1/E3 cases use the same sampled coordinates, baseline state, target, and SI. CPU constructor draws differ with model size, but the final sampling/training path uses CUDA RNG and a separately seeded perturbation generator; no E-dependent optimizer/Gumbel stream divergence was demonstrated. Do not attribute the final-S loss to constructor RNG. The decoded labels differ across plane counts and can produce different optimization outcomes under the unchanged final-S warm fit.

The final-S fit remains the warm 28-epoch schedule even though its decoded labels are more accurate. The audit reports final-S training hard MSE `0.032` versus `0.0026` for the lower-error reference labels; the current 200,000-position pilot also leaves 98.2% of the 11,181,642-float target unsampled. The full study's 3,600,000 positions cover about 32.2% of that target. These facts make end-to-end final-S results sensitive to sample coverage and fit capacity. They do not erase the acquisition-stage WMAPE evidence, but they limit what can be inferred from that evidence about final compression.

## Earlier warm-pilot evidence and limits

The deterministic-prefix corrected warm pilot used the final-S schedule for the newly initialized refiner. Its E1 acquisition WMAPE was 36.3764% versus E0's 36.2988%; E3's final acquired stage was 35.6873%, and E6's was 35.9237%. A bounded diagnostic on saved warm E6 bins found stages 5–9 exact duplicates of stage 4 on the 50,000 inspected positions and conditional entropy zero on the 200,000-position pool. The warm E3 acquisition curve still contained a useful earlier stage. These are bounded observations from one seed and the saved sample pool, not a general claim that every trained refiner collapses.

## Full-study results — complete

The cold full study ran from [the full cold-fit plan](plans/s_dedicated_refiner_coldfit_five_seed_20261011.json) to `_/experiments/s_dedicated_refiner_coldfit_five_seed_20261011`. The supervisor's final health record is timestamped **2026-10-11 06:34:46 CEST (04:34:46 UTC)**: runner exited 0, 30/30 cases completed, and no error markers. It used seeds 43–47 for O and S_e0/e1/e2/e3/e6 with 3,600,000 paired sample positions. This is a fixed SVHN client-0, round-5, one-trained-round measurement with 150 decoder-visible SI columns, not a 1+9 protocol window. The completed [full report](../../_/experiments/s_dedicated_refiner_coldfit_five_seed_20261011/final_analysis/report.md) contains raw per-seed acquisition and final-S diagnostics, both directed matrices, and stage RD tables; the saved [summary](../../_/experiments/s_dedicated_refiner_coldfit_five_seed_20261011/final_analysis/summary.json) and [raw records](../../_/experiments/s_dedicated_refiner_coldfit_five_seed_20261011/final_analysis/raw_records.json) are alongside it.

The table gives the full-SI rANS equal-distortion rate gap versus O's all-five mean curve, computed from each cohort's mean RD curves. Positive values mean S uses more rate than O at equal distortion; reverse matrix entries are recomputed with their own denominator. These are not averages of per-seed gap percentages. The independent best-three selection ranks each method against the all-five O curve. S_e2 seed45 and S_e3 seed47 are unrankable (no valid shared distortion span) and were excluded before selecting the best three; this leaves the listed three seeds for each method.

| Method | All five seeds: gap vs O (%) | Best three seeds: gap vs O (%) | Best-three seeds |
|---|---:|---:|---|
| O | +0.000 | +0.000 | 43, 46, 47 |
| S_e0 | +13.489 | +13.831 | 44, 45, 46 |
| S_e1 | +15.277 | +12.945 | 43, 44, 46 |
| S_e2 | +28.757 | +17.024 | 43, 46, 47 |
| S_e3 | +20.334 | +16.376 | 43, 44, 45 |
| S_e6 | +38.111 | +19.027 | 43, 46, 47 |

Every S-versus-O aggregate gap is positive, including after best-three trimming: this study shows no overall RD-rate win over O at these settings. S's lower acquisition WMAPE does not establish a final compression benefit. With one measured client, there is no meaningful client-exclusion sensitivity cohort; the report records no exclusions and 1/1 retained.

The complete directed full-SI rANS matrices follow. Each cell is `rate_gap_percent(row, column)`; negative favors the row.

All five seeds:

```text
                     O       S_e0       S_e1       S_e2       S_e3       S_e6
          O     +0.000    -11.886    -13.252    -22.334    -16.898    -27.595
       S_e0    +13.489     +0.000     -0.824    -10.735     -5.172    -16.809
       S_e1    +15.277     +0.831     +0.000     -9.652     -4.460    -15.806
       S_e2    +28.757    +12.026    +10.684     +0.000     +5.851     -6.857
       S_e3    +20.334     +5.454     +4.668     -5.527     +0.000    -11.939
       S_e6    +38.111    +20.206    +18.774     +7.362    +13.557     +0.000
```

Best three per method:

```text
                     O       S_e0       S_e1       S_e2       S_e3       S_e6
          O     +0.000    -12.150    -11.461    -14.548    -14.071    -15.986
       S_e0    +13.831     +0.000     +0.879     -2.414     -2.183     -4.142
       S_e1    +12.945     -0.871     +0.000     -3.115     -3.119     -4.801
       S_e2    +17.024     +2.473     +3.215     +0.000     -0.040     -1.735
       S_e3    +16.376     +2.232     +3.219     +0.040     +0.000     -1.696
       S_e6    +19.027     +4.321     +5.043     +1.765     +1.725     +0.000
```

The native-transport matrices match these full-SI rANS matrices in this fixed-window study. Final S mean WMAPE and full-SI rANS rate are shown below for context; rates are bits per full-gradient float. Averaging WMAPE here describes the displayed cohort and is not used in place of the RD-curve comparison above.

| Method | Acquisition WMAPE %, all5 / best3 | Acquisition rANS, all5 / best3 (bits/full float) | Extra-plane rANS growth, all5 / best3 (bits/full float) | Final S WMAPE %, all5 / best3 | Final S full-SI rANS, all5 / best3 |
|---|---:|---:|---:|---:|---:|
| S_e0 | 36.3994 / 36.3837 | 0.251364 / 0.251470 | 0.000000 / 0.000000 | 39.3954 / 38.9767 | 0.747777688 / 0.742941093 |
| S_e1 | 32.8020 / 32.9567 | 0.276196 / 0.275995 | 0.024832 / 0.024617 | 37.1096 / 36.8885 | 0.763283103 / 0.754322129 |
| S_e2 | 30.6354 / 31.1062 | 0.344614 / 0.290040 | 0.093251 / 0.038722 | 35.6040 / 36.2559 | 0.846490059 / 0.781742789 |
| S_e3 | 32.1134 / 32.2866 | 0.289726 / 0.281538 | 0.038362 / 0.030196 | 37.7492 / 36.5619 | 0.796002161 / 0.788062672 |
| S_e6 | 27.7803 / 28.9673 | 0.432548 / 0.315350 | 0.181185 / 0.064033 | 35.4586 / 36.5690 | 0.911955650 / 0.787441654 |

Acquisition WMAPE is the final acquired stage, including the ordinary three-plane prefix. The following values show the per-extra-stage acquisition WMAPE sequence (stage 1 onward), averaged across all five / selected best-three fits. S_e0 has no extra stage.

| Method | Extra-stage acquisition WMAPE %, all5 / best3 |
|---|---|
| S_e1 | 32.8020 / 32.9567 |
| S_e2 | [34.1201, 30.6354] / [34.0365, 31.1062] |
| S_e3 | [34.9748, 33.7659, 32.1134] / [35.7989, 33.7984, 32.2866] |
| S_e6 | [34.1505, 33.7529, 32.0376, 29.2632, 28.1258, 27.7803] / [34.7939, 34.1841, 32.6920, 30.4886, 29.2284, 28.9673] |

All 30 completed records have zero symbol errors. S_e6 seed47 stages 8 and 9 are dead (one occupied bin); other low-entropy stage flags and per-seed failures are listed in the [full report](../../_/experiments/s_dedicated_refiner_coldfit_five_seed_20261011/final_analysis/report.md). Final-S curves for S_e2 seed45 and S_e3 seed47 are nonmonotone/unrankable and were not selected. Best-three trimming is per-method fit-seed selection, not a matched-seed estimate.

### Pilot versus full-size evidence

The earlier cold pilot used only 200,000 sampled positions. Its S_e3 final-S WMAPE was 43.206843%; the 3.6M full-size seed43 case was 34.797471%. These are different sample sizes and the pilot result must not be extrapolated to the full study. The full-size seed43 values retained in the earlier progress snapshot were interim one-seed diagnostics only; the completed all-five and best-three results above supersede them for aggregate claims. The full study's 3.6M positions cover about 32.2% of the 11,181,642-float target.
