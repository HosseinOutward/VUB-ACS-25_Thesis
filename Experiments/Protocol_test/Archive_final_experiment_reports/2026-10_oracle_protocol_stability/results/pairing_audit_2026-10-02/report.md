# Matched protocol RD and outlier sensitivity

Negative gaps favor the first codec. Rates are cumulative bits/float; WMAPE is averaged in linear space before dB conversion.
Outliers: any native or full-SI rANS stage <0.01 or >1.05 bits anywhere in that case's observed window.
Exclude the union of flagged clients from BOTH codecs at the same start, for all matched rounds. This is post-hoc sensitivity analysis, not a deployable selection policy.
The legacy OMR/TMR comparison retains only complete matched rounds 6–12. Revised OMR/TMR has only two trained clients.
OR/OMR has a shared start only at 12 (14 trained clients). The supplementary frozen comparison uses OR start5 versus OMR start7, rounds 8–14: different training placement/history, not a controlled ablation.
Fine-tuned prior stage breakdowns are unavailable: only final totals are shown, so no equal-distortion fine-tuned-prior curve gap is inferred.

## Exclusions

| Pairing | Starts | Observed outliers | Removed matched clients |
|---|---|---|---|
| OR vs TR | 5/5 | [] | [] |
| OR vs TR | 12/12 | [1, 13, 20, 21] | [1, 13, 20, 21] |
| OR vs TR | 20/20 | [10, 15] | [10, 15] |
| OR vs OMR | 12/12 | [1, 13, 20, 21] | [1, 13] |
| OMR vs TMR revised | 7/7 | [21] | [] |
| OMR vs TMR legacy | 6/6 | [1, 5, 11] | [1, 5, 11] |
| OR vs OMR frozen different placements | 5/7 | [21] | [21] |

## Equal-distortion gaps

| Pairing | Start | Population | Phase | Rows/codec | Source | Gap bits/float | Gap % |
|---|---|---|---|---:|---|---:|---:|
| OR vs TR | 5 | all clients | overall | 300 | transport_rans | -0.06081 | -10.89% |
| OR vs TR | 5 | all clients | overall | 300 | full_rans | -0.04685 | -8.61% |
| OR vs TR | 5 | all clients | overall | 300 | pre_rate | -0.31009 | -34.14% |
| OR vs TR | 5 | all clients | trained | 30 | transport_rans | -0.05787 | -10.70% |
| OR vs TR | 5 | all clients | trained | 30 | full_rans | -0.03752 | -7.21% |
| OR vs TR | 5 | all clients | trained | 30 | pre_rate | -0.31610 | -35.38% |
| OR vs TR | 5 | all clients | frozen | 270 | transport_rans | -0.06102 | -10.89% |
| OR vs TR | 5 | all clients | frozen | 270 | full_rans | -0.04780 | -8.74% |
| OR vs TR | 5 | all clients | frozen | 270 | pre_rate | -0.30928 | -33.99% |
| OR vs TR | 5 | exclude outliers | overall | 300 | transport_rans | -0.06081 | -10.89% |
| OR vs TR | 5 | exclude outliers | overall | 300 | full_rans | -0.04685 | -8.61% |
| OR vs TR | 5 | exclude outliers | overall | 300 | pre_rate | -0.31009 | -34.14% |
| OR vs TR | 5 | exclude outliers | trained | 30 | transport_rans | -0.05787 | -10.70% |
| OR vs TR | 5 | exclude outliers | trained | 30 | full_rans | -0.03752 | -7.21% |
| OR vs TR | 5 | exclude outliers | trained | 30 | pre_rate | -0.31610 | -35.38% |
| OR vs TR | 5 | exclude outliers | frozen | 270 | transport_rans | -0.06102 | -10.89% |
| OR vs TR | 5 | exclude outliers | frozen | 270 | full_rans | -0.04780 | -8.74% |
| OR vs TR | 5 | exclude outliers | frozen | 270 | pre_rate | -0.30928 | -33.99% |
| OR vs TR | 12 | all clients | overall | 300 | transport_rans | +0.02714 | +4.92% |
| OR vs TR | 12 | all clients | overall | 300 | full_rans | +0.04208 | +7.85% |
| OR vs TR | 12 | all clients | overall | 300 | pre_rate | -0.12469 | -14.61% |
| OR vs TR | 12 | all clients | trained | 30 | transport_rans | -0.05317 | -9.90% |
| OR vs TR | 12 | all clients | trained | 30 | full_rans | -0.03397 | -6.56% |
| OR vs TR | 12 | all clients | trained | 30 | pre_rate | -0.14448 | -17.41% |
| OR vs TR | 12 | all clients | frozen | 270 | transport_rans | +0.03606 | +6.52% |
| OR vs TR | 12 | all clients | frozen | 270 | full_rans | +0.05053 | +9.38% |
| OR vs TR | 12 | all clients | frozen | 270 | pre_rate | -0.12250 | -14.31% |
| OR vs TR | 12 | exclude outliers | overall | 260 | transport_rans | -0.04589 | -8.37% |
| OR vs TR | 12 | exclude outliers | overall | 260 | full_rans | -0.03004 | -5.64% |
| OR vs TR | 12 | exclude outliers | overall | 260 | pre_rate | -0.13761 | -16.59% |
| OR vs TR | 12 | exclude outliers | trained | 26 | transport_rans | -0.05451 | -10.17% |
| OR vs TR | 12 | exclude outliers | trained | 26 | full_rans | -0.03546 | -6.86% |
| OR vs TR | 12 | exclude outliers | trained | 26 | pre_rate | -0.15214 | -18.92% |
| OR vs TR | 12 | exclude outliers | frozen | 234 | transport_rans | -0.04494 | -8.18% |
| OR vs TR | 12 | exclude outliers | frozen | 234 | full_rans | -0.02944 | -5.51% |
| OR vs TR | 12 | exclude outliers | frozen | 234 | pre_rate | -0.13601 | -16.34% |
| OR vs TR | 20 | all clients | overall | 300 | transport_rans | -0.12551 | -19.04% |
| OR vs TR | 20 | all clients | overall | 300 | full_rans | -0.08560 | -13.82% |
| OR vs TR | 20 | all clients | overall | 300 | pre_rate | -0.16278 | -16.80% |
| OR vs TR | 20 | all clients | trained | 30 | transport_rans | -0.05557 | -10.02% |
| OR vs TR | 20 | all clients | trained | 30 | full_rans | -0.03668 | -6.85% |
| OR vs TR | 20 | all clients | trained | 30 | pre_rate | -0.17167 | -18.12% |
| OR vs TR | 20 | all clients | frozen | 270 | transport_rans | -0.13330 | -19.87% |
| OR vs TR | 20 | all clients | frozen | 270 | full_rans | -0.09103 | -14.48% |
| OR vs TR | 20 | all clients | frozen | 270 | pre_rate | -0.16183 | -16.66% |
| OR vs TR | 20 | exclude outliers | overall | 280 | transport_rans | -0.04905 | -8.60% |
| OR vs TR | 20 | exclude outliers | overall | 280 | full_rans | -0.03614 | -6.48% |
| OR vs TR | 20 | exclude outliers | overall | 280 | pre_rate | -0.15225 | -16.03% |
| OR vs TR | 20 | exclude outliers | trained | 28 | transport_rans | -0.05512 | -9.98% |
| OR vs TR | 20 | exclude outliers | trained | 28 | full_rans | -0.03611 | -6.77% |
| OR vs TR | 20 | exclude outliers | trained | 28 | pre_rate | -0.16276 | -17.55% |
| OR vs TR | 20 | exclude outliers | frozen | 252 | transport_rans | -0.04838 | -8.45% |
| OR vs TR | 20 | exclude outliers | frozen | 252 | full_rans | -0.03614 | -6.45% |
| OR vs TR | 20 | exclude outliers | frozen | 252 | pre_rate | -0.15111 | -15.87% |
| OR vs OMR | 12 | all clients | overall | 14 | transport_rans | -0.01050 | -1.98% |
| OR vs OMR | 12 | all clients | overall | 14 | full_rans | -0.01050 | -1.98% |
| OR vs OMR | 12 | all clients | overall | 14 | pre_rate | -0.01852 | -2.61% |
| OR vs OMR | 12 | all clients | trained | 14 | transport_rans | -0.01050 | -1.98% |
| OR vs OMR | 12 | all clients | trained | 14 | full_rans | -0.01050 | -1.98% |
| OR vs OMR | 12 | all clients | trained | 14 | pre_rate | -0.01852 | -2.61% |
| OR vs OMR | 12 | exclude outliers | overall | 12 | transport_rans | -0.01238 | -2.34% |
| OR vs OMR | 12 | exclude outliers | overall | 12 | full_rans | -0.01238 | -2.34% |
| OR vs OMR | 12 | exclude outliers | overall | 12 | pre_rate | -0.05456 | -7.90% |
| OR vs OMR | 12 | exclude outliers | trained | 12 | transport_rans | -0.01238 | -2.34% |
| OR vs OMR | 12 | exclude outliers | trained | 12 | full_rans | -0.01238 | -2.34% |
| OR vs OMR | 12 | exclude outliers | trained | 12 | pre_rate | -0.05456 | -7.90% |
| OMR vs TMR revised | 7 | all clients | overall | 2 | transport_rans | -0.02710 | -5.03% |
| OMR vs TMR revised | 7 | all clients | overall | 2 | full_rans | -0.00333 | -0.65% |
| OMR vs TMR revised | 7 | all clients | overall | 2 | pre_rate | -0.01825 | -3.35% |
| OMR vs TMR revised | 7 | all clients | trained | 2 | transport_rans | -0.02710 | -5.03% |
| OMR vs TMR revised | 7 | all clients | trained | 2 | full_rans | -0.00333 | -0.65% |
| OMR vs TMR revised | 7 | all clients | trained | 2 | pre_rate | -0.01825 | -3.35% |
| OMR vs TMR revised | 7 | exclude outliers | overall | 2 | transport_rans | -0.02710 | -5.03% |
| OMR vs TMR revised | 7 | exclude outliers | overall | 2 | full_rans | -0.00333 | -0.65% |
| OMR vs TMR revised | 7 | exclude outliers | overall | 2 | pre_rate | -0.01825 | -3.35% |
| OMR vs TMR revised | 7 | exclude outliers | trained | 2 | transport_rans | -0.02710 | -5.03% |
| OMR vs TMR revised | 7 | exclude outliers | trained | 2 | full_rans | -0.00333 | -0.65% |
| OMR vs TMR revised | 7 | exclude outliers | trained | 2 | pre_rate | -0.01825 | -3.35% |
| OMR vs TMR legacy | 6 | all clients | overall | 210 | transport_rans | +0.03154 | +5.48% |
| OMR vs TMR legacy | 6 | all clients | overall | 210 | full_rans | +0.04135 | +7.31% |
| OMR vs TMR legacy | 6 | all clients | overall | 210 | pre_rate | +0.08691 | +11.70% |
| OMR vs TMR legacy | 6 | all clients | trained | 30 | transport_rans | -0.04213 | -7.60% |
| OMR vs TMR legacy | 6 | all clients | trained | 30 | full_rans | -0.01927 | -3.63% |
| OMR vs TMR legacy | 6 | all clients | trained | 30 | pre_rate | +0.07044 | +9.60% |
| OMR vs TMR legacy | 6 | all clients | frozen | 180 | transport_rans | +0.04384 | +7.58% |
| OMR vs TMR legacy | 6 | all clients | frozen | 180 | full_rans | +0.05147 | +9.01% |
| OMR vs TMR legacy | 6 | all clients | frozen | 180 | pre_rate | +0.08969 | +12.05% |
| OMR vs TMR legacy | 6 | exclude outliers | overall | 189 | transport_rans | -0.04627 | -8.23% |
| OMR vs TMR legacy | 6 | exclude outliers | overall | 189 | full_rans | -0.02818 | -5.18% |
| OMR vs TMR legacy | 6 | exclude outliers | overall | 189 | pre_rate | +0.06366 | +8.89% |
| OMR vs TMR legacy | 6 | exclude outliers | trained | 27 | transport_rans | -0.04258 | -7.72% |
| OMR vs TMR legacy | 6 | exclude outliers | trained | 27 | full_rans | -0.01974 | -3.74% |
| OMR vs TMR legacy | 6 | exclude outliers | trained | 27 | pre_rate | +0.06759 | +9.57% |
| OMR vs TMR legacy | 6 | exclude outliers | frozen | 162 | transport_rans | -0.04683 | -8.30% |
| OMR vs TMR legacy | 6 | exclude outliers | frozen | 162 | full_rans | -0.02954 | -5.40% |
| OMR vs TMR legacy | 6 | exclude outliers | frozen | 162 | pre_rate | +0.06307 | +8.78% |
| OR vs OMR frozen different placements | 5 | all clients | overall | 210 | transport_rans | -0.02041 | -3.89% |
| OR vs OMR frozen different placements | 5 | all clients | overall | 210 | full_rans | -0.02041 | -3.89% |
| OR vs OMR frozen different placements | 5 | all clients | overall | 210 | pre_rate | -0.01152 | -1.86% |
| OR vs OMR frozen different placements | 5 | all clients | frozen | 210 | transport_rans | -0.02041 | -3.89% |
| OR vs OMR frozen different placements | 5 | all clients | frozen | 210 | full_rans | -0.02041 | -3.89% |
| OR vs OMR frozen different placements | 5 | all clients | frozen | 210 | pre_rate | -0.01152 | -1.86% |
| OR vs OMR frozen different placements | 5 | exclude outliers | overall | 203 | transport_rans | +0.01156 | +2.35% |
| OR vs OMR frozen different placements | 5 | exclude outliers | overall | 203 | full_rans | +0.01156 | +2.35% |
| OR vs OMR frozen different placements | 5 | exclude outliers | overall | 203 | pre_rate | +0.00724 | +1.20% |
| OR vs OMR frozen different placements | 5 | exclude outliers | frozen | 203 | transport_rans | +0.01156 | +2.35% |
| OR vs OMR frozen different placements | 5 | exclude outliers | frozen | 203 | full_rans | +0.01156 | +2.35% |
| OR vs OMR frozen different placements | 5 | exclude outliers | frozen | 203 | pre_rate | +0.00724 | +1.20% |
| OR vs TR | pooled | all clients | frozen | 810 | transport_rans | -0.05209 | -8.66% |
| OR vs TR | pooled | all clients | frozen | 810 | full_rans | -0.02855 | -4.94% |
| OR vs TR | pooled | all clients | frozen | 810 | pre_rate | -0.19643 | -21.35% |
| OR vs TR | pooled | all clients | overall | 900 | transport_rans | -0.05244 | -8.79% |
| OR vs TR | pooled | all clients | overall | 900 | full_rans | -0.02926 | -5.10% |
| OR vs TR | pooled | all clients | overall | 900 | pre_rate | -0.19783 | -21.55% |
| OR vs TR | pooled | all clients | trained | 90 | transport_rans | -0.05544 | -10.05% |
| OR vs TR | pooled | all clients | trained | 90 | full_rans | -0.03556 | -6.69% |
| OR vs TR | pooled | all clients | trained | 90 | pre_rate | -0.21052 | -23.44% |
| OR vs TR | pooled | exclude outliers | frozen | 756 | transport_rans | -0.05118 | -9.02% |
| OR vs TR | pooled | exclude outliers | frozen | 756 | full_rans | -0.03737 | -6.75% |
| OR vs TR | pooled | exclude outliers | frozen | 756 | pre_rate | -0.20148 | -22.22% |
| OR vs TR | pooled | exclude outliers | overall | 840 | transport_rans | -0.05166 | -9.14% |
| OR vs TR | pooled | exclude outliers | overall | 840 | full_rans | -0.03724 | -6.76% |
| OR vs TR | pooled | exclude outliers | overall | 840 | pre_rate | -0.20271 | -22.41% |
| OR vs TR | pooled | exclude outliers | trained | 84 | transport_rans | -0.05582 | -10.15% |
| OR vs TR | pooled | exclude outliers | trained | 84 | full_rans | -0.03595 | -6.78% |
| OR vs TR | pooled | exclude outliers | trained | 84 | pre_rate | -0.21385 | -24.19% |

## OR vs TR / 5 / 5 / all clients / overall

Rows per codec: 300. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4495 | — | — | — | 0.3676 | -3.375 (67.803%) |
| OR | 2 | 0.5434 | — | — | — | 0.4474 | -4.403 (60.237%) |
| OR | 3 | 0.7367 | — | 0.6160 | — | 0.6219 | -6.901 (45.179%) |
| TR | 1 | 0.7039 | — | — | 0.3752 | 0.3727 | -2.179 (77.815%) |
| TR | 2 | 0.7982 | — | — | 0.4589 | 0.4493 | -3.592 (66.131%) |
| TR | 3 | 1.0674 | 0.6962 | 0.6759 | 0.7020 | 0.6817 | -7.373 (42.793%) |

## OR vs TR / 5 / 5 / all clients / trained

Rows per codec: 30. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4387 | — | — | — | 0.3591 | -3.700 (65.313%) |
| OR | 2 | 0.5238 | — | — | — | 0.4321 | -4.638 (58.628%) |
| OR | 3 | 0.7050 | — | 0.5957 | — | 0.6015 | -7.059 (44.365%) |
| TR | 1 | 0.7048 | — | — | 0.3738 | 0.3676 | -2.950 (71.204%) |
| TR | 2 | 0.7945 | — | — | 0.4532 | 0.4378 | -4.066 (62.615%) |
| TR | 3 | 1.0642 | 0.6864 | 0.6574 | 0.6924 | 0.6633 | -7.646 (41.465%) |

## OR vs TR / 5 / 5 / all clients / frozen

Rows per codec: 270. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4507 | — | — | — | 0.3685 | -3.340 (68.079%) |
| OR | 2 | 0.5456 | — | — | — | 0.4490 | -4.377 (60.416%) |
| OR | 3 | 0.7402 | — | 0.6183 | — | 0.6241 | -6.884 (45.270%) |
| TR | 1 | 0.7038 | — | — | 0.3753 | 0.3733 | -2.097 (78.549%) |
| TR | 2 | 0.7986 | — | — | 0.4595 | 0.4506 | -3.541 (66.522%) |
| TR | 3 | 1.0677 | 0.6973 | 0.6780 | 0.7031 | 0.6837 | -7.343 (42.940%) |

## OR vs TR / 5 / 5 / exclude outliers / overall

Rows per codec: 300. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4495 | — | — | — | 0.3676 | -3.375 (67.803%) |
| OR | 2 | 0.5434 | — | — | — | 0.4474 | -4.403 (60.237%) |
| OR | 3 | 0.7367 | — | 0.6160 | — | 0.6219 | -6.901 (45.179%) |
| TR | 1 | 0.7039 | — | — | 0.3752 | 0.3727 | -2.179 (77.815%) |
| TR | 2 | 0.7982 | — | — | 0.4589 | 0.4493 | -3.592 (66.131%) |
| TR | 3 | 1.0674 | 0.6962 | 0.6759 | 0.7020 | 0.6817 | -7.373 (42.793%) |

## OR vs TR / 5 / 5 / exclude outliers / trained

Rows per codec: 30. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4387 | — | — | — | 0.3591 | -3.700 (65.313%) |
| OR | 2 | 0.5238 | — | — | — | 0.4321 | -4.638 (58.628%) |
| OR | 3 | 0.7050 | — | 0.5957 | — | 0.6015 | -7.059 (44.365%) |
| TR | 1 | 0.7048 | — | — | 0.3738 | 0.3676 | -2.950 (71.204%) |
| TR | 2 | 0.7945 | — | — | 0.4532 | 0.4378 | -4.066 (62.615%) |
| TR | 3 | 1.0642 | 0.6864 | 0.6574 | 0.6924 | 0.6633 | -7.646 (41.465%) |

## OR vs TR / 5 / 5 / exclude outliers / frozen

Rows per codec: 270. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4507 | — | — | — | 0.3685 | -3.340 (68.079%) |
| OR | 2 | 0.5456 | — | — | — | 0.4490 | -4.377 (60.416%) |
| OR | 3 | 0.7402 | — | 0.6183 | — | 0.6241 | -6.884 (45.270%) |
| TR | 1 | 0.7038 | — | — | 0.3753 | 0.3733 | -2.097 (78.549%) |
| TR | 2 | 0.7986 | — | — | 0.4595 | 0.4506 | -3.541 (66.522%) |
| TR | 3 | 1.0677 | 0.6973 | 0.6780 | 0.7031 | 0.6837 | -7.343 (42.940%) |

## OR vs TR / 12 / 12 / all clients / overall

Rows per codec: 300. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5731 | — | — | — | 0.4490 | -3.548 (66.467%) |
| OR | 2 | 0.6631 | — | — | — | 0.5233 | -4.423 (60.093%) |
| OR | 3 | 0.9275 | — | 0.7383 | — | 0.7445 | -7.516 (42.090%) |
| TR | 1 | 0.6209 | — | — | 0.3706 | 0.3660 | -2.640 (73.793%) |
| TR | 2 | 0.7277 | — | — | 0.4554 | 0.4441 | -3.851 (64.185%) |
| TR | 3 | 1.0072 | 0.6629 | 0.6435 | 0.6686 | 0.6492 | -6.885 (45.266%) |

## OR vs TR / 12 / 12 / all clients / trained

Rows per codec: 30. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5543 | — | — | — | 0.3647 | -3.743 (64.993%) |
| OR | 2 | 0.6279 | — | — | — | 0.4299 | -4.544 (59.265%) |
| OR | 3 | 0.8523 | — | 0.6330 | — | 0.6390 | -7.602 (41.676%) |
| TR | 1 | 0.6160 | — | — | 0.3631 | 0.3579 | -2.872 (71.843%) |
| TR | 2 | 0.7083 | — | — | 0.4424 | 0.4281 | -3.976 (63.271%) |
| TR | 3 | 0.9722 | 0.6421 | 0.6172 | 0.6480 | 0.6230 | -6.965 (44.850%) |

## OR vs TR / 12 / 12 / all clients / frozen

Rows per codec: 270. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5752 | — | — | — | 0.4584 | -3.526 (66.631%) |
| OR | 2 | 0.6670 | — | — | — | 0.5337 | -4.410 (60.185%) |
| OR | 3 | 0.9358 | — | 0.7500 | — | 0.7562 | -7.507 (42.136%) |
| TR | 1 | 0.6214 | — | — | 0.3714 | 0.3669 | -2.614 (74.009%) |
| TR | 2 | 0.7299 | — | — | 0.4568 | 0.4458 | -3.838 (64.287%) |
| TR | 3 | 1.0111 | 0.6652 | 0.6464 | 0.6709 | 0.6521 | -6.876 (45.312%) |

## OR vs TR / 12 / 12 / exclude outliers / overall

Rows per codec: 260. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5353 | — | — | — | 0.3728 | -3.568 (66.311%) |
| OR | 2 | 0.6262 | — | — | — | 0.4475 | -4.444 (59.954%) |
| OR | 3 | 0.8898 | — | 0.6619 | — | 0.6677 | -7.500 (42.170%) |
| TR | 1 | 0.5904 | — | — | 0.3662 | 0.3609 | -2.674 (73.503%) |
| TR | 2 | 0.6989 | — | — | 0.4516 | 0.4395 | -3.864 (64.089%) |
| TR | 3 | 0.9880 | 0.6603 | 0.6398 | 0.6660 | 0.6455 | -6.885 (45.263%) |

## OR vs TR / 12 / 12 / exclude outliers / trained

Rows per codec: 26. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5206 | — | — | — | 0.3621 | -3.740 (65.015%) |
| OR | 2 | 0.5949 | — | — | — | 0.4276 | -4.546 (59.253%) |
| OR | 3 | 0.8182 | — | 0.6298 | — | 0.6357 | -7.583 (41.769%) |
| TR | 1 | 0.5848 | — | — | 0.3605 | 0.3555 | -2.860 (71.944%) |
| TR | 2 | 0.6780 | — | — | 0.4404 | 0.4263 | -3.967 (63.334%) |
| TR | 3 | 0.9510 | 0.6412 | 0.6164 | 0.6470 | 0.6222 | -6.958 (44.886%) |

## OR vs TR / 12 / 12 / exclude outliers / frozen

Rows per codec: 234. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5369 | — | — | — | 0.3740 | -3.549 (66.455%) |
| OR | 2 | 0.6297 | — | — | — | 0.4497 | -4.432 (60.032%) |
| OR | 3 | 0.8978 | — | 0.6655 | — | 0.6712 | -7.491 (42.214%) |
| TR | 1 | 0.5910 | — | — | 0.3668 | 0.3615 | -2.654 (73.676%) |
| TR | 2 | 0.7012 | — | — | 0.4529 | 0.4410 | -3.853 (64.173%) |
| TR | 3 | 0.9921 | 0.6624 | 0.6424 | 0.6681 | 0.6480 | -6.877 (45.305%) |

## OR vs TR / 20 / 20 / all clients / overall

Rows per codec: 300. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.6433 | — | — | — | 0.3901 | -3.048 (70.408%) |
| OR | 2 | 0.7287 | — | — | — | 0.4645 | -4.098 (62.384%) |
| OR | 3 | 0.9654 | — | 0.6691 | — | 0.6751 | -7.201 (43.646%) |
| TR | 1 | 0.7210 | — | — | 0.4629 | 0.4333 | -1.906 (80.296%) |
| TR | 2 | 0.8569 | — | — | 0.5553 | 0.5182 | -3.603 (66.050%) |
| TR | 3 | 1.1479 | 0.8177 | 0.7727 | 0.8232 | 0.7785 | -7.543 (41.963%) |

## OR vs TR / 20 / 20 / all clients / trained

Rows per codec: 30. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.6337 | — | — | — | 0.3671 | -3.722 (65.145%) |
| OR | 2 | 0.7033 | — | — | — | 0.4299 | -4.537 (59.315%) |
| OR | 3 | 0.9088 | — | 0.6186 | — | 0.6245 | -7.441 (42.459%) |
| TR | 1 | 0.7164 | — | — | 0.3682 | 0.3643 | -2.871 (71.856%) |
| TR | 2 | 0.8403 | — | — | 0.4518 | 0.4378 | -4.074 (62.563%) |
| TR | 3 | 1.1088 | 0.7018 | 0.6755 | 0.7079 | 0.6815 | -7.828 (40.607%) |

## OR vs TR / 20 / 20 / all clients / frozen

Rows per codec: 270. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.6443 | — | — | — | 0.3927 | -2.976 (70.993%) |
| OR | 2 | 0.7315 | — | — | — | 0.4683 | -4.051 (62.726%) |
| OR | 3 | 0.9716 | — | 0.6748 | — | 0.6807 | -7.175 (43.778%) |
| TR | 1 | 0.7215 | — | — | 0.4735 | 0.4410 | -1.805 (81.234%) |
| TR | 2 | 0.8587 | — | — | 0.5667 | 0.5272 | -3.552 (66.437%) |
| TR | 3 | 1.1523 | 0.8305 | 0.7835 | 0.8360 | 0.7892 | -7.511 (42.114%) |

## OR vs TR / 20 / 20 / exclude outliers / overall

Rows per codec: 280. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.6348 | — | — | — | 0.3783 | -3.045 (70.430%) |
| OR | 2 | 0.7205 | — | — | — | 0.4528 | -4.093 (62.423%) |
| OR | 3 | 0.9563 | — | 0.6558 | — | 0.6615 | -7.166 (43.824%) |
| TR | 1 | 0.6983 | — | — | 0.3732 | 0.3707 | -1.879 (80.546%) |
| TR | 2 | 0.8371 | — | — | 0.4660 | 0.4560 | -3.586 (66.173%) |
| TR | 3 | 1.1302 | 0.7290 | 0.7111 | 0.7351 | 0.7173 | -7.528 (42.036%) |

## OR vs TR / 20 / 20 / exclude outliers / trained

Rows per codec: 28. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.6234 | — | — | — | 0.3664 | -3.714 (65.211%) |
| OR | 2 | 0.6927 | — | — | — | 0.4292 | -4.526 (59.390%) |
| OR | 3 | 0.8965 | — | 0.6158 | — | 0.6218 | -7.401 (42.653%) |
| TR | 1 | 0.6951 | — | — | 0.3672 | 0.3631 | -2.877 (71.802%) |
| TR | 2 | 0.8213 | — | — | 0.4509 | 0.4367 | -4.076 (62.545%) |
| TR | 3 | 1.0911 | 0.7019 | 0.6753 | 0.7079 | 0.6813 | -7.825 (40.623%) |

## OR vs TR / 20 / 20 / exclude outliers / frozen

Rows per codec: 252. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.6361 | — | — | — | 0.3796 | -2.974 (71.010%) |
| OR | 2 | 0.7236 | — | — | — | 0.4555 | -4.046 (62.760%) |
| OR | 3 | 0.9630 | — | 0.6602 | — | 0.6660 | -7.140 (43.954%) |
| TR | 1 | 0.6987 | — | — | 0.3738 | 0.3715 | -1.775 (81.518%) |
| TR | 2 | 0.8388 | — | — | 0.4677 | 0.4582 | -3.534 (66.576%) |
| TR | 3 | 1.1345 | 0.7320 | 0.7151 | 0.7382 | 0.7213 | -7.495 (42.193%) |

## OR vs OMR / 12 / 12 / all clients / overall

Rows per codec: 14. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5253 | — | — | — | 0.3658 | -3.770 (64.790%) |
| OR | 2 | 0.6052 | — | — | — | 0.4367 | -4.660 (58.481%) |
| OR | 3 | 0.8434 | — | 0.6567 | — | 0.6627 | -8.035 (39.652%) |
| OMR | 1 | 0.5351 | — | — | — | 0.3668 | -3.633 (65.817%) |
| OMR | 2 | 0.6162 | — | — | — | 0.4390 | -4.536 (59.319%) |
| OMR | 3 | 0.8609 | — | 0.6683 | — | 0.6742 | -8.061 (39.530%) |

## OR vs OMR / 12 / 12 / all clients / trained

Rows per codec: 14. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5253 | — | — | — | 0.3658 | -3.770 (64.790%) |
| OR | 2 | 0.6052 | — | — | — | 0.4367 | -4.660 (58.481%) |
| OR | 3 | 0.8434 | — | 0.6567 | — | 0.6627 | -8.035 (39.652%) |
| OMR | 1 | 0.5351 | — | — | — | 0.3668 | -3.633 (65.817%) |
| OMR | 2 | 0.6162 | — | — | — | 0.4390 | -4.536 (59.319%) |
| OMR | 3 | 0.8609 | — | 0.6683 | — | 0.6742 | -8.061 (39.530%) |

## OR vs OMR / 12 / 12 / exclude outliers / overall

Rows per codec: 12. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4687 | — | — | — | 0.3634 | -3.758 (64.877%) |
| OR | 2 | 0.5508 | — | — | — | 0.4356 | -4.667 (58.432%) |
| OR | 3 | 0.7897 | — | 0.6568 | — | 0.6628 | -8.061 (39.532%) |
| OMR | 1 | 0.5196 | — | — | — | 0.3687 | -3.659 (65.621%) |
| OMR | 2 | 0.5995 | — | — | — | 0.4415 | -4.551 (59.216%) |
| OMR | 3 | 0.8452 | — | 0.6725 | — | 0.6784 | -8.142 (39.167%) |

## OR vs OMR / 12 / 12 / exclude outliers / trained

Rows per codec: 12. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4687 | — | — | — | 0.3634 | -3.758 (64.877%) |
| OR | 2 | 0.5508 | — | — | — | 0.4356 | -4.667 (58.432%) |
| OR | 3 | 0.7897 | — | 0.6568 | — | 0.6628 | -8.061 (39.532%) |
| OMR | 1 | 0.5196 | — | — | — | 0.3687 | -3.659 (65.621%) |
| OMR | 2 | 0.5995 | — | — | — | 0.4415 | -4.551 (59.216%) |
| OMR | 3 | 0.8452 | — | 0.6725 | — | 0.6784 | -8.142 (39.167%) |

## OMR vs TMR revised / 7 / 7 / all clients / overall

Rows per codec: 2. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR | 1 | 0.3760 | — | — | — | 0.3704 | -3.816 (64.444%) |
| OMR | 2 | 0.4467 | — | — | — | 0.4348 | -4.505 (59.530%) |
| OMR | 3 | 0.6596 | — | 0.6330 | — | 0.6386 | -7.461 (42.358%) |
| TMR | 1 | 0.3428 | — | — | 0.3456 | 0.3386 | -2.906 (71.567%) |
| TMR | 2 | 0.4443 | — | — | 0.4392 | 0.4209 | -4.185 (61.766%) |
| TMR | 3 | 0.7250 | 0.7107 | 0.6767 | 0.7166 | 0.6828 | -8.245 (38.705%) |

## OMR vs TMR revised / 7 / 7 / all clients / trained

Rows per codec: 2. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR | 1 | 0.3760 | — | — | — | 0.3704 | -3.816 (64.444%) |
| OMR | 2 | 0.4467 | — | — | — | 0.4348 | -4.505 (59.530%) |
| OMR | 3 | 0.6596 | — | 0.6330 | — | 0.6386 | -7.461 (42.358%) |
| TMR | 1 | 0.3428 | — | — | 0.3456 | 0.3386 | -2.906 (71.567%) |
| TMR | 2 | 0.4443 | — | — | 0.4392 | 0.4209 | -4.185 (61.766%) |
| TMR | 3 | 0.7250 | 0.7107 | 0.6767 | 0.7166 | 0.6828 | -8.245 (38.705%) |

## OMR vs TMR revised / 7 / 7 / exclude outliers / overall

Rows per codec: 2. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR | 1 | 0.3760 | — | — | — | 0.3704 | -3.816 (64.444%) |
| OMR | 2 | 0.4467 | — | — | — | 0.4348 | -4.505 (59.530%) |
| OMR | 3 | 0.6596 | — | 0.6330 | — | 0.6386 | -7.461 (42.358%) |
| TMR | 1 | 0.3428 | — | — | 0.3456 | 0.3386 | -2.906 (71.567%) |
| TMR | 2 | 0.4443 | — | — | 0.4392 | 0.4209 | -4.185 (61.766%) |
| TMR | 3 | 0.7250 | 0.7107 | 0.6767 | 0.7166 | 0.6828 | -8.245 (38.705%) |

## OMR vs TMR revised / 7 / 7 / exclude outliers / trained

Rows per codec: 2. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR | 1 | 0.3760 | — | — | — | 0.3704 | -3.816 (64.444%) |
| OMR | 2 | 0.4467 | — | — | — | 0.4348 | -4.505 (59.530%) |
| OMR | 3 | 0.6596 | — | 0.6330 | — | 0.6386 | -7.461 (42.358%) |
| TMR | 1 | 0.3428 | — | — | 0.3456 | 0.3386 | -2.906 (71.567%) |
| TMR | 2 | 0.4443 | — | — | 0.4392 | 0.4209 | -4.185 (61.766%) |
| TMR | 3 | 0.7250 | 0.7107 | 0.6767 | 0.7166 | 0.6828 | -8.245 (38.705%) |

## OMR vs TMR legacy / 6 / 6 / all clients / overall

Rows per codec: 210. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR legacy | 1 | 0.6527 | — | — | — | 0.4469 | -3.403 (67.582%) |
| OMR legacy | 2 | 0.7544 | — | — | — | 0.5350 | -4.313 (60.860%) |
| OMR legacy | 3 | 0.9986 | — | 0.7588 | — | 0.7643 | -7.728 (41.078%) |
| TMR legacy | 1 | 0.5149 | — | — | 0.3691 | 0.3732 | -2.262 (77.073%) |
| TMR legacy | 2 | 0.6210 | — | — | 0.4608 | 0.4564 | -3.705 (65.275%) |
| TMR legacy | 3 | 0.8864 | 0.7036 | 0.6868 | 0.7096 | 0.6934 | -7.450 (42.415%) |

## OMR vs TMR legacy / 6 / 6 / all clients / trained

Rows per codec: 30. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR legacy | 1 | 0.6416 | — | — | — | 0.3697 | -3.625 (65.878%) |
| OMR legacy | 2 | 0.7244 | — | — | — | 0.4353 | -4.465 (59.805%) |
| OMR legacy | 3 | 0.9615 | — | 0.6533 | — | 0.6592 | -7.823 (40.632%) |
| TMR legacy | 1 | 0.5136 | — | — | 0.3570 | 0.3503 | -2.822 (72.261%) |
| TMR legacy | 2 | 0.6177 | — | — | 0.4447 | 0.4279 | -4.003 (63.076%) |
| TMR legacy | 3 | 0.8775 | 0.6833 | 0.6528 | 0.6893 | 0.6587 | -7.636 (41.514%) |

## OMR vs TMR legacy / 6 / 6 / all clients / frozen

Rows per codec: 180. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR legacy | 1 | 0.6546 | — | — | — | 0.4598 | -3.367 (67.866%) |
| OMR legacy | 2 | 0.7594 | — | — | — | 0.5516 | -4.288 (61.036%) |
| OMR legacy | 3 | 1.0048 | — | 0.7764 | — | 0.7819 | -7.712 (41.153%) |
| TMR legacy | 1 | 0.5151 | — | — | 0.3711 | 0.3770 | -2.172 (77.875%) |
| TMR legacy | 2 | 0.6216 | — | — | 0.4635 | 0.4611 | -3.656 (65.641%) |
| TMR legacy | 3 | 0.8879 | 0.7070 | 0.6925 | 0.7130 | 0.6992 | -7.419 (42.565%) |

## OMR vs TMR legacy / 6 / 6 / exclude outliers / overall

Rows per codec: 189. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR legacy | 1 | 0.6253 | — | — | — | 0.3741 | -3.386 (67.716%) |
| OMR legacy | 2 | 0.7037 | — | — | — | 0.4444 | -4.302 (60.936%) |
| OMR legacy | 3 | 0.9492 | — | 0.6682 | — | 0.6740 | -7.720 (41.115%) |
| TMR legacy | 1 | 0.4894 | — | — | 0.3582 | 0.3541 | -2.236 (77.302%) |
| TMR legacy | 2 | 0.5945 | — | — | 0.4487 | 0.4360 | -3.679 (65.469%) |
| TMR legacy | 3 | 0.8591 | 0.6896 | 0.6651 | 0.6955 | 0.6710 | -7.383 (42.743%) |

## OMR vs TMR legacy / 6 / 6 / exclude outliers / trained

Rows per codec: 27. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR legacy | 1 | 0.6214 | — | — | — | 0.3682 | -3.614 (65.962%) |
| OMR legacy | 2 | 0.6949 | — | — | — | 0.4338 | -4.460 (59.845%) |
| OMR legacy | 3 | 0.9329 | — | 0.6522 | — | 0.6581 | -7.817 (40.657%) |
| TMR legacy | 1 | 0.4877 | — | — | 0.3563 | 0.3495 | -2.830 (72.198%) |
| TMR legacy | 2 | 0.5906 | — | — | 0.4427 | 0.4259 | -3.991 (63.163%) |
| TMR legacy | 3 | 0.8497 | 0.6794 | 0.6488 | 0.6853 | 0.6548 | -7.573 (41.816%) |

## OMR vs TMR legacy / 6 / 6 / exclude outliers / frozen

Rows per codec: 162. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OMR legacy | 1 | 0.6259 | — | — | — | 0.3751 | -3.349 (68.009%) |
| OMR legacy | 2 | 0.7051 | — | — | — | 0.4462 | -4.277 (61.118%) |
| OMR legacy | 3 | 0.9520 | — | 0.6708 | — | 0.6766 | -7.704 (41.192%) |
| TMR legacy | 1 | 0.4897 | — | — | 0.3586 | 0.3549 | -2.141 (78.153%) |
| TMR legacy | 2 | 0.5952 | — | — | 0.4497 | 0.4377 | -3.628 (65.853%) |
| TMR legacy | 3 | 0.8606 | 0.6913 | 0.6678 | 0.6972 | 0.6737 | -7.351 (42.898%) |

## OR vs OMR frozen different placements / 5 / 7 / all clients / overall

Rows per codec: 210. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4517 | — | — | — | 0.3693 | -3.310 (68.312%) |
| OR | 2 | 0.5482 | — | — | — | 0.4505 | -4.354 (60.573%) |
| OR | 3 | 0.7443 | — | 0.6202 | — | 0.6261 | -6.869 (45.349%) |
| OMR | 1 | 0.4472 | — | — | — | 0.3945 | -3.398 (67.624%) |
| OMR | 2 | 0.5613 | — | — | — | 0.4692 | -4.320 (60.815%) |
| OMR | 3 | 0.8150 | — | 0.6911 | — | 0.6968 | -7.599 (41.691%) |

## OR vs OMR frozen different placements / 5 / 7 / all clients / frozen

Rows per codec: 210. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4517 | — | — | — | 0.3693 | -3.310 (68.312%) |
| OR | 2 | 0.5482 | — | — | — | 0.4505 | -4.354 (60.573%) |
| OR | 3 | 0.7443 | — | 0.6202 | — | 0.6261 | -6.869 (45.349%) |
| OMR | 1 | 0.4472 | — | — | — | 0.3945 | -3.398 (67.624%) |
| OMR | 2 | 0.5613 | — | — | — | 0.4692 | -4.320 (60.815%) |
| OMR | 3 | 0.8150 | — | 0.6911 | — | 0.6968 | -7.599 (41.691%) |

## OR vs OMR frozen different placements / 5 / 7 / exclude outliers / overall

Rows per codec: 203. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4544 | — | — | — | 0.3692 | -3.292 (68.455%) |
| OR | 2 | 0.5505 | — | — | — | 0.4505 | -4.345 (60.637%) |
| OR | 3 | 0.7441 | — | 0.6189 | — | 0.6248 | -6.843 (45.485%) |
| OMR | 1 | 0.4292 | — | — | — | 0.3621 | -3.384 (67.735%) |
| OMR | 2 | 0.5449 | — | — | — | 0.4374 | -4.311 (60.877%) |
| OMR | 3 | 0.7979 | — | 0.6586 | — | 0.6645 | -7.585 (41.759%) |

## OR vs OMR frozen different placements / 5 / 7 / exclude outliers / frozen

Rows per codec: 203. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.4544 | — | — | — | 0.3692 | -3.292 (68.455%) |
| OR | 2 | 0.5505 | — | — | — | 0.4505 | -4.345 (60.637%) |
| OR | 3 | 0.7441 | — | 0.6189 | — | 0.6248 | -6.843 (45.485%) |
| OMR | 1 | 0.4292 | — | — | — | 0.3621 | -3.384 (67.735%) |
| OMR | 2 | 0.5449 | — | — | — | 0.4374 | -4.311 (60.877%) |
| OMR | 3 | 0.7979 | — | 0.6586 | — | 0.6645 | -7.585 (41.759%) |

## OR vs TR / pooled / pooled / all clients / frozen

Rows per codec: 810. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5568 | — | — | — | 0.4065 | -3.278 (68.568%) |
| OR | 2 | 0.6480 | — | — | — | 0.4837 | -4.278 (61.109%) |
| OR | 3 | 0.8826 | — | 0.6810 | — | 0.6870 | -7.185 (43.728%) |
| TR | 1 | 0.6822 | — | — | 0.4067 | 0.3937 | -2.166 (77.931%) |
| TR | 2 | 0.7957 | — | — | 0.4944 | 0.4745 | -3.642 (65.749%) |
| TR | 3 | 1.0770 | 0.7310 | 0.7026 | 0.7367 | 0.7083 | -7.239 (43.455%) |

## OR vs TR / pooled / pooled / all clients / overall

Rows per codec: 900. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5553 | — | — | — | 0.4022 | -3.321 (68.226%) |
| OR | 2 | 0.6451 | — | — | — | 0.4784 | -4.307 (60.905%) |
| OR | 3 | 0.8765 | — | 0.6745 | — | 0.6805 | -7.203 (43.638%) |
| TR | 1 | 0.6819 | — | — | 0.4029 | 0.3907 | -2.236 (77.301%) |
| TR | 2 | 0.7943 | — | — | 0.4898 | 0.4705 | -3.681 (65.455%) |
| TR | 3 | 1.0742 | 0.7256 | 0.6974 | 0.7313 | 0.7031 | -7.262 (43.341%) |

## OR vs TR / pooled / pooled / all clients / trained

Rows per codec: 90. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5422 | — | — | — | 0.3637 | -3.722 (65.150%) |
| OR | 2 | 0.6183 | — | — | — | 0.4307 | -4.573 (59.069%) |
| OR | 3 | 0.8221 | — | 0.6158 | — | 0.6217 | -7.364 (42.833%) |
| TR | 1 | 0.6791 | — | — | 0.3684 | 0.3632 | -2.898 (71.634%) |
| TR | 2 | 0.7810 | — | — | 0.4491 | 0.4346 | -4.039 (62.816%) |
| TR | 3 | 1.0484 | 0.6768 | 0.6500 | 0.6828 | 0.6559 | -7.472 (42.307%) |

## OR vs TR / pooled / pooled / exclude outliers / frozen

Rows per codec: 756. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5392 | — | — | — | 0.3739 | -3.279 (68.553%) |
| OR | 2 | 0.6310 | — | — | — | 0.4514 | -4.282 (61.078%) |
| OR | 3 | 0.8632 | — | 0.6469 | — | 0.6527 | -7.154 (43.885%) |
| TR | 1 | 0.6672 | — | — | 0.3722 | 0.3691 | -2.155 (78.030%) |
| TR | 2 | 0.7819 | — | — | 0.4602 | 0.4502 | -3.634 (65.813%) |
| TR | 3 | 1.0666 | 0.6980 | 0.6793 | 0.7039 | 0.6852 | -7.246 (43.423%) |

## OR vs TR / pooled / pooled / exclude outliers / overall

Rows per codec: 840. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5378 | — | — | — | 0.3728 | -3.322 (68.217%) |
| OR | 2 | 0.6281 | — | — | — | 0.4492 | -4.311 (60.878%) |
| OR | 3 | 0.8573 | — | 0.6435 | — | 0.6493 | -7.171 (43.796%) |
| TR | 1 | 0.6669 | — | — | 0.3717 | 0.3684 | -2.226 (77.390%) |
| TR | 2 | 0.7804 | — | — | 0.4590 | 0.4485 | -3.673 (65.513%) |
| TR | 3 | 1.0637 | 0.6960 | 0.6765 | 0.7019 | 0.6823 | -7.269 (43.305%) |

## OR vs TR / pooled / pooled / exclude outliers / trained

Rows per codec: 84. Phases with no matched coverage are not reported.

| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---:|---:|---:|---:|---:|---:|---|
| OR | 1 | 0.5256 | — | — | — | 0.3625 | -3.717 (65.187%) |
| OR | 2 | 0.6021 | — | — | — | 0.4297 | -4.572 (59.075%) |
| OR | 3 | 0.8039 | — | 0.6130 | — | 0.6188 | -7.332 (42.991%) |
| TR | 1 | 0.6644 | — | — | 0.3675 | 0.3623 | -2.898 (71.632%) |
| TR | 2 | 0.7674 | — | — | 0.4485 | 0.4339 | -4.039 (62.814%) |
| TR | 3 | 1.0381 | 0.6776 | 0.6507 | 0.6835 | 0.6566 | -7.485 (42.243%) |
