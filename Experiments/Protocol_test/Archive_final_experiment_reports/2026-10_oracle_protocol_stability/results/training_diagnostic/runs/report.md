# Controlled sampled-trace diagnostic (incomplete until complete.json)

Matched completed client/seed jobs: [(0, 43), (0, 44), (0, 45), (5, 43), (5, 44), (5, 45), (11, 43), (11, 44), (11, 45)].

Same compressed raw SI history; smaller optimizer-update budget than full-vector runs.
2048-lane rANS on short samples is flush-dominated: use priors to diagnose training, not sample rANS savings to conclude protocol superiority.
Preserve rows report the original joint prior, not a fine-tuned prior. Alternate-SI rates are unavailable there.

| Phase | Model | Prior fit | Stage | Pre-fit | Temporal prior | Full prior | Temporal rANS | Full rANS | WMAPE dB (raw %) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| future | O | current | 1 | 2.2237 | 0.6709 | 3.2882 | 3.4232 | 5.1965 | -1.842 (80.895%) |
| future | O | current | 2 | 3.6061 | 1.2292 | 5.6183 | 3.6933 | 6.8183 | -4.009 (63.029%) |
| future | O | current | 3 | 4.8198 | 1.9337 | 8.0415 | 4.3058 | 8.5606 | -5.290 (54.388%) |
| future | O | multi | 1 | 2.2237 | 0.6308 | 3.0091 | 3.4135 | 5.0179 | -1.842 (80.895%) |
| future | O | multi | 2 | 3.6061 | 0.9868 | 5.0438 | 3.4816 | 6.4739 | -4.009 (63.029%) |
| future | O | multi | 3 | 4.8198 | 1.3084 | 7.0711 | 3.7055 | 7.9931 | -5.290 (54.388%) |
| future | O | preserve | 1 | 2.2237 | — | 2.2237 | — | 4.4118 | -1.842 (80.895%) |
| future | O | preserve | 2 | 3.6061 | — | 3.6061 | — | 5.5018 | -4.009 (63.029%) |
| future | O | preserve | 3 | 4.8198 | — | 4.8198 | — | 6.5382 | -5.290 (54.388%) |
| future | O_from_T | current | 1 | 0.4688 | 0.3581 | 1.0900 | 3.2140 | 3.7002 | -2.248 (77.194%) |
| future | O_from_T | current | 2 | 0.8317 | 0.7800 | 2.4816 | 3.3703 | 4.6755 | -4.927 (56.706%) |
| future | O_from_T | current | 3 | 1.1827 | 1.4434 | 4.0723 | 3.8598 | 5.8068 | -7.400 (42.658%) |
| future | O_from_T | multi | 1 | 0.4688 | 0.3289 | 0.6164 | 3.2116 | 3.3578 | -2.248 (77.194%) |
| future | O_from_T | multi | 2 | 0.8317 | 0.5766 | 1.5648 | 3.2257 | 4.0149 | -4.927 (56.706%) |
| future | O_from_T | multi | 3 | 1.1827 | 0.8651 | 2.6359 | 3.3534 | 4.8526 | -7.400 (42.658%) |
| future | O_from_T | preserve | 1 | 0.4688 | — | 0.4688 | — | 3.2180 | -2.248 (77.194%) |
| future | O_from_T | preserve | 2 | 0.8317 | — | 0.8317 | — | 3.3911 | -4.927 (56.706%) |
| future | O_from_T | preserve | 3 | 1.1827 | — | 1.1827 | — | 3.6513 | -7.400 (42.658%) |
| future | T | current | 1 | 0.4448 | 0.3280 | 2.3106 | 3.2114 | 4.5536 | -2.107 (78.462%) |
| future | T | current | 2 | 0.6407 | 0.5418 | 3.7579 | 3.2253 | 5.4880 | -4.297 (60.974%) |
| future | T | current | 3 | 0.8744 | 0.8180 | 5.6723 | 3.3327 | 6.7663 | -7.234 (43.481%) |
| future | T | multi | 1 | 0.4448 | 0.3235 | 2.1825 | 3.2113 | 4.4701 | -2.107 (78.462%) |
| future | T | multi | 2 | 0.6407 | 0.5217 | 3.5831 | 3.2191 | 5.3769 | -4.297 (60.974%) |
| future | T | multi | 3 | 0.8744 | 0.7601 | 5.4810 | 3.2816 | 6.6414 | -7.234 (43.481%) |
| future | T | preserve | 1 | 0.4448 | 0.4448 | — | 3.2113 | — | -2.107 (78.462%) |
| future | T | preserve | 2 | 0.6407 | 0.6407 | — | 3.2313 | — | -4.297 (60.974%) |
| future | T | preserve | 3 | 0.8744 | 0.8744 | — | 3.3265 | — | -7.234 (43.481%) |
| historical_validation | O | current | 1 | 2.1611 | 0.4403 | 3.0746 | 16.0579 | 17.7114 | -2.013 (79.316%) |
| historical_validation | O | current | 2 | 3.5211 | 1.0103 | 5.3797 | 16.1313 | 19.0458 | -4.138 (62.105%) |
| historical_validation | O | current | 3 | 4.7493 | 1.7271 | 7.8299 | 16.3887 | 20.5842 | -5.434 (53.492%) |
| historical_validation | O | multi | 1 | 2.1611 | 0.3953 | 2.7958 | 16.0566 | 17.4824 | -2.013 (79.316%) |
| historical_validation | O | multi | 2 | 3.5211 | 0.7492 | 4.8028 | 16.0575 | 18.6168 | -4.138 (62.105%) |
| historical_validation | O | multi | 3 | 4.7493 | 1.0700 | 6.8488 | 16.0734 | 19.9232 | -5.434 (53.492%) |
| historical_validation | O | preserve | 1 | 2.1611 | — | 2.1611 | — | 16.9232 | -2.013 (79.316%) |
| historical_validation | O | preserve | 2 | 3.5211 | — | 3.5211 | — | 17.6070 | -4.138 (62.105%) |
| historical_validation | O | preserve | 3 | 4.7493 | — | 4.7493 | — | 18.3559 | -5.434 (53.492%) |
| historical_validation | O_from_T | current | 1 | 0.4726 | 0.3693 | 1.0703 | 16.0571 | 16.3961 | -2.463 (75.307%) |
| historical_validation | O_from_T | current | 2 | 0.8380 | 0.7943 | 2.4627 | 16.1241 | 17.1704 | -5.007 (56.190%) |
| historical_validation | O_from_T | current | 3 | 1.2002 | 1.4640 | 4.1073 | 16.3707 | 18.1675 | -7.450 (42.413%) |
| historical_validation | O_from_T | multi | 1 | 0.4726 | 0.3368 | 0.6081 | 16.0566 | 16.1326 | -2.463 (75.307%) |
| historical_validation | O_from_T | multi | 2 | 0.8380 | 0.5870 | 1.5373 | 16.0573 | 16.5412 | -5.007 (56.190%) |
| historical_validation | O_from_T | multi | 3 | 1.2002 | 0.8813 | 2.6471 | 16.0731 | 17.1311 | -7.450 (42.413%) |
| historical_validation | O_from_T | preserve | 1 | 0.4726 | — | 0.4726 | — | 16.0579 | -2.463 (75.307%) |
| historical_validation | O_from_T | preserve | 2 | 0.8380 | — | 0.8380 | — | 16.0833 | -5.007 (56.190%) |
| historical_validation | O_from_T | preserve | 3 | 1.2002 | — | 1.2002 | — | 16.1534 | -7.450 (42.413%) |
| historical_validation | T | current | 1 | 0.4524 | 0.3360 | 2.3381 | 16.0566 | 17.2936 | -2.377 (76.060%) |
| historical_validation | T | current | 2 | 0.6551 | 0.5572 | 3.8023 | 16.0579 | 18.1523 | -4.428 (60.061%) |
| historical_validation | T | current | 3 | 0.8922 | 0.8417 | 5.7184 | 16.0740 | 19.2995 | -7.342 (42.943%) |
| historical_validation | T | multi | 1 | 0.4524 | 0.3300 | 2.2004 | 16.0566 | 17.1910 | -2.377 (76.060%) |
| historical_validation | T | multi | 2 | 0.6551 | 0.5349 | 3.6420 | 16.0569 | 18.0302 | -4.428 (60.061%) |
| historical_validation | T | multi | 3 | 0.8922 | 0.7774 | 5.5839 | 16.0592 | 19.2049 | -7.342 (42.943%) |
| historical_validation | T | preserve | 1 | 0.4524 | 0.4524 | — | 16.0566 | — | -2.377 (76.060%) |
| historical_validation | T | preserve | 2 | 0.6551 | 0.6551 | — | 16.0569 | — | -4.428 (60.061%) |
| historical_validation | T | preserve | 3 | 0.8922 | 0.8922 | — | 16.0579 | — | -7.342 (42.943%) |

Equal-distortion gaps vs T: positive means worse. All available rate-source gaps are in summary_comparison_gaps.csv.

| Phase | Model | Prior fit | Source | Gap bits/float | Gap % |
|---|---|---|---|---:|---:|
| future | O | current | full_prior | +2.13380 | +63.489% |
| future | O | current | full_rans | +1.50952 | +28.841% |
| future | O | current | native_prior | +5.01189 | +1038.095% |
| future | O | current | native_rans | +3.51726 | +109.021% |
| future | O_from_T | current | full_prior | -1.59268 | -39.387% |
| future | O_from_T | current | full_rans | -1.02735 | -18.071% |
| future | O_from_T | current | native_prior | +1.86892 | +321.093% |
| future | O_from_T | current | native_rans | +1.40335 | +43.123% |
| future | O | multi | full_prior | +1.71670 | +53.625% |
| future | O | multi | full_rans | +1.26612 | +24.673% |
| future | O | multi | native_prior | +4.45191 | +955.111% |
| future | O | multi | native_rans | +3.17799 | +98.705% |
| future | O_from_T | multi | full_prior | -2.33185 | -60.208% |
| future | O_from_T | multi | full_rans | -1.55593 | -27.909% |
| future | O_from_T | multi | native_prior | +0.98726 | +178.247% |
| future | O_from_T | multi | native_rans | +0.78314 | +24.201% |
| future | O | preserve | native_prior | +2.90214 | +495.589% |
| future | O | preserve | native_rans | +2.19496 | +67.967% |
| future | O_from_T | preserve | native_prior | +0.13810 | +20.554% |
| future | O_from_T | preserve | native_rans | +0.14522 | +4.461% |
| historical_validation | O | current | full_prior | +1.92934 | +56.413% |
| historical_validation | O | current | full_rans | +1.12422 | +6.270% |
| historical_validation | O | current | native_prior | +4.85018 | +971.641% |
| historical_validation | O | current | native_rans | +2.99526 | +18.652% |
| historical_validation | O_from_T | current | full_prior | -1.61618 | -39.495% |
| historical_validation | O_from_T | current | full_rans | -1.13266 | -6.180% |
| historical_validation | O_from_T | current | native_prior | +1.87643 | +312.997% |
| historical_validation | O_from_T | current | native_rans | +1.13414 | +7.061% |
| historical_validation | O | multi | full_prior | +1.48911 | +45.556% |
| historical_validation | O | multi | full_rans | +0.80779 | +4.534% |
| historical_validation | O | multi | native_prior | +4.27796 | +891.403% |
| historical_validation | O | multi | native_rans | +2.56626 | +15.982% |
| historical_validation | O_from_T | multi | full_prior | -2.39495 | -60.726% |
| historical_validation | O_from_T | multi | full_rans | -1.64947 | -9.053% |
| historical_validation | O_from_T | multi | native_prior | +0.98115 | +172.809% |
| historical_validation | O_from_T | multi | native_rans | +0.51211 | +3.189% |
| historical_validation | O | preserve | native_prior | +2.85650 | +475.597% |
| historical_validation | O | preserve | native_rans | +1.54582 | +9.627% |
| historical_validation | O_from_T | preserve | native_prior | +0.13873 | +20.198% |
| historical_validation | O_from_T | preserve | native_rans | +0.03561 | +0.222% |
