# Full-vector confirmation (incomplete until complete.json)

Production full-vector training and update budget; common decoded TMR warm-pool history.
Round8 is chronological validation; rounds9–16 are untouched frozen evaluation.
Joint priors and exact T-in-O candidates remain explicit alternatives. No every-round fine-tuning.
Rates are cumulative bits/float; WMAPE is averaged in linear space before conversion to dB.

| Phase | Model | Prior policy | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| future | O_cold | fine | 1 | 0.5646 | — | 0.3755 | — | 0.3809 | -3.656 (65.645%) |
| future | O_cold | fine | 2 | 0.6556 | — | 0.4518 | — | 0.4572 | -4.505 (59.529%) |
| future | O_cold | fine | 3 | 0.9096 | — | 0.6908 | — | 0.6967 | -8.041 (39.622%) |
| future | O_cold | joint | 1 | 0.5646 | — | 0.5646 | — | 0.5635 | -3.656 (65.645%) |
| future | O_cold | joint | 2 | 0.6556 | — | 0.6556 | — | 0.6543 | -4.505 (59.529%) |
| future | O_cold | joint | 3 | 0.9096 | — | 0.9096 | — | 0.9088 | -8.041 (39.622%) |
| future | O_from_T | fine | 1 | 0.5362 | — | 0.6220 | — | 0.6280 | -3.462 (67.128%) |
| future | O_from_T | fine | 2 | 0.6448 | — | 0.7123 | — | 0.7184 | -4.592 (58.940%) |
| future | O_from_T | fine | 3 | 0.9077 | — | 0.9344 | — | 0.9411 | -7.801 (40.731%) |
| future | O_from_T | joint | 1 | 0.5362 | — | 0.5362 | — | 0.5450 | -3.462 (67.128%) |
| future | O_from_T | joint | 2 | 0.6448 | — | 0.6448 | — | 0.6524 | -4.592 (58.940%) |
| future | O_from_T | joint | 3 | 0.9077 | — | 0.9077 | — | 0.9068 | -7.801 (40.731%) |
| future | T | fine | 1 | 0.6124 | 0.3637 | 0.3610 | 0.3690 | 0.3662 | -2.652 (73.687%) |
| future | T | fine | 2 | 0.7529 | 0.4809 | 0.4690 | 0.4865 | 0.4744 | -4.235 (61.413%) |
| future | T | fine | 3 | 0.9721 | 0.6919 | 0.6725 | 0.6978 | 0.6784 | -7.281 (43.246%) |
| future | T | joint | 1 | 0.6124 | 0.6124 | 0.3610 | 0.6176 | 0.3662 | -2.652 (73.687%) |
| future | T | joint | 2 | 0.7529 | 0.7529 | 0.4690 | 0.7582 | 0.4744 | -4.235 (61.413%) |
| future | T | joint | 3 | 0.9721 | 0.9721 | 0.6725 | 0.9762 | 0.6784 | -7.281 (43.246%) |
| future | T_in_O | fine | 1 | 0.6124 | — | 0.3612 | — | 0.3664 | -2.647 (73.729%) |
| future | T_in_O | fine | 2 | 0.7529 | — | 0.4695 | — | 0.4750 | -4.231 (61.439%) |
| future | T_in_O | fine | 3 | 0.9721 | — | 0.6726 | — | 0.6787 | -7.278 (43.264%) |
| future | T_in_O | joint | 1 | 0.6124 | — | 0.6124 | — | 0.6176 | -2.647 (73.729%) |
| future | T_in_O | joint | 2 | 0.7529 | — | 0.7529 | — | 0.7581 | -4.231 (61.439%) |
| future | T_in_O | joint | 3 | 0.9721 | — | 0.9721 | — | 0.9761 | -7.278 (43.264%) |
| future | T_in_O | temporal_fine | 1 | 0.6124 | — | 0.3637 | — | 0.3690 | -2.647 (73.729%) |
| future | T_in_O | temporal_fine | 2 | 0.7529 | — | 0.4809 | — | 0.4865 | -4.231 (61.439%) |
| future | T_in_O | temporal_fine | 3 | 0.9721 | — | 0.6919 | — | 0.6978 | -7.278 (43.264%) |
| validation | O_cold | fine | 1 | 0.5653 | — | 0.3754 | — | 0.3807 | -3.683 (65.441%) |
| validation | O_cold | fine | 2 | 0.6494 | — | 0.4488 | — | 0.4541 | -4.545 (59.256%) |
| validation | O_cold | fine | 3 | 0.8985 | — | 0.6874 | — | 0.6936 | -8.122 (39.254%) |
| validation | O_cold | joint | 1 | 0.5653 | — | 0.5653 | — | 0.5641 | -3.683 (65.441%) |
| validation | O_cold | joint | 2 | 0.6494 | — | 0.6494 | — | 0.6480 | -4.545 (59.256%) |
| validation | O_cold | joint | 3 | 0.8985 | — | 0.8985 | — | 0.8979 | -8.122 (39.254%) |
| validation | O_from_T | fine | 1 | 0.5328 | — | 0.3556 | — | 0.3607 | -3.520 (66.677%) |
| validation | O_from_T | fine | 2 | 0.6360 | — | 0.4432 | — | 0.4486 | -4.597 (58.904%) |
| validation | O_from_T | fine | 3 | 0.8959 | — | 0.6648 | — | 0.6704 | -7.872 (40.403%) |
| validation | O_from_T | joint | 1 | 0.5328 | — | 0.5328 | — | 0.5405 | -3.520 (66.677%) |
| validation | O_from_T | joint | 2 | 0.6360 | — | 0.6360 | — | 0.6430 | -4.597 (58.904%) |
| validation | O_from_T | joint | 3 | 0.8959 | — | 0.8959 | — | 0.8939 | -7.872 (40.403%) |
| validation | T | fine | 1 | 0.6142 | 0.3635 | 0.3612 | 0.3686 | 0.3662 | -2.860 (71.947%) |
| validation | T | fine | 2 | 0.7529 | 0.4783 | 0.4667 | 0.4838 | 0.4722 | -4.415 (60.149%) |
| validation | T | fine | 3 | 0.9699 | 0.6892 | 0.6694 | 0.6950 | 0.6749 | -7.501 (42.165%) |
| validation | T | joint | 1 | 0.6142 | 0.6142 | 0.3612 | 0.6191 | 0.3662 | -2.860 (71.947%) |
| validation | T | joint | 2 | 0.7529 | 0.7529 | 0.4667 | 0.7579 | 0.4722 | -4.415 (60.149%) |
| validation | T | joint | 3 | 0.9699 | 0.9699 | 0.6694 | 0.9747 | 0.6749 | -7.501 (42.165%) |
| validation | T_in_O | fine | 1 | 0.6142 | — | 0.3610 | — | 0.3662 | -2.855 (71.983%) |
| validation | T_in_O | fine | 2 | 0.7529 | — | 0.4666 | — | 0.4721 | -4.412 (60.174%) |
| validation | T_in_O | fine | 3 | 0.9699 | — | 0.6687 | — | 0.6744 | -7.498 (42.179%) |
| validation | T_in_O | joint | 1 | 0.6142 | — | 0.6142 | — | 0.6191 | -2.855 (71.983%) |
| validation | T_in_O | joint | 2 | 0.7529 | — | 0.7529 | — | 0.7579 | -4.412 (60.174%) |
| validation | T_in_O | joint | 3 | 0.9699 | — | 0.9699 | — | 0.9747 | -7.498 (42.179%) |
| validation | T_in_O | temporal_fine | 1 | 0.6142 | — | 0.3635 | — | 0.3686 | -2.855 (71.983%) |
| validation | T_in_O | temporal_fine | 2 | 0.7529 | — | 0.4783 | — | 0.4838 | -4.412 (60.174%) |
| validation | T_in_O | temporal_fine | 3 | 0.9699 | — | 0.6892 | — | 0.6950 | -7.498 (42.179%) |

Equal-distortion gaps relative to T's fine-tuned temporal transport; negative favors the candidate.
Selected_O_vs_selected_T instead compares both chronological selections. Full-SI T checks are listed separately above.

| Phase | Model | Prior policy | Source | Gap bits/float | Gap % |
|---|---|---|---|---:|---:|
| future | O_cold | fine | pre_rate | -0.11964 | -14.23% |
| future | O_cold | fine | prior_rate | -0.05147 | -9.09% |
| future | O_cold | fine | rans_rate | -0.05159 | -9.02% |
| future | O_cold | joint | pre_rate | -0.11964 | -14.23% |
| future | O_cold | joint | prior_rate | +0.15514 | +27.40% |
| future | O_cold | joint | rans_rate | +0.14832 | +25.94% |
| future | O_from_T | fine | pre_rate | -0.12708 | -15.25% |
| future | O_from_T | fine | prior_rate | +0.20523 | +36.70% |
| future | O_from_T | fine | rans_rate | +0.20577 | +36.42% |
| future | O_from_T | joint | pre_rate | -0.12708 | -15.25% |
| future | O_from_T | joint | prior_rate | +0.14706 | +26.29% |
| future | O_from_T | joint | rans_rate | +0.14658 | +25.94% |
| future | T | fine | pre_rate | +0.00000 | +0.00% |
| future | T | fine | prior_rate | +0.00000 | +0.00% |
| future | T | fine | rans_rate | +0.00000 | +0.00% |
| future | T | full_check | pre_rate | +0.00000 | +0.00% |
| future | T | full_check | prior_rate | -0.01283 | -2.42% |
| future | T | full_check | rans_rate | -0.01295 | -2.42% |
| future | T | joint | pre_rate | +0.00000 | +0.00% |
| future | T | joint | prior_rate | +0.27074 | +51.05% |
| future | T | joint | rans_rate | +0.26990 | +50.36% |
| future | T_in_O | fine | pre_rate | +0.00028 | +0.03% |
| future | T_in_O | fine | prior_rate | -0.01221 | -2.30% |
| future | T_in_O | fine | rans_rate | -0.01222 | -2.28% |
| future | T_in_O | joint | pre_rate | +0.00028 | +0.03% |
| future | T_in_O | joint | prior_rate | +0.27101 | +51.12% |
| future | T_in_O | joint | rans_rate | +0.27017 | +50.42% |
| future | T_in_O | temporal_fine | pre_rate | +0.00028 | +0.03% |
| future | T_in_O | temporal_fine | prior_rate | +0.00028 | +0.05% |
| future | T_in_O | temporal_fine | rans_rate | +0.00028 | +0.05% |
| validation | O_cold | fine | pre_rate | -0.11480 | -13.76% |
| validation | O_cold | fine | prior_rate | -0.04147 | -7.43% |
| validation | O_cold | fine | rans_rate | -0.04152 | -7.36% |
| validation | O_cold | joint | pre_rate | -0.11480 | -13.76% |
| validation | O_cold | joint | prior_rate | +0.16125 | +28.88% |
| validation | O_cold | joint | rans_rate | +0.15451 | +27.40% |
| validation | O_from_T | fine | pre_rate | -0.12193 | -14.72% |
| validation | O_from_T | fine | prior_rate | -0.04955 | -8.97% |
| validation | O_from_T | fine | rans_rate | -0.04974 | -8.91% |
| validation | O_from_T | joint | pre_rate | -0.12193 | -14.72% |
| validation | O_from_T | joint | prior_rate | +0.15355 | +27.79% |
| validation | O_from_T | joint | rans_rate | +0.15210 | +27.25% |
| validation | T | fine | pre_rate | +0.00000 | +0.00% |
| validation | T | fine | prior_rate | +0.00000 | +0.00% |
| validation | T | fine | rans_rate | +0.00000 | +0.00% |
| validation | T | full_check | pre_rate | +0.00000 | +0.00% |
| validation | T | full_check | prior_rate | -0.01271 | -2.40% |
| validation | T | full_check | rans_rate | -0.01289 | -2.41% |
| validation | T | joint | pre_rate | +0.00000 | +0.00% |
| validation | T | joint | prior_rate | +0.27268 | +51.53% |
| validation | T | joint | rans_rate | +0.27202 | +50.87% |
| validation | T_in_O | fine | pre_rate | +0.00026 | +0.03% |
| validation | T_in_O | fine | prior_rate | -0.01283 | -2.42% |
| validation | T_in_O | fine | rans_rate | -0.01287 | -2.41% |
| validation | T_in_O | joint | pre_rate | +0.00026 | +0.03% |
| validation | T_in_O | joint | prior_rate | +0.27293 | +51.59% |
| validation | T_in_O | joint | rans_rate | +0.27227 | +50.93% |
| validation | T_in_O | temporal_fine | pre_rate | +0.00026 | +0.03% |
| validation | T_in_O | temporal_fine | prior_rate | +0.00025 | +0.05% |
| validation | T_in_O | temporal_fine | rans_rate | +0.00025 | +0.05% |
| future | selected_O_vs_selected_T | chronological_validation | pre_rate | -0.20979 | -25.08% |
| future | selected_O_vs_selected_T | chronological_validation | prior_rate | -0.05494 | -9.77% |
| future | selected_O_vs_selected_T | chronological_validation | rans_rate | -0.05520 | -9.72% |
| validation | selected_O_vs_selected_T | chronological_validation | pre_rate | -0.20222 | -24.36% |
| validation | selected_O_vs_selected_T | chronological_validation | prior_rate | -0.04285 | -7.73% |
| validation | selected_O_vs_selected_T | chronological_validation | rans_rate | -0.04333 | -7.73% |
