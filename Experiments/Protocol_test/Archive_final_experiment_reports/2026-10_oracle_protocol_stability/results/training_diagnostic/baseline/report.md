# Preserved OMR/TMR baseline (incomplete experiments)

Recorded rows: OMR 300/900; TMR 225/900.
Matched complete rounds: [6, 7, 8, 9, 10, 11, 12]; equal client weights, no exclusions.

Rates are cumulative bits/float. Fine-tuned prior stage breakdowns were not recorded; only final-stage totals are available.
OMR has no temporal-prior/rANS measurement in these original records; — means unavailable, not zero.

| Phase | Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| all | OMR | 1 | 0.6527 | — | — | — | 0.4469 | -3.403 (67.582%) |
| all | OMR | 2 | 0.7544 | — | — | — | 0.5350 | -4.313 (60.860%) |
| all | OMR | 3 | 0.9986 | — | 0.7588 | — | 0.7643 | -7.728 (41.078%) |
| trained | OMR | 1 | 0.6416 | — | — | — | 0.3697 | -3.625 (65.878%) |
| trained | OMR | 2 | 0.7244 | — | — | — | 0.4353 | -4.465 (59.805%) |
| trained | OMR | 3 | 0.9615 | — | 0.6533 | — | 0.6592 | -7.823 (40.632%) |
| frozen | OMR | 1 | 0.6546 | — | — | — | 0.4598 | -3.367 (67.866%) |
| frozen | OMR | 2 | 0.7594 | — | — | — | 0.5516 | -4.288 (61.036%) |
| frozen | OMR | 3 | 1.0048 | — | 0.7764 | — | 0.7819 | -7.712 (41.153%) |
| all | TMR | 1 | 0.5149 | — | — | 0.3691 | 0.3732 | -2.262 (77.073%) |
| all | TMR | 2 | 0.6210 | — | — | 0.4608 | 0.4564 | -3.705 (65.275%) |
| all | TMR | 3 | 0.8864 | 0.7036 | 0.6868 | 0.7096 | 0.6934 | -7.450 (42.415%) |
| trained | TMR | 1 | 0.5136 | — | — | 0.3570 | 0.3503 | -2.822 (72.261%) |
| trained | TMR | 2 | 0.6177 | — | — | 0.4447 | 0.4279 | -4.003 (63.076%) |
| trained | TMR | 3 | 0.8775 | 0.6833 | 0.6528 | 0.6893 | 0.6587 | -7.636 (41.514%) |
| frozen | TMR | 1 | 0.5151 | — | — | 0.3711 | 0.3770 | -2.172 (77.875%) |
| frozen | TMR | 2 | 0.6216 | — | — | 0.4635 | 0.4611 | -3.656 (65.641%) |
| frozen | TMR | 3 | 0.8879 | 0.7070 | 0.6925 | 0.7130 | 0.6992 | -7.419 (42.565%) |

Equal-distortion mean rate gaps: OMR minus TMR; positive means OMR worse.

| Phase | Rate source | Gap bits/float | Gap % |
|---|---|---:|---:|
| all | pre_rate | +0.08691 | +11.697% |
| all | full_rans | +0.04135 | +7.313% |
| all | transport_rans | +0.03154 | +5.483% |
| frozen | pre_rate | +0.08969 | +12.047% |
| frozen | full_rans | +0.05147 | +9.013% |
| frozen | transport_rans | +0.04384 | +7.576% |
| trained | pre_rate | +0.07044 | +9.600% |
| trained | full_rans | -0.01927 | -3.628% |
| trained | transport_rans | -0.04213 | -7.602% |

Suspect stage details: `suspect_stages.csv`. Thresholds are <0.01 and >1.05 bits per stage.
These results alone cannot establish an oracle-vs-temporal limit: histories differ after decoding, selection is finite-budget, and frozen priors fail for specific clients.
