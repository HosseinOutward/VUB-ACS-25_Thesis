# Completed full-vector confirmation: full-SI rANS matrices

These are frozen diagnostic results, NOT complete trained+9-frozen protocol averages.
Clients 0/5/11/21, rounds9–16: 32 matched client-rounds per curve. Each entry is the
piecewise-linear equal-distortion percentage rate gap of row minus column; negative favors
the row. Both directions are calculated independently, because reference denominators differ.
Curves average cumulative rates and raw WMAPE before conversion to dB.

CF/CJ: cold scratch-trained O with fine-tuned/joint prior. WF/WJ: T-initialized O with
fine-tuned/joint prior. EF/ET/EJ: T-in-O embedding with full-SI fine-tuned/retained temporal/
joint prior. MT: TMR baseline encoder with its independent full-SI fine-tuned prior.
The temporal-retaining embedding ignores the extra SI, although full SI is available.

## No exclusions

| Row / Column | CF | CJ | WF | WJ | EF | ET | EJ | MT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CF | 0% | −26.97% | −32.41% | −27.36% | −6.71% | −9.06% | −38.50% | −6.58% |
| CJ | +36.92% | 0% | −7.14% | −0.20% | +29.14% | +25.88% | −14.86% | +29.31% |
| WF | +47.94% | +7.69% | 0% | +7.71% | +39.86% | +36.36% | −8.07% | +40.04% |
| WJ | +37.66% | +0.20% | −7.16% | 0% | +29.11% | +25.89% | −15.13% | +29.28% |
| EF | +7.19% | −22.57% | −28.50% | −22.55% | 0% | −2.33% | −35.04% | +0.14% |
| ET | +9.97% | −20.56% | −26.67% | −20.56% | +2.39% | 0% | −33.49% | +2.53% |
| EJ | +62.60% | +17.46% | +8.78% | +17.83% | +53.94% | +50.35% | 0% | +54.14% |
| MT | +7.05% | −22.67% | −28.59% | −22.65% | −0.14% | −2.47% | −35.13% | 0% |

## Pairwise symmetric client exclusion

Flag an actual full-SI rANS stage below0.01 or above1.05 bits anywhere in a curve's future
window; remove the union of flagged clients from BOTH curves across all eight rounds.
Only WF/WJ client0 is flagged among these displayed policies. Pairs involving WF/WJ retain
24 observations; other pairs retain32. This is retrospective sensitivity, not a correction.

| Row / Column | CF | CJ | WF | WJ | EF | ET | EJ | MT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CF | 0% | −26.97% | +2.09% | −6.30% | −6.71% | −9.06% | −38.50% | −6.58% |
| CJ | +36.92% | 0% | +28.29% | +17.75% | +29.14% | +25.88% | −14.86% | +29.31% |
| WF | −2.04% | −22.05% | 0% | −8.12% | −8.12% | −10.54% | −34.33% | −8.05% |
| WJ | +6.72% | −15.07% | +8.84% | 0% | −0.42% | −3.04% | −28.83% | −0.34% |
| EF | +7.19% | −22.57% | +8.83% | +0.42% | 0% | −2.33% | −35.04% | +0.14% |
| ET | +9.97% | −20.56% | +11.78% | +3.13% | +2.39% | 0% | −33.49% | +2.53% |
| EJ | +62.60% | +17.46% | +52.29% | +40.51% | +53.94% | +50.35% | 0% | +54.14% |
| MT | +7.05% | −22.67% | +8.76% | +0.35% | −0.14% | −2.47% | −35.13% | 0% |

The chronological selection chose CF for clients0/5 and WF for11/21 before future evaluation.
It beats MT by7.32% full-SI rANS without exclusion; no selected curve is flagged. The selected
full-SI prior gap is−7.36%, and native-transport gap (O full SI versus T temporal) is−9.72%.
This avoids one unselected WF failure, not all all-client TMR baseline failures (client14 is
outside this four-client diagnostic). Full rate values and raw WMAPE are in `runs/report.md`;
source stage points in `runs/stage_points.csv` permit recalculation. The custom candidate table
did not store marginal entropy; ordinary baseline `compression_records.csv` does. Do not
invent missing candidate marginal measurements from joint-prior rates.
