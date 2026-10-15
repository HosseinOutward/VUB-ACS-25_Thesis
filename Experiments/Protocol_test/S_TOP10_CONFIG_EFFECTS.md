# Configuration effects in the selected top ten S variants

The ten variants are selected from the previous S avg vs O trimmed ordering. They are displayed below in S trimmed vs O trimmed order.
Avg means all five seeds; trimmed means each configuration’s independently selected best three. Original O configuration is used. O trimmed retains seeds 43,46,47.
Full-SI equal-distortion rate changes (%); negative favors S trimmed. These are one-client/one-round trained diagnostics.

| Variant | Samples | Batch | Epochs | LR × | Tau | S trimmed vs O trimmed % | S retained seeds |
|---|---:|---:|---:|---:|---:|---:|---|
| S04 | 3.6M | 7500 | 28 | 0.8 | 0.2 | +0.457 | [43, 44, 46] |
| S08 | 3.6M | 7500 | 28 | 0.8 | 0.3 | +0.834 | [43, 44, 47] |
| S22 | 3M | 6250 | 35 | 1.0 | 0.25 | +1.041 | [43, 44, 45] |
| S32 | 2.4M | 7500 | 42 | 0.8 | 0.3 | +1.334 | [44, 46, 47] |
| S03 | 3.6M | 5000 | 28 | 0.8 | 0.2 | +1.599 | [43, 44, 45] |
| S28 | 2.4M | 7500 | 42 | 0.8 | 0.2 | +1.706 | [43, 46, 47] |
| S37 | 2.4M | 7500 | 42 | 1.2 | 0.2 | +1.743 | [44, 45, 46] |
| S19 | 3M | 6250 | 35 | 1.0 | 0.2 | +1.899 | [44, 45, 47] |
| S34 | 3.6M | 7500 | 42 | 0.8 | 0.3 | +2.193 | [45, 46, 47] |
| S31 | 2.4M | 5000 | 42 | 0.8 | 0.3 | +2.432 | [43, 46, 47] |

## Controlled changes

Only one parameter changes within each pair. The RD columns use each S curve’s shared range with O trimmed.
The rate-effect and paired-seed columns use the same WMAPE dB range [-5.017826973, -3.954611888], preventing changes in comparison range from being mistaken for parameter effects.
Paired-seed counts use all five same-seed comparisons before trimming. Best-three averages can retain different seeds.

| Change | Pair | Before vs O trimmed % | After vs O trimmed % | Common-span trimmed rate change % | Improved paired seeds / 5 |
|---|---|---:|---:|---:|---:|
| Batch 5000 → 7500 | S03 → S04 | +1.599 | +0.457 | -1.483 | 2 |
| Batch 5000 → 7500 | S31 → S32 | +2.432 | +1.334 | -0.313 | 2 |
| Epochs 28 → 42 | S08 → S34 | +0.834 | +2.193 | +1.662 | 1 |
| Samples 2.4M → 3.6M | S32 → S34 | +1.334 | +2.193 | +0.817 | 1 |
| Tau .20 → .30 | S04 → S08 | +0.457 | +0.834 | +0.289 | 2 |
| Tau .20 → .30 | S28 → S32 | +1.706 | +1.334 | -0.804 | 3 |
| LR .8× → 1.2× | S28 → S37 | +1.706 | +1.743 | -0.300 | 3 |
| Tau .20 → .25 | S19 → S22 | +1.899 | +1.041 | -2.240 | 4 |

## Interpretation

- Batch size: 7500 improves the trimmed comparison in S03→S04 and S31→S32. Only 2/5 same-seed comparisons improve in each pair, so this is a selected-best-three effect rather than a consistent per-seed gain.
- Epochs: at 3.6M samples, batch 7500, LR .8× and tau .30, 42 epochs worsens S08→S34. The common-span trimmed rate rises 1.662%; four of five individual seeds also worsen.
- Sample count changes the available pool, not the fixed 550,000 draws per epoch; pools at different counts need not be nested. At 42 epochs, batch 7500, LR .8× and tau .30, raising 2.4M to 3.6M worsens S32→S34. Four of five seeds worsen; increasing the sample pool is not universally helpful.
- Tau interacts with the other settings: .20→.30 worsens S04→S08 at 28 epochs/3.6M, but improves S28→S32 at 42 epochs/2.4M. At the center, .20→.25 improves S19→S22 in four of five seeds.
- LR: seven selected variants use .8×, but that frequency comes from selecting top-performing configurations. In the isolated S28→S37 change, .8×→1.2× is nearly tied against O trimmed (+1.706% versus +1.743%); the common-span trimmed rate slightly improves by .300%. There is no consistent preference from that pair.
- Center: S22 is the best setting among the measured one-axis changes about the center. Every individual lower/higher change in epochs, LR, tau, sample count or batch worsens its trimmed gap against O trimmed. S04 improves the reported gap through several simultaneous changes, rather than one universally favorable knob.
- Interpretation limits: changing the selected seeds contributes to the trimmed results. Epochs and batch size also change the optimizer trajectory and the subsequent prior’s continuing RNG stream. These measurements do not isolate a mechanistic explanation such as overfitting.

S04 is closest to O trimmed within this selected ten (+0.457%). None of these ten beats O trimmed.
[All 43 variants and cross-cohort comparisons](S_O_AVG_TRIMMED_REPORT.md).
