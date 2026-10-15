# Real-gradient history storage results

The proposed round-shared normalization schedule and its timing analysis are in
[ROUND_SHARED_REPORT.md](ROUND_SHARED_REPORT.md).
The affected-coordinate percentages and raw full-SI histograms are in
[HISTOGRAM_REPORT.md](HISTOGRAM_REPORT.md).

Branch: `experiment/history-storage-distortion`. Measured seed-44 ResNet18/SVHN
local model updates from clients 0, 10, and 20 in rounds 0–5. All 30 clients
participated in generating the identity FL trajectory. This covers 18 vectors,
62 trainable parameter tensors per vector, and 201,269,556 float32 values.
Layer-observation RMS ranges from 0.000310 to 0.01925. The window includes the
production normalization refresh in round 5. No clients or layers were excluded.

**The active history storage is lossy, and global WMAPE hides severe relative
errors near zero.** The main mechanism in the current implementation is centering
before mantissa truncation. A small raw value can become a comparatively large
normalized value; truncating it and restoring the center loses precision through
cancellation. Float16 without scaling has an additional underflow problem.
Across whole layers, error does not systematically explode as layer RMS falls;
the within-layer small-value diagnostic reveals the strongest effect.

All errors below are percentages. Memory is local tensor payload bytes per
original float, including scale metadata, and is not a communication bitrate.
WMAPE and MAPE pool their respective sums before division. Worst-layer MAPE
is the worst individual parameter-tensor/client/round observation.

| Storage variant | Global WMAPE | Nonzero MAPE | Worst-layer MAPE | Bytes/float |
| --- | ---: | ---: | ---: | ---: |
| Exact float32 history | 0 | 0 | 0 | 4.0000 |
| Uncentered 18-bit prefix | 0.070233 | 0.069443 | 0.094834 | 1.7953 |
| Current history, five-round factors | 0.071690 | 0.119519 | 44.095190 | 1.7946 |
| Current normalization fitted every vector | 0.070915 | 0.088809 | 44.095190 | 1.7948 |
| Power-of-two scaled 18-bit prefix | 0.070233 | 0.069443 | 0.094834 | 1.7946 |
| Unscaled float16 | 0.017688 | 0.034636 | 0.411085 | 2.0000 |
| Power-of-two scaled float16 | 0.017610 | 0.017547 | 0.037072 | 2.000022 |

There are 125,042,809 exact zero source values (62.13% of all values). Current
history changes every one of them to a nonzero value, with maximum absolute
error about 2e-6; freshly fitted centering also changes them, though its maximum
zero error is smaller, about 1.85e-7. Uncentered variants preserve these zeros.
Current history flips the sign of 1,320 nonzero values; fresh centering flips
660. Uncentered variants produce no sign reversals in this sample.

The smallest nonzero magnitude bin contains 8,809 values; the next contains
80,772. Small-value MAPE is deliberately sensitive to tiny denominators and
should be read alongside those counts, WMAPE, and the absolute errors.

| Nonzero magnitude / layer RMS | Uncentered prefix MAPE | Current MAPE | Fresh-centered MAPE | Unscaled float16 MAPE | Scaled float16 MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Below 0.0001 | 0.000753 | 227.787369 | 83.218921 | 50.317723 | 0.271590 |
| 0.0001–0.001 | 0.008288 | 9.049673 | 4.028288 | 4.231622 | 0.006997 |
| 0.001–0.01 | 0.045465 | 0.923901 | 0.410313 | 0.418575 | 0.011286 |
| 0.01–0.1 | 0.067543 | 0.113334 | 0.081382 | 0.043335 | 0.017618 |
| 0.1–1 | 0.069805 | 0.070565 | 0.069870 | 0.017637 | 0.017568 |
| At least 1 | 0.070378 | 0.070526 | 0.071206 | 0.017633 | 0.017633 |

The worst-layer example is `model.layer4.0.bn1.bias`, client 10, round 0:
WMAPE is 0.0708%, MAPE is 44.10%, and the maximum individual percentage error
is 22,500.7%. Its 99th percentile is only 2.59%, showing that the mean MAPE is
driven by an extreme near-zero tail. Across the entire run the largest current
individual percentage error is 181,350%. Mean layer-observation MAPE is 0.2685%
for current history and 0.0673% for uncentered prefix storage.

Cheap power-of-two scaling substantially helps float16: nonzero values lost to
zero fall from 2,932 to 15, and smallest-bin MAPE falls from 50.32% to 0.272%.
It uses about 11.4% more payload memory than current packed history, while
reducing global WMAPE about fourfold. Scaling does not eliminate underflow for
every value within a layer. With prefix storage, power-of-two scaling produces
identical measured errors to unscaled prefix storage; keeping the full float32
exponent already protects small normal values.

Refreshing robust centered factors more often helps but does not remove the
problem. Current nonzero MAPE climbs from 0.1030% in round 0 to 0.1400% in
round 4, then falls to 0.0809% after the round-5 refresh. Fresh fitting still has
83.22% MAPE in the smallest bin and changes the exact zeros.

For preserving tiny values at the present memory budget, the simplest candidate
is uncentered prefix storage. Cheap scale-only normalization offers similar
memory use with the same observed errors. Scaled float16 is another candidate
if slightly higher memory use is acceptable. A downstream normalization step
cannot restore information lost when history was stored.

These measurements isolate history round-trip error on raw real FL updates.
Production history normally receives codec reconstructions, whose distributions
may differ. History copies do not directly replace the just-decoded update used
by aggregation; they affect subsequent training and decoding through side
information. This experiment does not establish a convergence or end-to-end RD
improvement. Those comparisons would require matched codec replay.

The three focused tests pass, covering zero-aware metrics, centering errors in
the actual decoder row-access path, and scaling against float16 underflow.
Coverage, complete magnitude-bin partitions, and exact float32 round trips were
also verified. Evidence: [summary CSV](results/svhn_seed44_round0_5_complete/summary.csv),
[per-layer CSV](results/svhn_seed44_round0_5_complete/layers.csv),
[magnitude-bin CSV](results/svhn_seed44_round0_5_complete/magnitude_bins.csv),
[storage CSV](results/svhn_seed44_round0_5_complete/storage.csv), and
[trace metadata](results/svhn_seed44_round0_5_complete/metadata.json).

![Layer and small-value distortion](results/svhn_seed44_round0_5_complete/distortion.png)
