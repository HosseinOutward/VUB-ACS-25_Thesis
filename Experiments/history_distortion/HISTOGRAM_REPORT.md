# Fraction of coordinates distorted: history entry and full SI

These two histograms compare current five-round/per-client factors against the
round-shared proposal, using the same seed-44 ResNet18/SVHN identity trajectory.
The single entry is client 10, round 4, with 11,181,642 coordinates. Full server
SI is the snapshot at the start of round 5: all 30 clients' five entries from
rounds 0–4, totaling 150 entries and 1,677,246,300 matrix coordinates. Each model
parameter occurs 150 times in that matrix. Every matrix coordinate is measured;
there is no sampling or layer exclusion.

All values are in **original gradient units, before downstream normalization**.
The storage transform is reversed first. Full SI is read through the actual
`SideInfoMatrix` using identity target factors; eight columns and 200,000 rows
are processed at a time. Its exact reference is the corresponding raw updates.
The single-entry reference is that entry's raw update. Histograms weight individual
coordinates equally, so parameter tensors contribute in proportion to their size.

| Scope / storage | Numerically changed, all coordinates (%) | Error >0.1% among nonzero coordinates (%) | Error >1% (%) | Error >10% (%) | Error >100% (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single entry / current | 99.996959 | 27.970222 | 0.404055 | 0.040970 | 0.003141 |
| Single entry / round-shared | 99.997192 | 26.468434 | 0.080192 | 0.008336 | 0.000472 |
| Full SI / current | 99.996865 | 28.009541 | 0.475967 | 0.047201 | 0.004559 |
| Full SI / round-shared | 99.997335 | 26.916488 | 0.210459 | 0.020668 | 0.002082 |

“Numerically changed” means `stored != reference`, counting any nonzero numerical
error rather than requiring a tolerance. Signed-zero-only differences do not
count. APE is `100*abs(stored-reference)/abs(reference)` on **strictly nonzero
references**, without an epsilon. All threshold columns use the nonzero-coordinate
denominator. In full SI, the >1% counts are 3,023,449 current and 1,336,884 shared
out of 635,222,867 nonzero reference cells. Thus most numerical changes are small;
large relative errors occur in a much smaller tail. Shared factors reduce the
full-SI >1% tail by about 55.8% while leaving a tail.

Exact zeros account for **62.127% of all source coordinates**. Both storage
schemes change every one of them into nonzero values: 6,946,824 for the single
entry and 1,042,023,433 for full SI. Their percentage error is undefined, so they
are counted separately rather than assigned an artificial denominator. The plots
therefore describe the remaining 37.873% of coordinates. Counting any numerical
change across all coordinates includes these zero changes and explains the
approximately 99.997% affected fraction.

The y-axis is linear and capped at **0.5% of nonzero reference coordinates per
bin**, exposing the tail while clipping the bulk peak. Actual peak bins are
annotated (about 16–18%). The x-axis uses a logarithmic scale because errors span
many orders of magnitude. Zero-error and <1e-6% bins, and the >=1e6% overflow bin,
are retained in the CSV although not drawn on the logarithmic x-axis. Peak-bin
clipping changes only the display; counts and threshold statistics remain complete.

![Single-entry and full-SI histograms](results/svhn_seed44_raw_full_si_histogram_20261006/histograms.png)

Stacking history entries introduces no additional storage quantization. The
different histograms reflect the selected entry versus the full population of
stored updates. These inputs are real raw identity-FL updates; codec-generated
reconstructions could have different distributions. No downstream quantizer or
entropy coder is trained here.

Coverage, complete bin counts, and zero accounting were verified. The selected
single entry reproduces its earlier recorded MAPE and WMAPE. Five focused tests
pass, including an explicit zero-reference/threshold-count test.

Evidence: [PDF figure](results/svhn_seed44_raw_full_si_histogram_20261006/histograms.pdf),
[summary counts](results/svhn_seed44_raw_full_si_histogram_20261006/summary.csv),
[complete histogram bins](results/svhn_seed44_raw_full_si_histogram_20261006/histogram_bins.csv),
[configuration](results/svhn_seed44_raw_full_si_histogram_20261006/metadata.json).
