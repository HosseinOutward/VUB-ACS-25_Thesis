# Does splitting the vector across specialised quantizers help?

Setup and scope: [00_setup.md](00_setup.md).

---

## M1 — Which coordinates actually use the extra side information?
[run5](results/records/run5_longf_o_elem/compression_records.csv) vs
[run7](results/records/run7_longf_t_elem/compression_records.csv) element dumps; compact derived
tables are retained in [the M1 analysis](results/analysis/elements_o_vs_t/strata_summary.csv).

**Idea.** If some coordinates need the cross-worker columns and others do not, separate quantizers
could specialise on that axis. First measure which is which.

**How.** For each coordinate, its **own 3-stage RD curve** under O against its curve under T. Rounds
16/18/20 averaged, **workers kept separate** — each worker trains its own quantizers, so the
partition is per worker and need not be shared. Distortion is WMAPE in dB; first-crossing
interpolation on the real measured points, never a monotonised or extrapolated curve.

**Result.** ~17% T-preferring, ~65% indifferent, ~18% O-preferring per worker. Cross-worker
correlation of the per-coordinate gap: **0.004-0.025**.

**Verdict.** Two thirds of every worker's vector does not care which side information the codec had.
Which coordinates benefit is **not a property of the coordinate** — it barely correlates between
workers.

---

## T13 — Three quantizers on the measured partition
[run28_longf_op3db](results/records/run28_longf_op3db/compression_records.csv), using the retained
[three-way partition](results/partitions/three_way_db.npy)

**Idea.** Give T-preferring, indifferent and O-preferring coordinates their own quantizer, all with
O's full side information, so only the partition differs.

**How.** [`OP_RoundCodec`](experiment/experiment_codecs/partition_round_codec.py): one quantizer per
label, applied to *rows* after a single whole-vector
preprocessing, so normalization, metadata and SI stay identical to O. Each part priced by its own
conditional prior. A `frozen_round_codec` hook keeps the split alive through the frozen rounds —
`F_RoundCodec` takes one quantizer, so without it all 8 frozen rounds would use one part's model.

**Result.** **-0.0016 bits, t = -0.30** on held-out rounds with workers as the unit. The same test
detects O-vs-T at **t = +5.45**.

**Verdict.** No improvement, and the control shows the test has ample power to find one.

---

## T14 — Reassign each coordinate to whichever quantizer codes it best
[run29_longf_op3_crosseval](results/records/run29_longf_op3_crosseval/compression_records.csv) ->
[run30_longf_op3_iter2](results/records/run30_longf_op3_iter2/compression_records.csv), using the
[reassigned partition](results/partitions/three_way_reassigned.npy)

**Idea.** T13's partition was chosen before those three quantizers existed. Ask them directly — this
is Lloyd's algorithm over a mixture of quantizers, and costs nothing extra to transmit.

**How.** `cross_evaluate=True` codes the **whole** vector with all three, each priced by a prior
trained on its own whole-vector symbols. Each coordinate goes to the one reaching lowest distortion
**at matched rate** — rate is the monotone axis; matching on distortion instead rewards a part that
never reaches the target, which a hand-built test caught.

**Result.** 39.7% of coordinates changed hands. Pooled, **-0.69%**. But split by round: **-0.0099 on
the three rounds the partition was fitted to, -0.0005 on rounds it never saw**.

**Verdict.** The apparent gain was **overfitting** — 11.18M coordinates assigned from 15 client-round
measurements. Held out, it is worse than T13. Any future reassignment must be scored on unfitted
rounds.

---

## M2 — What did the split actually learn?
analysis of the [run29-derived labels](results/partitions/three_way_reassigned.json)

**Idea.** If the assignment clusters structurally — by layer, or by the weights feeding one neuron —
it could be assigned for free instead of transmitted.

**How.** Cross-tabulate the data-driven labels against depth; against each unit's incoming weights
(reshape to (units, incoming), compare purity to the same tensor's labels **shuffled**); and against
magnitude and novelty deciles, computed separately inside and outside layer4 to disentangle layer
from magnitude.

**Result.** What each part became, measured by coding the whole vector with each: **part1 is
cheap-and-coarse** (0.752 bits / 12.6% wmape on worker 0; 0.670 / 16.2% on worker 4), **parts 0 and 2
are expensive-and-accurate** (~0.90 bits / 10-11%).

Share of the vector each part receives:

| worker | part0 | part1 | part2 |
|---|---:|---:|---:|
| 0 | 0.158 | 0.668 | 0.174 |
| 1 | 0.196 | 0.637 | 0.167 |
| 2 | 0.217 | 0.666 | 0.118 |
| 3 | 0.154 | 0.630 | 0.215 |
| 4 | 0.395 | 0.210 | 0.395 |

**Magnitude decides everything.** Decile of normalized `|x|`, averaged over all 5 workers:

| decile | part0 | part1 | part2 |
|---|---:|---:|---:|
| 0-5 (small) | 0.04-0.17 | **0.69-0.79** | 0.04-0.24 |
| 6 | 0.378 | 0.418 | 0.203 |
| 7-9 (large) | ~0.38 | **0.23-0.29** | ~0.37 |

Small coordinates go to the cheap quantizer, large ones split evenly between the two accurate ones,
and the transition sits at decile 6.

**Layer and kind are proxies, not causes.** By depth, mean over workers:

| depth | coords | part0 | part1 | part2 |
|---|---:|---:|---:|---:|
| stem/fc | 14.7k | 0.382 | 0.252 | 0.367 |
| layer1 | 148k | 0.380 | 0.256 | 0.364 |
| layer2 | 526k | 0.382 | 0.256 | 0.362 |
| layer3 | 2.10M | 0.380 | 0.262 | 0.358 |
| **layer4** | **8.39M** | 0.172 | **0.662** | 0.165 |

Everything but layer4 sits at a flat 0.38 / 0.25 / 0.36. Same by kind: conv 0.221 / 0.567 / 0.211,
while downsample, batchnorm and fc are all ~0.38 / 0.25 / 0.36. Inside layer4, deciles 0-5 go
81-99.6% to part1; **outside layer4 the decile pattern is flat**, which is what separates the layer
story from the magnitude one. Neuron purity **0.6950 against 0.6948 shuffled**.

## Interpretation

The three quantizers specialised by **rate operating point**, not by side-information preference. A
mixture sitting at different points on the same curve can only slide the aggregate along it, which is
exactly what T13 and T14 do.

The reassignment rediscovered a **magnitude threshold** — something the multi-stage quantizer and
per-slice normalization already do, so there was nothing left for it to win. Layer4 looks special
only because it holds most of the small-magnitude mass: the layer signal is magnitude wearing a layer
costume. There is **no neuron-level structure**, so the partition cannot be assigned for free and
would have to be transmitted.
