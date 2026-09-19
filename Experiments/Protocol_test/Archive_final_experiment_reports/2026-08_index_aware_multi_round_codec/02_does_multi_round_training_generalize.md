# Does lag-1 training generalize after freezing?

Setup and caveats: [00_setup.md](00_setup.md). Results come from the six complete final records and
[rd_gaps.csv](results/analysis/rd_gaps.csv).

## T4 — Non-index OM against O
[`OM`](results/records/OM/compression_records.csv), [`O`](results/records/O/compression_records.csv)

**Idea.** Measure whether an aligned lag-1 problem helps without any coordinate embedding.

**How.** OM reuses the same corrected snapshot/live-current policy as OMI, but keeps the ordinary
74×6, 140-epoch codec and uniform sampling over current and lag-1 blocks. O is identical except for
the extra problem. This control rules out the claim that every multi-round gain requires indexing.

**Result.** OM is `−0.00879 bit` vs O (1.44× the 0.0061 floor) and `−0.01706 bit` vs T (2.80× floor).

**Verdict.** Lag-1 data helps the oracle codec slightly; OM is the best final curve.

## T5 — Non-index TM against T
[`TM`](results/records/TM/compression_records.csv), [`T`](results/records/T/compression_records.csv)

**Idea.** Repeat T4 where only worker-local temporal SI is legal.

**How.** TM uses the same ordinary model and uniform current/lag-1 pool as OM, but both training and
frozen refresh see only the current worker's temporal ledger. T is the no-lag control.

**Result.** TM is `+0.01466 bit` vs T (2.40× floor): `+0.00305` during training but `+0.01611` frozen.

**Verdict.** Lag-1 replay helps oracle SI but hurts temporal-only SI in this implementation.

## T6 — Training-round gain against frozen-round reversal
[`OMI`](results/records/OMI/compression_records.csv),
[`TMI`](results/records/TMI/compression_records.csv),
[`OM`](results/records/OM/compression_records.csv),
[`TM`](results/records/TM/compression_records.csv)

**Idea.** Separate adaptation quality from reusable quality; averaging all 45 rows can hide a model
that is excellent exactly where it was fitted and poor everywhere it is reused.

**How.** The same RD-gap calculation is applied once to the five round-12 rows and once to the forty
rounds-13-20 rows. OMI is compared with OM and TMI with TM so both sides receive lag-1 data and the
same decoder-visibility policy; capacity and sampler remain intentionally unmatched as stated in
the scope.

**Result.** OMI vs OM changes from `−0.01903 bit` training (3.12× floor better) to `+0.02100 bit`
frozen (3.44× worse). TMI vs TM changes from `−0.01939` to `+0.01308 bit`. OMI still beats TMI in
both subsets: `−0.01665` training and `−0.02548 bit` frozen.

**Verdict.** The index adapter improves fit, not frozen generalization; forty frozen rows reverse the win.
