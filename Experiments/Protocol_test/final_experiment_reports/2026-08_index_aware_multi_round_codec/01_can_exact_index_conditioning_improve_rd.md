# Can exact coordinate conditioning improve rate-distortion?

Setup and caveats: [00_setup.md](00_setup.md). The compact source for every number is
[archive_rd_gaps.csv](results/analysis/archive_rd_gaps.csv) or
[rd_gaps.csv](results/analysis/rd_gaps.csv).

## M1 — Exact nearby-coordinate identity
[Implementation and tests](experiment/index_aware_codec/index_aware_model.py),
[quantizer integration](experiment/index_aware_codec/index_quantizer.py)

**Idea.** Determine whether adjacent coordinates can receive different learned features; value
hashing or modulo aliasing would make a negative RD result uninterpretable.

**How.** Coordinates are exact integers in `[0, vector_size)`, supplied to eight seeded hashes and
the shared router. V2 bypasses cross-index encoder/decoder deduplication, resets coordinates for each
lagged vector, and uses 32-bit training matching. A regression programs the router and embedding
bank so adjacent equal-valued rows produce different outputs. Encoder, decoder and in-model prior
share the same conditioner parameters.

**Result.** Exact through `2**24` positions; adjacent indices are distinguishable; no RD comparison.

**Verdict.** Precision was achieved and is not the explanation for the final negative result.

## T1 — First-generation single-round OI and TI
[`index_aware_v1/OI`](results/archive/index_aware_v1/index_aware_OI/compression_records.csv),
[`index_aware_v1/TI`](results/archive/index_aware_v1/index_aware_TI/compression_records.csv)

**Idea.** Add index conditioning to O and T without changing their single current-round training
problem; this is the direct first test of whether coordinates carry reusable structure.

**How.** V1 uses six 16-wide hashes, 48 routes, a 16-wide embedding, 88×7 RNNs and 140 epochs. It
retains approximate cross-coordinate deduplication and has no bounded residual or weighted lagged
training. OI is compared with the historical O rows and TI with historical T over identical keys.

**Result.** OI is `−0.05152 bit` on 5 training rows but `+0.13240 bit` on 40 frozen rows, yielding
`+0.11413 bit` full vs O (18.71× the 0.0061 floor). TI similarly moves from `−0.05348` to `+0.09866`,
yielding `+0.08175 bit` full vs T (13.40× floor).

**Verdict.** The conditioner memorizes the training problem and catastrophically fails frozen reuse.

## T2 — First-generation multi-round OMI and TMI
[`index_aware_v1/OMI`](results/archive/index_aware_v1/index_aware_OMI/compression_records.csv),
[`index_aware_v1/TMI`](results/archive/index_aware_v1/index_aware_TMI/compression_records.csv)

**Idea.** Give the indexed model several historical target/SI problems so coordinate features must
explain more than one round.

**How.** V1 OMI/TMI train on current plus offsets `(1,2,3)` with equal block exposure. Lagged pairs
are aligned by round, but this version predates the exact-index, live-current-SI and bounded-router
fixes. The control is the same historical O/T records as T1.

**Result.** OMI is `+0.01625 bit` vs O (2.66× floor); TMI is `+0.03016 bit` vs T (4.94× floor).

**Verdict.** Multi-round data removes most single-round overfit, but all indexed gains remain negative.

## T3 — Final expanded OMI and TMI
[`OMI`](results/records/OMI/compression_records.csv),
[`TMI`](results/records/TMI/compression_records.csv)

**Idea.** Test the complete improved proposal: exact identity, shared gradients, larger models,
longer training, weighted lag-1 replay, bounded routing and candidate collapse rejection.

**How.** Both indexed models use 112×8 RNNs, 180 epochs, two training attempts and the V2 router in
[00_setup.md](00_setup.md). OMI reserves 75% of samples for the current problem and permits a 0.05
index residual; TMI reserves 95% and caps it at 0.001. Final curves contain all five training and
forty frozen client-rounds.

**Result.** OMI beats TMI by `−0.02457 bit` (4.03× floor), but is `+0.00787 bit` vs O (1.29× floor),
`+0.01666 bit` vs OM (2.73× floor), and `−0.00043 bit` vs T (0.07× floor). TMI is `+0.02417 bit` vs T
(3.96× floor).

**Verdict.** OMI beats the non-WZ indexed control, but exact indexing does not produce the best curve.
