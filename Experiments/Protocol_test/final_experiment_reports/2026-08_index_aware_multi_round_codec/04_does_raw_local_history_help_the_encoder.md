# Does raw local history help the encoder?

Setup and caveats: [00_setup.md](00_setup.md). Results come from the complete sparse multi-start
records and [encoder_history_gaps.csv](results/sparse_encoder_history/analysis/encoder_history_gaps.csv).
Positive matched-distortion rate gap is worse.

## M3 — Duplicate-repeat stability
[`repeat_gaps.csv`](results/sparse_encoder_history/analysis/repeat_gaps.csv)

**Idea.** Establish a local run-to-run scale before interpreting the encoder-history gaps.

**How.** Each of the ten unique client/start trajectories was run twice with the same seed, data and
configuration. Repeat 2 is compared with repeat 1 using the same averaged three-stage RD calculation
as the codec comparisons. This checks execution and prior-training variation, not cross-seed model
variation; the older `0.0061-bit` cross-seed O floor remains the conservative reference.

**Result.** The largest full-curve repeat gap among OM, TM, OM_FE and TM_FE is `0.00049 bit`; the
largest training/frozen subset gap is `0.00135 bit`, both below the `0.0061-bit` historical floor.

**Verdict.** The approximately `0.012-bit` FE losses are not duplicate-run noise.

## T11 — Feed local raw-source history to the OM encoder
[`OM_FE`](results/sparse_encoder_history/records/OM_FE/compression_records.csv),
[`OM`](results/sparse_encoder_history/records/OM/compression_records.csv)

**Idea.** Test whether the requesting client's past updates help the encoder choose symbols when the
decoder and entropy prior retain OM's stronger server-visible side information.

**How.** OM_FE is OM with only the encoder widened. Alongside the normalized current coordinate it
receives the five aligned history values belonging to the requesting client; the decoder and
in-model prior still receive ordinary OM side information. The same selection is made separately
for the current, lag-1 and lag-2 training problems, sampled with probabilities `0.50/0.33/0.17`.
The comparison pools 20 one-train-plus-12-frozen trajectories: clients 0–4, starts `10+5c` and
`15+5c`, each duplicated once.

**Result.** OM_FE is `+0.01233 bit` vs OM over all 260 rows (2.02× the `0.0061-bit` floor):
`+0.01195 bit` on training rows and `+0.01235 bit` on frozen rows.

**Verdict.** Local history at the encoder worsens OM consistently across training and reuse.

## T12 — Feed local raw-source history to the TM encoder
[`TM_FE`](results/sparse_encoder_history/records/TM_FE/compression_records.csv),
[`TM`](results/sparse_encoder_history/records/TM/compression_records.csv)

**Idea.** Repeat T11 when the requesting client's temporal history is the only decoder-visible side
information, so exactly the same values become available to both encoder and decoder.

**How.** TM_FE widens only TM's encoder and feeds its five temporal-history values. Target, decoder,
prior, lag offsets, block weights and sparse trajectory schedule match TM. This isolates
the effect of revealing already decoder-known temporal context to the encoder in this architecture.

**Result.** TM_FE is `+0.01213 bit` vs TM over all 260 rows (1.99× the `0.0061-bit` floor):
`+0.01578 bit` on training rows and `+0.01182 bit` on frozen rows.

**Verdict.** Making temporal side information encoder-visible also worsens RD; this FE integration is rejected.
