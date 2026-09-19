# Common setup, controls and scope

Shared by every report in this folder. Preserved configurations and rerun commands are indexed in
the [run manifest](results/run_manifest.md).

## The measurement

Every final run replays rounds 12-20 for five clients from
`_/data/federated_learning/raw_client_deltas`: five independently trained codecs at round 12 and
forty frozen evaluations at rounds 13-20. Each float emits three 3-way refinement symbols. Rate is
the retrained conditional-prior estimate; distortion is stage WMAPE in dB. Reported gaps are the
mean horizontal separation between piecewise-linear RD curves over their shared distortion range.
Negative means the candidate needs fewer bits at matched distortion.

The standing noise floor is **0.0061 bit**, measured by retraining O at seed 44 in the preceding
cross-worker investigation. Every verdict here compares its absolute gap with that floor.

The six final controls are:

| mode | training SI | extra problem | model |
|---|---|---|---|
| O | all server-visible histories | none | 74×6, 140 epochs |
| T | own temporal history | none | 74×6, 140 epochs |
| OM | all server-visible histories | lag 1 | 74×6, 140 epochs |
| TM | own temporal history | lag 1 | 74×6, 140 epochs |
| OMI | all server-visible histories + coordinate embedding | lag 1 | 112×8, 180 epochs |
| TMI | own temporal history + coordinate embedding | lag 1 | 112×8, 180 epochs |

O/OM/OMI are oracle study modes: the current raw target and server-only SI meet during training.
T/TM/TMI train where both model halves are available and are the non-WZ comparison requested by
the study. Frozen rounds refresh only SI legitimately visible at their declared decode location.

## Exact index conditioner

The local coordinate is always in `[0, vector_size)`. V2 transports it exactly as a float32 integer
under the asserted `vector_size <= 2**24` limit. Eight seeded hash projections of width 24 and the
raw 24 identity bits feed one 64-route router and a shared 64×24 embedding bank. Only the routed
24-wide embedding is appended to ordinary SI; raw bits never enter the downstream RNN directly.
The same conditioner feeds encoder, decoder and in-model prior, so all three losses return gradients
to the router and embedding bank.

V2 disables cross-index inference deduplication and uses 32-bit training matching. Nearby positions
can therefore select different mixtures even when their values and ordinary SI match. The routed
residual is bounded and nonnegative: OMI starts at 0.01 with a 0.05 cap; TMI starts at 0.0002 with a
0.001 cap. TMI is deliberately conservative because temporal SI is already strong.

## Multi-round policy

Lagged target/SI pairs come from a frozen round-start snapshot so every pair is aligned. The current
OM/OMI SI is refreshed from live server history, including earlier clients committed in the same
round; this fixes the original snapshot bug. Only lag 1 is used. Earlier dense `(1,2,3)` and gapped
histories diluted a fixed 550k-sample budget with drifting problems and performed worse.

OMI samples 75% current and 25% lag-1 rows. TMI samples 95% current and 5% lag-1 rows. Ordinary
OM/TM retain the historical uniform two-block sampler. Each final indexed codec trains two
candidates; any candidate whose refinement plane emits only one symbol over the full nonconstant
vector is rejected and retried.

## Scope

This is one operating point, model, dataset and coordinate presentation. It establishes that this
specific shared hashed-router integration does not improve frozen RD. It does not rule out learned
coordinate features generally.

The final comparison intentionally gives OMI/TMI greater capacity and a different sampler than
OM/TM because the proposed flagship was the complete improved package. Consequently OMI-vs-OM
cannot isolate the embedding from capacity, epochs or sampling. A matched-capacity non-index model
is still required for that causal claim.

Quantizer parameters are cached under `_/data/protocol_cache/flagship_comparison`. Separate entropy
priors are not retained for indexed or multi-round reference modes. Caches are runtime accelerators
and remain outside this portable report.

## Sparse encoder-history extension

T11–T12 use a separate multi-start replay preserved under
[results/sparse_encoder_history](results/sparse_encoder_history/). For each client `c`, training
starts at rounds `10+5c` and `15+5c`; every start is duplicated once with the same seed. Each of the
20 cases trains once and freezes for twelve rounds, giving 260 rows per codec. OM, TM, OM_FE and
TM_FE share the ordinary 74×6, 140-epoch model, lag offsets 1 and 2, and current/lag-1/lag-2 sampling
weights `0.50/0.33/0.17`.

OM_FE and TM_FE change only encoder input: the requesting client's five aligned local-history
values are appended beside the current scalar. Decoder and conditional prior inputs are unchanged.
The source files are lossless float32 raw client deltas, but replay history converts them to float16
when committed. “Raw-source history” below distinguishes them from reconstructions; it does not
mean bit-exact float32 preservation. The matched-distortion calculation is unchanged. Duplicate
repeats give a maximum `0.00049-bit` full-curve spread for the four relevant modes, while the older
`0.0061-bit` O cross-seed floor is retained as the conservative threshold.
