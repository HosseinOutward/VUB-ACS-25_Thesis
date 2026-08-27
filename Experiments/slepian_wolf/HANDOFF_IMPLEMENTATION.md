# Slepian–Wolf implementation handoff

Temporary reference for `FL_code/cancer_protocol/sw_coder.py`.

## Implemented path

Input is a `(stages, positions)` tensor of power-of-two quantizer symbols. For each stage:

1. Obtain decoder-owned `q(U_stage | decoded earlier stages, side information)`.
2. Split symbols into bit planes, least-significant bit first.
3. Condition each plane on its decoded lower planes and form source LLRs.
4. Send a packed LDPC syndrome for each block and plane.
5. Recover planes with sum-product BP, then use the recovered stage in the next-stage prior.

`SWRatePlan` completely identifies block length, rows, graph profile, constructor version and seed. Matrices are reconstructed deterministically. `SWPayload` contains the source length, plan ID and packed syndromes. On this path there is no CRC, retransmission or uncoded fallback.

## Two code families

`matrix_backend` selects how a plan's row count becomes a parity-check matrix.

| | `balanced_4cycle_v1` | `ldpca_split_v1` |
|---|---|---|
| construction | one matrix per rate, rejected until full GF(2) row rank | one rate-one mother per plane; a rate is a merge of its checks |
| cost at n=2400 | ~0.3 s per rate | 0.26 s per plane, then **6 ms** per rate |
| rank | full by rejection | measured full at every level below n, deficient by 1 at n |
| rate adaptivity | none: the rate is fixed before anything is sent | nested, one row per level, over a feedback channel |
| BSC blocks decoded exactly, matched rows | 36/84/96/99 of 100 | 18/52/79/93 of 100 |

The last row is one BSC (p=0.05, H=0.286) at rows equal to `(H + 0.06/0.10/0.14/0.18) * n`; at p=0.2 the ordering reverses, so the merged code's disadvantage is a low-rate one.

At the rates this codec operates at, roughly 0.3 to 0.6, the merged code is the weaker of the two at a fixed rate, and that is inherent to LDPCA: `_merged_structure` inherits the mother's cycles instead of suppressing them at the row count actually used. It wins anyway because it is the only one that can charge each block separately -- see `HANDOFF_LDPCA.md`.

`decode_ladder` is the LDPCA-only decoder. It starts every block at the plan's row count and climbs `SWCoderConfig.ladder_step` rows at a time until the block's hash verifies, so the plan is a starting guess rather than a commitment, and a `PlaneVerification` is mandatory rather than optional.

## Paper fidelity

| Item | Status |
|---|---|
| Zhang–Tao LSB-first multilevel equations and entropy chain rule | Implemented |
| Decoder-side conditional prior and successive decoding | Implemented |
| `ceil(nH)` row accounting, `n=2400`, 100 BP iterations | Implemented/defaults |
| Gaussian-CEO posterior and paper rate figures | Implemented in `paper_replication_benchmark.py` |
| Exact paper LDPC matrices and finite-length BER | Not reproducible from the papers |

The papers omit their degree distribution, graph constructor, seeds, matrix rank policy and any finite-length rate backoff. The current operational code therefore uses:

- the node-perspective conversion of Liveris et al.'s rate-1/2 irregular profile;
- an independent deterministic balanced/four-cycle-suppressing constructor, `balanced_4cycle_v1`;
- full-rank rejection and configurable rate margin.

These choices must not be attributed to Zhang–Tao. Exact paper replication currently means the rate/LLR methodology, not the reported finite-LDPC BER.

The core replaces the paper-specific Gaussian-CEO prior with `ConditionalPrior`, because the downstream learned quantizer supplies its own decoder-valid prior. Outer pickle/gzip serialization, self-contained plan transmission, staged quantizer symbols and oracle rate searches are repository integration choices, not paper methods.

The core encoder is not oracle: `encode_syndromes` receives only symbols and an agreed plan. The cheating behavior is confined to `OracleSWRoundMixin`, which uses current decoder-only information and known source symbols to trial-decode and select a rate on the vector being transmitted.

There is no neural decoder in this module.  The non-oracle rate controllers live beside it in
`practical_sw_mixin.py`: see `HANDOFF_PRACTICAL.md` for the fixed-rate pilot controller and why a
satisfied syndrome is not proof of a correct decode, and `HANDOFF_LDPCA.md` for the rate-adaptive
ladder.  LDPCA is Varodayan, Aaron and Girod, *Rate-adaptive distributed source coding using
low-density parity-check codes*, Signal Processing 86(11), 2006; the nesting rule that turns one
level into the next is this module's own and is not theirs.
