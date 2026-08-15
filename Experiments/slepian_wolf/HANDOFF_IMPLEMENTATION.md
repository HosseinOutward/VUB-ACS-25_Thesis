# Slepian–Wolf implementation handoff

Temporary reference for `FL_code/cancer_protocol/sw_coder.py`.

## Implemented path

Input is a `(stages, positions)` tensor of power-of-two quantizer symbols. For each stage:

1. Obtain decoder-owned `q(U_stage | decoded earlier stages, side information)`.
2. Split symbols into bit planes, least-significant bit first.
3. Condition each plane on its decoded lower planes and form source LLRs.
4. Send a packed LDPC syndrome for each block and plane.
5. Recover planes with sum-product BP, then use the recovered stage in the next-stage prior.

`SWRatePlan` completely identifies block length, rows, graph profile, constructor version and seed. Matrices are reconstructed deterministically and checked for full GF(2) row rank. `SWPayload` contains the source length, plan ID and packed syndromes. There is no CRC, retransmission or uncoded fallback.

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

There is no LDPCA backend, practical online rate controller or neural decoder in this module.
