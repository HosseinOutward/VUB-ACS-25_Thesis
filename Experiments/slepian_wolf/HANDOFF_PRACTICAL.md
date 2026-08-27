# Practical (non-oracle) Slepian–Wolf handoff

Reference for `PracticalSWRoundMixin` in
`FL_code/cancer_protocol/bin_coder/practical_sw_mixin.py` and the additions it needed in
`sw_coder.py`. Companion to `HANDOFF_IMPLEMENTATION.md` (the core), `HANDOFF_INTEGRATION.md` (the
oracle mixin) and `HANDOFF_LDPCA.md` (the rate-adaptive transport that supersedes this one on
every measurement so far, and which lives in the same module).

## What makes it non-oracle

`OracleSWRoundMixin` picks its rate by decoding the very vector it is about to send, against the
true symbols. `PracticalSWRoundMixin` never does: every rate is decided at the decoder, from
either the prior alone or from a pilot the decoder already holds losslessly, and is then spent on
positions that took no part in the decision.

Per round, in wire order (both ends run in one process, but every bit is counted):

| step | direction | content |
|---|---|---|
| 1 | up | preprocessing metadata, already in the ordinary payload |
| 2 | down | pilot plan from `expected_plane_rates` + `bootstrap_margin` |
| 3 | up | pilot syndromes + one hash per block; failed blocks resent raw |
| 4 | down | bulk plan: per-plane rows that decoded the pilot exactly, + `safety_rows` |
| 5 | up | bulk syndromes + hashes; failed blocks resent raw |

`pilot_blocks=0` keeps only steps 1, 2 and 5: that is the paper's one-shot Algorithm 1 controller
and the A/B that says what the pilot is worth.

One seeded permutation picks the pilot and interleaves the bulk, so a block is a sample of the
whole vector rather than one contiguous slab of one layer. On iid synthetic data that is a no-op;
on real deltas it is what lets a single per-plane rate serve every block.

## What the core gained

* `expected_plane_rates` — the learned-prior form of Zhang and Tao's Eq. (45): every plane's model
  conditional entropy with no symbol observed. Lower bits of a stage are enumerated exactly and
  weighted by their own probability; earlier stages are drawn from the model rather than
  enumerated, so cost is linear, not exponential, in the stage index.
* `plane_rows` — the one place a rate becomes a row count. `SWCoderConfig.rate_margin` is gone:
  a margin is a planner's decision and no longer hides inside `decode_syndromes`.
* `calibrate_plane_rows` — the per-plane search over a parallel-built candidate grid, run on
  symbols the caller already holds. The grid always ends at the full rate, so an uncalibratable
  plane ends up uncompressed and visibly so.
* `decode_syndromes(..., recovery, verify)` — a decoder can now ask for the raw bits of a block it
  could not recover, inside the plane loop, so a failure never reaches the next plane's prior.

## Two things that were wrong

**1. A satisfied syndrome does not mean a correct decode.** Near the conditional entropy, belief
propagation regularly converges on a different word with the same syndrome. Measured on a
synthetic staged source (60 blocks, 3 stages): at a margin of 0.20 bit/plane *every* syndrome
checked out and one block was still wrong, and at 0.05–0.10 there were 3–4 such blocks. Repair
alone therefore cannot make a decode exact, and the earlier oracle path had no way to notice.
The transport now sends `verification_bits` (default 16) of a random GF(2) hash per coded block,
the same role the CRC plays in Varodayan et al.'s rate-adaptive design. It costs
`planes × verification_bits / block_length` — 0.02 bit/position at 3 planes — and is what makes
"decoded symbols equal the source" a checkable claim rather than a hope.

**2. A plan built from the model's own entropy is systematically short.** `next_plan` used the
plane entropies `E[H(q)]`, but what a syndrome code has to carry is the cross-entropy
`E[-log q(u)]` of the true symbols. On the synthetic Gaussian run the quantizer's prior was
over-confident by **17%** (model entropy 1.009 vs prior cross-entropy 1.185 bit/position), and no
fixed margin repairs that: the one-shot controller planned below the source's own entropy and
needed 183 raw block repairs to stay exact. The pilot measures the gap directly — it read 1.178
against a true 1.185 — which is the single largest thing the pilot buys.

## Measured on synthetic Gaussian data

`python Experiments/slepian_wolf/practical_sw_benchmark.py --bins 2 --stages 3` and
`--bins 4 --stages 2 --reconst-ld 160`, 1M positions, block 2400, 20 pilot blocks (4.8%), all
methods verified to decode every symbol. Rates are bits per original position, uplink, including
hashes, repairs and framing; the oracle is run at `target_ber=1e-5` so it sits at the same
distortion and the comparison is pure rate.

| config | seed | prior cross-entropy | model entropy | pilot measured | oracle | one-shot | pilot |
|---|---|---|---|---|---|---|---|
| 2 bins x 3 stages | 0* | 1.1854 | 1.0101 | 1.1776 | 1.8300 | 1.6698 | **1.6451** |
| 2 bins x 3 stages | 1 | 1.2977 | 1.1040 | 1.2927 | 1.9205 | 1.8317 | **1.7430** |
| 2 bins x 3 stages | 2 | 0.7252 | 0.5410 | 0.7171 | 1.4676 | 1.2027 | **1.0536** |
| 4 bins x 2 stages | 0* | 1.5539 | 1.3086 | 1.5501 | 2.7712 | 2.3568 | **2.1731** |
| 4 bins x 2 stages | 1 | 1.5003 | 1.3057 | 1.4914 | 2.2466 | 2.2024 | **2.0357** |
| 4 bins x 2 stages | 2 | 1.4585 | 1.2340 | 1.4535 | 2.4105 | 2.1591 | **2.0397** |

`*` seed 0 is the tuning seed; 1 and 2 are held out. Raw indices cost 3 and 4 bit/position, the
empirical marginal 2.85-2.93 and 3.85-3.96.

Read across one row:

* **model entropy under-reads the rate the source actually needs by 13-25%** and the pilot
  measures that back to within 0.2-1.1% of the truth. That is the one thing the pilot buys, and it
  is worth far more than the 4.8% of positions it costs.
* the **pilot variant beats the cheating oracle by 9-28%** on every row. That is not a paradox:
  the oracle searches a single common row backoff for the whole vector at a strict BER target,
  while the practical transport rates each plane separately and repairs the few blocks that fail.
  An oracle for one transport does not bound a different transport -- the honest floor to compare
  against is the prior cross-entropy, not the oracle.
* the practical variants decode **every symbol exactly**; the oracle leaves 9-17 wrong symbols.
* what is left over the floor is **+34% to +45%**, and it is almost all the LDPC finite-length gap
  (0.09-0.13 bit per plane, present in the oracle too) plus the raw-repair cost. See the section
  below for the two things that would actually move it.

The calibration gap is a property of how well the prior is fitted, not a constant. Repeating
seed 1 with the 180-epoch training that `brent_quantizer_check_paper` uses instead of the
benchmark's 60 narrows it from 14.9% to 9.0% (2x3) and from 13.0% to 11.2% (4x2) -- and the pilot
still lands 11.8% and 14.3% under the oracle. So a longer-trained prior shrinks the gap without
closing it, and nothing available to the encoder says which case it is in.

## Tuning, and how it was chosen

Swept on seed 0 and then confirmed on held-out seeds, because a default picked on the vector it is
reported for is the oracle habit this whole module exists to avoid.

* `safety_rows` 24 / 72 / 120 / 168 / 240 → total 1.682 / **1.648** / 1.669 / 1.688 / 1.752. The
  curve is shallow between 72 and 120 and it is a genuine trade: rows are cheap, a repaired block
  costs a full bit per position over its 2400 positions.
* `bootstrap_margin` 0.12 / **0.20** / 0.30 / 0.40 → pilot total 1.682 / 1.645 / 1.662 / 1.674,
  and 0.20 is the first value at which no pilot block needed repair.
* `pilot_blocks` 10 / 20 / 40 / 80 → 1.616 / 1.648 / 1.646 / 1.748 at `bootstrap_margin=0.12`.
  A bigger pilot does *not* buy a better calibration in net: at a mis-rated bootstrap margin most
  of its extra blocks need repair (41 of 80 at `pilot_blocks=80`). Fix the margin first; the pilot
  is nearly free either way, because its data is payload and not overhead.

## Not done here: the LDPC code family

Every variant above, oracle included, pays roughly 0.09–0.13 bit *per plane* over the conditional
entropy, and repairing a failed block costs a whole bit per position for that block. Both are
properties of the code, not of the rate controller.

* **Rate-adaptive LDPCA instead of one matrix per rate — now done**, in `sw_coder.py` and
  `LDPCASWRoundMixin`; see `HANDOFF_LDPCA.md`. It is the one change that removed both costs at
  once, because a decoder that can ask for a few hundred more rows never has to guess a margin and
  never has to resend a whole block.
* **A real constructor.** `_/slepian_wolf/ldpc_peg/{generate.m,ProgressiveEdgeGrowthACE.m}` is a
  seeded PEG+ACE builder (Hu, Eleftheriou & Arnold, IEEE Trans. IT 51(1), 2005; ACE: Tian et al.,
  IEEE Trans. Comm. 52(8), 2004), MATLAB-only — port it or run it once offline. It also lifts
  today's ceiling on block length: `balanced_4cycle_v1` plus a dense GF(2) rank check is what pins
  the codec at n=2400, and the finite-length gap shrinks as n grows.
* **Rate-matched degree profiles.** The Liveris rate-1/2 profile (Liveris, Xiong & Georghiades,
  IEEE Comm. Letters 6(10), 2002) is currently used at every rate, from 0.05 to 0.95; per-rate
  profiles from density evolution or EXIT charts are the standard fix.
* Any of it can be measured on the BSC harness already in `paper_replication_benchmark.py`, which
  maps known entropies to reliable fixed rates.

Note that the last two now matter *more* than they did, not less: with the rate controller no
longer the binding constraint, what is left over the prior cross-entropy is almost entirely the
code family.

## Still mimicked, not deployed

The encoder calls the decoder's half of the protocol in-process and counts the bits. Two
consequences worth remembering: the round needs three uplinks and two downlinks plus one repair
exchange per plane, which a deployment would have to batch or replace with LDPCA; and the whole
codec runs on the host, because the encoder's simulated decode and the receiver's rerun have to
agree bitwise about which blocks failed, which CUDA reductions do not guarantee near the decoding
threshold. `_RepairReplay` asserts loudly if they ever disagree.
