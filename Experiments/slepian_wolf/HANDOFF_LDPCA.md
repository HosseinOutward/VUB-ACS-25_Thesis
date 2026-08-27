# Rate-adaptive LDPCA handoff

Reference for the ladder half of `FL_code/cancer_protocol/bin_coder/sw_coder.py` and for
`LDPCASWRoundMixin` in `practical_sw_mixin.py`. Companion to `HANDOFF_IMPLEMENTATION.md` (the
fixed-rate core) and `HANDOFF_PRACTICAL.md` (the fixed-rate non-oracle controller it supersedes).

## The idea

Every fixed-rate transport in this repository has to answer one question before it sends a bit:
*what rate will be enough for the worst block in the vector?* It cannot know, so it guesses high
(`bootstrap_margin`, `safety_rows`) and resends whatever still fails at a full bit per position.

LDPCA removes the question. Varodayan, Aaron and Girod (*Rate-adaptive distributed source coding
using low-density parity-check codes*, Signal Processing 86(11), 2006) encode against a single
rate-one mother code and transmit the **accumulated** syndrome. Any nested subset of accumulated
positions differences back into the syndrome of a *merged* code whose checks are sums of the
mother's, so the decoder can ask for one more bit at a time and each block stops the moment it
decodes. The rate stops being a plan and becomes a measurement, taken per block.

## The three pieces

**The nesting rule.** `split_boundaries(n, level)` splits the widest interval in half, leftmost of
equals first, which makes level `k + 1` a refinement of level `k` for every `k`. A rung therefore
costs one bit instead of a jump to the next rung of a coarse ladder. Yu and Sharma (IEEE Trans.
Comm. 61(9), 2013) instead choose each split to shape the merged degree distribution; this does
not, and that is the first thing to try if the merged code disappoints.

**The accumulator.** `accumulate_ladder` builds each plane's rate-one mother syndrome, permutes it
into transmission order and cumulatively XORs it; `_ladder_syndrome` differences a served prefix
back. `SWLadder` is the encoder's side of the channel and charges each block for the highest rung
it served; `SWLadderReplay` serves the recorded rungs to the receiver's rerun and raises if the
rerun ever asks for one the worker never sent.

**The climb.** `decode_ladder` starts every block of a plane at the plan's row count and adds
`SWCoderConfig.ladder_step` rows to whichever blocks have not verified, batching all still-active
blocks into one belief-propagation pass per rung. `plan.syndrome_rows` is only where the climb
starts: starting below every block's requirement costs passes and no bits, starting above one
costs that block the difference forever, so `LDPCAConfig.start_margin` defaults to zero and the
first rung is the model's own conditional entropy.

A `PlaneVerification` is **mandatory** on this path, not optional as it is for `decode_syndromes`.
Every rung asks "is this the source block?" and a satisfied syndrome cannot answer it.

## What one round costs

| | direction | content |
|---|---|---|
| 1 | up | preprocessing metadata, already in the ordinary payload |
| 2 | down | the starting plan: one row count per plane, from `expected_plane_rates` |
| 3 | up | one hash per plane and block, and the first rung of every block |
| 4 | down | one bit per still-unverified block, per rung |
| 5 | up | the rungs that bit asked for; repeat 4-5 until every block verifies |

There is no pilot, no bootstrap margin, no safety margin and no calibration search. Steps 2-4 of
`HANDOFF_PRACTICAL.md` collapse into "start at the model entropy and ask for more where it is
needed".

## Measured on synthetic Gaussian data

`python Experiments/slepian_wolf/practical_sw_benchmark.py --bins 2 --stages 3 --seed S` and
`--bins 4 --stages 2 --reconst-ld 160 --seed S`, 1M positions, block 2400, `ladder_step=24`,
32 hash bits. Rates are bits per original position, uplink, including hashes, raw blocks and
framing. Every method here decodes every symbol exactly except the oracle, which is not verified
at all and leaves 9-17 wrong symbols. The three older rows reproduce
`HANDOFF_PRACTICAL.md` to the digit, so this is the same experiment with one method added.

| config | seed | prior cross-entropy | oracle | one-shot | pilot | **ladder** | vs pilot | vs floor |
|---|---|---|---|---|---|---|---|---|
| 2 bins x 3 stages | 0 | 1.1854 | 1.8300 | 1.6698 | 1.6451 | **1.4678** | -10.8% | +23.8% |
| 2 bins x 3 stages | 1 | 1.2977 | 1.9205 | 1.8317 | 1.7430 | **1.6101** | -7.6% | +24.1% |
| 2 bins x 3 stages | 2 | 0.7252 | 1.4676 | 1.2027 | 1.0536 | **0.9011** | -14.5% | +24.2% |
| 4 bins x 2 stages | 0 | 1.5539 | 2.7712 | 2.3568 | 2.1731 | **1.8276** | -15.9% | +17.6% |
| 4 bins x 2 stages | 1 | 1.5003 | 2.2466 | 2.2024 | 2.0357 | **1.9259** | -5.4% | +28.4% |
| 4 bins x 2 stages | 2 | 1.4585 | 2.4105 | 2.1591 | 2.0397 | **1.7617** | -13.6% | +20.8% |

Mean over three seeds: the ladder is **11.0%** (2x3) and **11.6%** (4x2) under the pilot, and what
is left over the prior cross-entropy falls from **+39%** to **+23%**.

| config | seed | raw blocks | hash rejections | rungs climbed | downlink | ladder s | pilot s |
|---|---|---|---|---|---|---|---|
| 2 bins x 3 stages | 0 | 0 | 4611 | 18642 | 0.0187 | 669 | 43 |
| 2 bins x 3 stages | 1 | 0 | 6177 | 20608 | 0.0207 | 716 | 39 |
| 2 bins x 3 stages | 2 | 3 | 1644 | 14277 | 0.0143 | 514 | 44 |
| 4 bins x 2 stages | 0 | 4 | 3424 | 20517 | 0.0206 | 713 | 50 |
| 4 bins x 2 stages | 1 | 8 | 8545 | 24369 | 0.0244 | 749 | 49 |
| 4 bins x 2 stages | 2 | 0 | 3571 | 21123 | 0.0212 | 580 | 50 |

Three things in that table are worth keeping:

**The win is the spread across blocks, not a better code.** At a fixed rate the merged code is the
*weaker* of the two families at these rates (see `HANDOFF_IMPLEMENTATION.md`); the ladder wins
anyway. Seed 0 at 4 bins is the clearest case: the four planes settled at mean levels
1274 / 50 / 1856 / 1052 out of 2400, and the plane whose model entropy was 0.003 bit -- so most of
its blocks stopped at rung 8 -- still had one block that needed 1736. A single per-plane rate has
to serve that block; a ladder does not.

**32 hash bits, not 16.** A ladder tests one candidate word per rung, and 1644-8545 of those
candidates per run satisfied their syndrome while differing from the source. At 16 bits that is
0.03-0.13 expected undetected errors *per run*; at 32 it is under `1e-5`, and the encoder asserts
the whole decode against the true symbols so a collision would be loud rather than silent. The
hashes cost 0.013 bit/position per plane.

**It is roughly 14x slower than the fixed-rate transport.** 15-20 belief-propagation passes per
block instead of one, run twice because the worker simulates the decoder before the receiver runs
it for real. `ladder_step=72` recovers most of that at a cost of about 0.011 bit per plane.

## Tuning, and how it was chosen

Swept on a BSC at `p=0.11` (`H=0.4999`, close to the per-plane operating point), 240k positions,
so that no quantizer or prior is in the loop. Rate shown is syndrome plus hashes.

* **block length** 1200 / **2400** / 4800 / 9600 → 0.6263 / 0.5947 / **0.5845** / 0.5918. LDPCA has
  no dense GF(2) rank check, so it lifts the `n=2400` ceiling that `HANDOFF_PRACTICAL.md` blamed on
  `balanced_4cycle_v1`; 4800 is genuinely better and 9600 is not, because longer blocks buy code
  performance and lose per-block adaptation. The headline table stays at 2400 to match every other
  method.
* **`ladder_step`** 8 / **24** / 72 → 0.5900 / 0.5947 / 0.6060, at 25s / 9s / 3s. Pure
  compute-for-rate.
* **`max_bp_iterations`** 100 / 200 / 400 → 0.5814 / 0.5805 / 0.5802 syndrome-only. Not a lever.
* **`girth_candidates`** 8 / 24 → 0.5814 / 0.6047 syndrome-only. Suppressing four-cycles harder in
  the *mother* makes the *merged* code worse, which is the clue that the merged code's cycles are
  what costs it.
* **`damping`** 0.9 / 1.0 → 0.5814 / 0.5825 syndrome-only. Keep the default.

Against a fixed rate chosen so that every one of 200 BSC blocks decodes exactly, with the
purpose-built `balanced_4cycle_v1` matrix:

| BSC | H | ladder gap | fixed-rate gap |
|---|---|---|---|
| p=0.05 | 0.2864 | +0.106 | +0.220 |
| p=0.11 | 0.4999 | **+0.081** | +0.240 |
| p=0.20 | 0.7219 | +0.055 | +0.160 |

## Known rough edges

* **The terminal rung costs twice.** A block that reaches level `block_length` and still fails is
  resent whole, so it pays `2n`. It happened 0-8 times out of 834-1251 blocks per run (0-0.010
  bit/position) and is reported as `sw_raw_blocks`, never hidden. The clean fix is to solve the
  rate-one system directly -- the merged code at level `n` is rank `n - 1`, so there are two
  candidates and the hash picks one -- which is not worth the GF(2) elimination until it matters.
* A plane whose *model* entropy already reaches one bit starts at rate one and would pay `2n` on
  every block that fails there. It has not happened on this data. The fixed-rate path's uncoded
  shortcut is the answer if it ever does.
* **Both ends still run in one process.** The worker calls the decoder's half to discover the
  levels, and everything is on the host, because the two runs have to agree bitwise about which
  blocks verified and CUDA reductions do not guarantee that near the decoding threshold.
  `SWLadderReplay` raises loudly if they ever disagree, and `decode_staged_indices` checks that the
  rerun climbed exactly as many rungs as the worker recorded.
* **What is left over the floor is now almost entirely the code family**, not the rate controller.
  The next levers are the two `HANDOFF_PRACTICAL.md` already names -- a PEG+ACE constructor and
  per-rate degree profiles -- plus Yu and Sharma's splitting rule for choosing *where* to split.
