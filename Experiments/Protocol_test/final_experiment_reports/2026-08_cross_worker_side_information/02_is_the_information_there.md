# Is the information in other workers' updates there at all?

Setup and scope: [00_setup.md](00_setup.md). Four probes on the raw deltas. CPU only, no GPU time — each was run
*before* the codec experiments it bears on, to decide whether they were worth the hours.

---

## P1 — Correlation structure
**Idea.** Before asking whether the model exploits the other workers, ask how related they are.

**How.** Per parameter tensor, correlation of a worker's current delta with its own past, and with
the other workers' — measured across rounds 12, 16, 20 to see the trend.

**Result.** Temporal **0.83-0.98**. Cross-worker **0.14-0.61**, and decaying **0.28 -> 0.08** over
rounds. `conv1.weight` cross-worker is **-0.04**.

**Verdict.** A worker's own past is far more informative, and the cross-worker relationship weakens
as training proceeds.

---

## P2 — How much variance each conditioning scheme explains
**Idea.** Put a number on what the 20 extra columns could contribute at best.

**How.** Least-squares on 300k coordinates predicting a worker's current delta, from: own history
alone; own plus cross-worker mean; own plus cross mean and std; all 25 raw columns; and with a
previously decoded neighbour added.

**Result.** Own history **90.19%**. All 25 columns **90.25%**. Pooled mean/std adds exactly nothing;
the neighbour adds nothing.

**Verdict.** The other workers add **0.06%** explanatory power. Also: their 20 raw columns carry
nothing their two summary statistics miss, so the model is spending input width on nothing.

---

## P3 — Do they predict the conditional *scale*?
**Idea.** An entropy coder does not need the other workers to move the prediction, only to say how
uncertain it is — rate is driven by conditional spread. A scheme could be useless for the mean and
valuable for the variance.

**How.** Predict `log|residual|` after own-history prediction, from own-history scale cues alone
versus plus cross-worker dispersion. Convert the incremental gain to bits by binning coordinates on
predicted scale and comparing weighted `0.5*log2(var)`.

**Result.** Uncontrolled, coordinates where workers disagree showed **109x** larger residuals.
Controlled for own-history scale cues: **+0.002 / +0.028 / -0.112** bits across three rounds.

**Verdict.** No reliable gain — inconsistent in sign. The 109x was a proxy for "this coordinate is
large", which the own past already knows. Worth recording as a trap.

---

## P4 — Is the codec even beating a free guess?
**Idea.** A sanity check on the whole learned apparatus.

**How.** WMAPE of zero-bit predictors — copy last round, mean of last 5, per-tensor linear fit on own
past — against the codec's stage curve.

**Result.** Copy last round 41.2%, linear fit **36.5%**, both at 0 bits. Codec: **30.7%** at 0.17
bits, **10.2%** at 0.89.

**Verdict.** The codec clearly beats free extrapolation. But stage 1 spends 0.17 bits to move
36.5% -> 30.7%, so it is largely re-deriving what prediction gives away — an argument for an explicit
predictive front end.
