# Does sparsification change the picture?

Setup and scope: [00_setup.md](00_setup.md).

---

## P5 — Does sparsification raise cross-worker correlation?
**Idea.** Workers might send the *same* important coordinates, which would be exactly the resource
the distributed side lacks. Checked on the raw deltas before spending GPU on it.

**How.** Simulate top-k with error feedback over rounds 7-20, then measure both the correlation of
the *values* sent and the overlap of the *support* — which coordinates were chosen — against what
overlap would be by chance.

**Result.** Value correlation **falls** (0.283 -> 0.198 at round 12). Support overlap is **0.21
against 0.053 by chance — 4x** — and unlike value correlation it does **not decay** (0.213 / 0.207 /
0.205 across rounds 12 / 16 / 20).

**Verdict.** My stated hypothesis was wrong about values and right about support. The *which* is
shared, the *what* is not. Naming which 10% of 11.18M coordinates are live costs `H(0.1) = 0.469`
bits/coordinate coded blind — over half the entire 0.89-bit budget.

---

## T11 — Top-k sparsification with error feedback
[run24_longf_sparse10_o](results/records/run24_longf_sparse10_o/compression_records.csv),
[run25_longf_sparse10_t](results/records/run25_longf_sparse10_t/compression_records.csv)

**Idea.** Code the vector the way sparse FL actually transmits it, and see whether the O-vs-T gap
widens.

**How.** `--sparsify-keep 0.1` in the
[report replay](experiment/replay.py): `effective = raw + residual`, send top-k by
magnitude, `residual = effective - sent`, carried per worker across rounds. Verified
`sent + residual == effective` exactly, exactly k coordinates sent, and keep=1.0 a no-op. A separate
`sparsification_records.csv` carries error against the *raw* delta, since sparsifying changes
`w_mean_of_vec` and makes the in-record WMAPE incomparable.

**Result.** O-vs-T rises to **+13.02%**. But raw wmape is 74% against the dense codec's 10.2%, and
64% of the update norm is held back in the residual.

**Verdict.** **Not a controlled comparison** — error feedback couples each codec's own error into
what it sends next, so from round 13 O and T are compressing different vectors. The +13% cannot be
claimed.

---

## T12 — One shared support for the whole cohort
[run26_longf_shared10_o](results/records/run26_longf_shared10_o/compression_records.csv),
[run27_longf_shared10_t](results/records/run27_longf_shared10_t/compression_records.csv)

**Idea.** Force the support to be *identical* across workers, so the mask stops being per-worker
information and every worker's side information lines up coordinate for coordinate.

**How.** `--sparsify-shared-support` unions the cohort's top-k, settled before any worker is coded so
the mask cannot depend on what earlier workers transmitted. Run at keep=0.025, whose union of 0.090
matches T11's 0.10 transmitted. Measured union sizes first: 2.51x per-worker k at keep=0.1, against
5.00x if workers chose disjointly.

**Result.** **Broken.** Raw wmape 1789% (O) and 604% (T).

**Verdict.** With 92.8% zeros, **13 of 62 parameter slices lose their normalization scale entirely**.
A sparse vector cannot be fed to a codec whose per-slice normalization assumes dense input. My own
earlier `scale == 0 -> 1.0` fallback turned a loud crash into silent garbage — the fix should be to
normalize on the non-zero coordinates only.
