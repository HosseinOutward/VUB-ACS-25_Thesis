# Experiment reports

One folder per investigation. Each folder has its own `README.md` indexing its parts.

Naming: `YYYY-MM_topic/`.

### 2026-10 — Oracle protocol stability and multi-round alternatives
`2026-10_oracle_protocol_stability/`

Stable WZ optimizer defaults restored usable OR quantizers. Complete OR/TR windows favor OR;
multi-round OMR/TMR and fallback-selection diagnostics did not establish a decisive incremental
win over simpler OR. The handoff separates complete 1-trained+9-frozen protocol averages from
partial/common-history diagnostics, documents frozen-prior failures and acceptance/retry limits,
and preserves experimental source snapshots and compact evidence. Large run data stay gitignored
in `Experiments/Protocol_test/Runs_exp/`.

---

### 2026-09 — Warm-start retraining studies
`2026-09_warm_start_retraining/`

Archive of the synthetic shifted-side-information study and the SVHN/ResNet-18
gradient warm-start development studies. Includes source/configuration, compact
reports and per-repeat evidence; excludes traces, checkpoints, caches, logs, and
abandoned run versions.

---

### 2026-09 — Sparse initial side information
`2026-09_sparse_initial_si/`

**How do O and T behave when their five-round initial SI is raw versus classically quantized?** Runs
30-client ResNet-18/SVHN windows starting at rounds 5, 10, and 20 with three learned binary planes
and the schedule `(codec, F×6)`.

---

### 2026-08 — Cross-worker side information
`2026-08_cross_worker_side_information/`

**Is the distributed half of the codec earning its keep?** Prices what the *other workers'* past
updates add over a worker's own, and tries six ways to widen it. Verdict: worth ~5% as currently
implemented; six attempts to widen it landed at or below zero. The sparsity mask is the one lead
left open.

---

### 2026-08 — Index-aware multi-round codec
`2026-08_index_aware_multi_round_codec/`

**Can exact coordinate embeddings plus lagged training make the oracle WZ codec the best curve?**
Exact identity and shared router gradients work, but their training-round gain reverses when frozen.
OMI beats TMI by 0.02457 bit yet ties T and loses to O; plain lag-1 OM is best by 0.00879 bit over O.
Giving OM/TM raw-source local history at the encoder also loses about 0.012 bit to matched controls.
