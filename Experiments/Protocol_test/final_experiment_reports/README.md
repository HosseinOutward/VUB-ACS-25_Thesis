# Experiment reports

One folder per investigation. Each folder has its own `README.md` indexing its parts.

Naming: `YYYY-MM_topic/`.

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
