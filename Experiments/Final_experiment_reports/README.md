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
