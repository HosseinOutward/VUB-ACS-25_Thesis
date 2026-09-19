# Cross-worker side information — August 2026

**Is the distributed half of the codec earning its keep?**

The codec conditions on two kinds of past data: a worker's **own** past updates (temporal prediction,
not distributed) and the **other workers'** (genuinely distributed, and the only part that makes this
Wyner-Ziv). These reports price the second, and try six ways to make it larger.

`T` = probe on raw data, no GPU. `M` = measurement. `T##` = codec run. Read `00_setup.md` first.

---

### [00_setup.md](00_setup.md) — common setup, and what this study does not settle

### [01_how_much_do_other_workers_add.md](01_how_much_do_other_workers_add.md) — how much do other workers add over a worker's own?
| | test | verdict |
|---|---|---|
| T1 | O against T — the measurement itself | **+5.12%**, ~8x noise |
| T2 | No side information at all | pipeline worth **4-8x**, nearly all from own past |
| T3 | Which half of the pipeline the gain comes from | **91% is the entropy coder**, not the quantizer |
| T4 | Seed noise floor | **0.0061 bits** |
| T5 | Does a frozen quantizer go stale, or its side information? | not stale; **SI quality** degrades |

### [02_is_the_information_there.md](02_is_the_information_there.md) — is the information there at all? (probes, no GPU)
| | probe | verdict |
|---|---|---|
| P1 | Correlation structure | temporal 0.83-0.98 vs cross-worker 0.14-0.61, **decaying** |
| P2 | Variance explained by each conditioning scheme | other workers add **0.06%** |
| P3 | Do they predict the conditional *scale*? | no; the 109x effect was a trap |
| P4 | Is the codec beating a free guess? | yes, but stage 1 barely |

### [03_can_the_model_be_forced_to_use_it.md](03_can_the_model_be_forced_to_use_it.md) — can the model be forced to use them?
| | test | verdict |
|---|---|---|
| T6 | Offset training, dense (1,2,3) | worse than O |
| T7 | Offset training, gapped (1,3,5,7) | worse still — **further back is worse** |
| T8 | Blank own columns during training (0.3, 0.6) | dose-dependent damage |
| T9 | Stop the flag becoming a gate | **recovers a third** — the fix worked |
| T10 | Scramble own columns instead of blanking | **3-6x worse than blanking** |

### [04_does_sparsification_change_it.md](04_does_sparsification_change_it.md) — does sparsification change it?
| | test | verdict |
|---|---|---|
| P5 | Does sparsification raise cross-worker correlation? | values no, **support 4x above chance and non-decaying** |
| T11 | Top-k with error feedback | +13% but **not a controlled comparison** |
| T12 | One shared support for the cohort | **broken** — normalization dies on sparse vectors |

### [05_does_splitting_the_vector_help.md](05_does_splitting_the_vector_help.md) — does splitting the vector help?
| | test | verdict |
|---|---|---|
| M1 | Which coordinates use the extra side information? | **not a property of the coordinate** (r = 0.004-0.025) |
| T13 | Three quantizers on the measured partition | **t = -0.30**, no effect |
| T14 | Reassign to whichever quantizer codes best | apparent gain was **overfitting** |
| M2 | What the split actually learned | split by **magnitude** into one cheap and two accurate parts; no layer, kind or neuron structure |

---

## Verdict

Conditional coding against past reconstructions is worth 4-8x against none, but almost all of that is
the worker's *own* past. Read as per-coordinate value conditioning, the other workers are worth
**5%**, and all ten interventions landed at or below zero.

Still open: the **sparsity mask** (P5) — workers overlap 4x above chance, it does not decay, and
naming the live coordinates costs 0.469 bits/coordinate coded blind.

## Evidence and reproduction

The [run manifest](results/run_manifest.md) links every retained records CSV and records the exact
protocol and replay-only arguments. Each run directory also contains the unchanged historical FL
and resolved protocol configurations. M1's compact derived tables are under
[results/analysis/elements_o_vs_t](results/analysis/elements_o_vs_t/), and the two partitions used
by T13/T14 are under [results/partitions](results/partitions/).

The raw [run5](results/records/run5_longf_o_elem/compression_records.csv) /
[run7](results/records/run7_longf_t_elem/compression_records.csv) element dumps are retained locally
under `results/raw_elements/`. They are ignored because they total roughly 10 GiB; they were moved
intact rather than deleted.

## Reproducibility gaps

- The exploratory probe implementations and raw outputs for P1-P5 are not present. Their methods
  and aggregate findings survive only in these reports.
- The surviving split builder produces an earlier two-way diagnostic, not the
  `three_way_db.npy` partition consumed by T13. The final partition is retained, but its initial
  three-way generator is missing.
- [Run29's](results/records/run29_longf_op3_crosseval/compression_records.csv) cross-evaluation
  element dumps are absent. The retained reassigned partition and summary cannot currently be
  regenerated by `experiment/reassign_split.py` from the stored files.
- `experiment/analyse_split_structure.py` covers the worker-0 depth, decile and neuron checks, but
  does not reproduce all five-worker, layer4-conditioned and parameter-kind aggregates quoted in
  M2.
- The OD replays are historical results. Their shared own-SI flag/dropout model logic was removed;
  the experiment-local TODO describes what must be restored before those commands run again.
