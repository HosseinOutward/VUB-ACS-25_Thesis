# Index-aware multi-round codec — August 2026

**Can an exact coordinate embedding plus lagged training make the oracle WZ codec the best curve?**

The experiment gives every coordinate a decoder-known identity, routes that identity to a shared
embedding, and appends the embedding to the side information seen by the encoder, decoder and
in-model prior. Multi-round variants add aligned earlier target/SI problems. The primary final
comparison is O, T, OM, TM, OMI and TMI over the same 45 client-rounds; a later sparse multi-start
extension tests raw local history at the OM/TM encoder.

Read [00_setup.md](00_setup.md) first. `M` is a structural measurement or existing-record analysis;
`T` is a GPU codec run. Negative matched-distortion rate gap is better.

---

### [00_setup.md](00_setup.md) — common setup, controls and scope

### [01_can_exact_index_conditioning_improve_rd.md](01_can_exact_index_conditioning_improve_rd.md) — can exact coordinate conditioning improve RD?

| | test | verdict |
|---|---|---|
| M1 | Exact nearby-coordinate identity | identity is exact and the router can distinguish neighbors; precision was not the failure |
| T1 | First-generation OI and TI | large training gains reverse to **+0.114/+0.082 bit** full losses |
| T2 | First-generation OMI and TMI | lagged training reduces, but does not remove, the loss |
| T3 | Final expanded OMI and TMI | OMI beats TMI by **0.02457 bit**, but ties T and loses to OM |

### [02_does_multi_round_training_generalize.md](02_does_multi_round_training_generalize.md) — does lag-1 training generalize after freezing?

| | test | verdict |
|---|---|---|
| T4 | Non-index OM against O | **OM wins by 0.00879 bit** and is the best final curve |
| T5 | Non-index TM against T | TM loses by **0.01466 bit** |
| T6 | Training round against frozen rounds | indexed gains reverse after freezing; the failure is generalization |

### [03_which_training_changes_made_it_stable.md](03_which_training_changes_made_it_stable.md) — which integration and training changes made it stable?

| | test | verdict |
|---|---|---|
| T7 | Initial expanded router | TMI loses **0.12924 bit** on its three-row probe |
| T8 | Raw index bits downstream | opens a direct memorization path; rejected |
| T9 | Bounded router-only lag-1 probes | client-0 OMI looked on par, but did not predict five-client generalization |
| T10 | Two attempts plus constant-plane rejection | removes zero-rate collapse; does not remove frozen RD loss |
| M2 | Replacement O/T reproducibility | O within noise; T shifts **−0.00772 bit**, just above the O seed floor |

### [04_does_raw_local_history_help_the_encoder.md](04_does_raw_local_history_help_the_encoder.md) — does raw local history help the encoder?

| | test | verdict |
|---|---|---|
| M3 | Duplicate-repeat stability | full-curve spread is at most **0.00049 bit**; FE losses are not rerun noise |
| T11 | Local history at the OM encoder | OM_FE loses **+0.01233 bit** to OM |
| T12 | Local history at the TM encoder | TM_FE loses **+0.01213 bit** to TM |

---

## Verdict

Exact coordinate conditioning works mechanically and OMI is decisively better than the non-WZ TMI
control: **−0.02457 bit** at matched distortion, 4.03 times the established 0.0061-bit noise floor.
It is not the flagship curve. OMI is **+0.00787 bit worse than O** and **+0.01666 bit worse than OM**;
against T it is a noise-level **−0.00043 bit** tie. The plain lag-1 oracle OM is best, beating O by
**0.00879 bit** and T by **0.01706 bit**.

The indexed models are excellent on the five training rows and bad on the forty frozen rows. OMI
beats OM by 0.01903 bit during training, then loses by 0.02100 bit after freezing. More capacity,
exact coordinates and a shared learned router therefore increased adaptation, not reusable coding
quality.

Giving the ordinary OM/TM encoder its requesting client's raw-source history also fails. OM_FE and
TM_FE lose `0.01233` and `0.01213 bit` respectively to matched controls, with the loss already present
on training rows and unchanged after freezing. Decoder-known local history should therefore remain
decoder-only in this implementation.

Still open: a matched-capacity OM control with OMI's 75/25 sampler, and a schedule that retrains or
regularizes the coordinate adapter across rounds. OMI remains an oracle study codec, so even a
future win would be an upper-bound result rather than a deployable worker-side protocol.

## Evidence and reproduction

The [run manifest](results/run_manifest.md) links every retained final and exploratory record and
gives current rerun commands. Compact derived evidence is under [results/analysis](results/analysis/):
`rd_gaps.csv`, `rd_curves.csv`, `archive_rd_gaps.csv` and `baseline_reproduction.csv`. Regenerate it
with [analyse_rd.py](experiment/analyse_rd.py). The complete experimental package and its tests are
under [experiment/index_aware_codec](experiment/index_aware_codec/).

The FE extension, including all aggregate/per-case records and logs, is under
[results/sparse_encoder_history](results/sparse_encoder_history/). Regenerate its tables and plots
with [analyse_sparse_encoder_history.py](experiment/analyse_sparse_encoder_history.py).

This report is the canonical owner of all its records; shared replay and RD helpers live in
`Experiments/Protocol_test/` without compatibility result links.

## Reproducibility gaps

- Final OMI/TMI and OM/TM are not pure index-only ablations: indexed models use 112×8 networks,
  180 epochs and weighted block sampling; ordinary references use 74×6, 140 epochs and uniform
  sampling. This tests the proposed package, not the isolated value of the index embedding.
- The raw-bits-downstream TMI collapse survives only in its external quantizer cache and resolved
  config; no compression row was emitted before that run was stopped.
- Final records were produced before the experiment package moved into this report. Their manifests
  intentionally preserve the original source path and source-tree hash; reruns use the new wrapper
  documented in the manifest.
- The 0.0061-bit floor is the O seed-44 measurement retained by the preceding cross-worker study;
  there is no independent seed sweep for every indexed configuration.
- Sparse `raw_deltas` history names its source, not its retained precision: the replay converts each
  float32 gradient to float16 on history commit. FE therefore tests raw-source selection after that
  shared conversion, not bit-exact float32 local history.
