# Oracle protocol stability and multi-round alternatives — September/October 2026

This is the self-contained handoff for branch `quantizer-stability-defaults`, closed on
2026-10-04. Read this file before the historical reports: several earlier tables deliberately
compare partial windows, and must not be interpreted as complete protocol rankings.

Final cleanup colocates plans in `plans/` and closed run outputs in `runs/`. Experimental
model/problem tensors and the single shared trace continuation checkpoint were removed at
the user's request; CSV/JSON evidence, logs and plots remain. `removed_artifacts.csv` lists
all 184 removed files and sizes (623,440,366 bytes total). These untracked tensors cannot be
recovered from Git; model re-evaluation and trace continuation require recomputation. The
shared-cache deletion was limited to `_/data/trace_states/9373b2e968ca413fd03c63aa1a4407da6dfa36947d2996d8f3ce6f120b3b1387.pt`;
datasets and other shared caches were not removed.

## Decision

Keep ordinary O/OR and T/TR, the stabilized WZ quantizer defaults, fixed-symbol prior
fine-tuning, same-symbol marginal acceptance, deterministic sparse replay and rate accounting.
Archive OM/TM/OMR/TMR, weighted historical target/SI fitting, sampled training diagnostics,
T-in-O embeddings, preserved-joint-prior alternatives and chronological selection experiments.
They produced useful evidence but no decisive incremental win over simpler OR in a controlled,
complete protocol comparison. This is a complexity/strength-of-evidence decision, not proof
that multi-round methods cannot work or that OR is universally optimal.

The completed original OR/TR comparison favors OR even without excluding failures. Frozen
conditional-prior generalization remains unresolved; extra retries and training-time acceptance
do not guarantee reliability after freezing. Every-round prior fine-tuning was discussed as the
next test but was NOT implemented or launched during this investigation.

## 1. Project and codec context

The research pipeline is learned Wyner–Ziv gradient quantization followed by conditional entropy
coding. A quantizer emits three binary symbol planes per floating-point coordinate. Its prior
estimates conditional code length; this experiment also measures actual lossless rANS bytes.
Quantization quality and entropy coding quality must be assessed separately.

O uses decoder/server-visible SI: the server's reconstruction history in normal client order,
including earlier clients' current-round reconstructions. T uses the requesting client's own
temporal history, a subset of O's available SI. OR/TR add rANS to O/T. R denotes rANS, not
retraining. F reuses the most recently trained codec: weights freeze, but SI is refreshed.
Full SI is an oracle/server-side assumption; it must not silently become a worker-available
input or a deployable worker-side prior without an availability argument.

The sparse runner measures compression on a deterministic, uncompressed FL trajectory; it
does not train a compressed FL model. The full client population advances the raw trace,
while decoded reconstructions advance each compression case's SI history. Before a sparse
window, the history is seeded from preceding raw deltas. This distinction is essential when
interpreting the results.

## 2. Setup and comparison contract

- Seed 43, SVHN, ResNet-18, all 30 clients; dataset revision
  `torchvision-svhn-train_32x32.mat-md5:e26dedcc434d2e4c54c9b2d4a06d8373`.
- Trace device remains `cuda:0` for every plan, independently of which GPU fits quantizers.
  A different trace GPU can produce a different FL trajectory. GPU numeric labels varied with
  the process/device environment; use each manifest's actual GPU provenance, not an assumed name.
- Original OR/TR starts: 5, 12, 20. Each is one trained round + nine F rounds, all 30 clients:
  300 client-rounds per start, 900 per method. Three binary planes; rANS has 2048 lanes.
- Warm protocol: clients 0–4 train cold, best of three accepted candidates, filling a five-slot
  pool; clients 5–29 warm-start from two pool parents, two accepted candidates per parent.
- Stabilized defaults: layers 4, hidden dimension 128, learning rate
  0.0028364098431326932, 140 cold epochs. Warm fits derive 35 epochs, learning rate times 0.3,
  tau 0.5. Production joint training uses 550,000 sampled coordinates/epoch and batch size 6,250.

**Default protocol result:** average the complete one-trained-plus-nine-frozen window, with
equal matched client-round weights. Average cumulative rates and raw WMAPE first, then convert
WMAPE percentages using `20*log10(mean_wmape/100)`. Compare the three-stage curves by the mean
piecewise-linear rate difference at equal distortion over their shared range, using
`Experiments/Protocol_test/rd_metrics.py`. Do not average per-round RD gap percentages.
Negative candidate-minus-reference gaps favor the candidate. Trained/frozen splits are separate
diagnostics. Partial windows are not complete protocol averages.

The requested like-for-like coder comparison is **full-SI rANS versus full-SI rANS**. Native
transport is separately labeled: OR uses full SI, TR temporal SI. Requested rate tables include
joint/pre-finetuning rate, fine-tuned temporal prior, fine-tuned full-SI prior, temporal rANS,
full-SI rANS, and WMAPE dB with raw percent in parentheses. Standard replay records store
fine-tuned priors as totals, not per-stage curves: never invent their missing stage breakdown.
The custom confirmation measured fine-tuned prior stage curves explicitly.

**Outlier sensitivity:** flag a native/full-SI rANS stage below 0.01 or above 1.05 bits. Remove
the union of flagged clients from BOTH methods at the same placement across the whole matched
window. This is retrospective sensitivity analysis, not an implemented correction. In the
confirmation matrices, exclusion is pair-specific; a failing unselected candidate must not
invalidate a healthy selected candidate. These thresholds detect symptoms, not their cause.

## 3. Chronology: what was tried and learned

### A. Optimizer collapse and retained defaults

The first OR/TR attempt was stopped because OR quantizers collapsed. Diagnostics identified
optimizer instability, not evidence that richer SI is intrinsically harmful. Former defaults
were six layers, learning rate 0.0056728196862653865 and 120 epochs. Across 36 cold attempts,
the old configuration had 10 full collapses and 16 cases with at least one dead stage;
four layers/half learning rate/140 epochs had zero of either. Selected RD was at least as good
on usable matched cases. Warm fits were usable and roughly a third of cold fitting time.

The underlying diagnostic is already archived at
`Experiments/Archive/O_collapse_diagnostic/README.md`. Its opening historical hypothesis is
superseded by its results section. Stable defaults are retained, not archived out of production.
Original stopped outputs were first moved, not deleted, to `_/tmp/Runs_exp_stopped_2026-09-29/`;
the final cleanup colocates them here under `runs/stopped_2026-09-29/`.

### B. Complete original OR/TR replay

Both plans completed all three 10-round windows under the stable defaults. They predate the
new prior-versus-marginal acceptance gate: the gate must not be credited retrospectively for
their measured RD. Original acceptance already checked finite/usable quantizer candidates.

Full-SI rANS equal-distortion gaps, OR minus TR:

| Population | Trained rounds (diagnostic) | Frozen rounds (diagnostic) | Complete 10-round overall |
|---|---:|---:|---:|
| All clients | −6.69% | −4.94% | **−5.10%** |
| Excluding flagged clients | −6.78% | −6.75% | **−6.76%** |

The overall gaps are −0.029259 and −0.037242 bits/float respectively. Native-transport overall
gaps are −8.79% without exclusions and −9.14% with them; these are NOT the full-SI comparison.

| Start | Full-SI overall, all | Full-SI overall, excluded | OR flagged clients | TR flagged clients |
|---|---:|---:|---|---|
| 5 | −8.61% | −8.61% | None | None |
| 12 | +7.85% | −5.64% | 1,13,20,21 | 1 |
| 20 | −13.82% | −6.48% | 10 | 10,15 |

Exclusion removes 60 of 900 rows per method overall. Both O and T can suffer frozen-prior
failures. Removing failures can help either method, depending on placement; it does not prove
an O-only defect. The large change at start12 is primarily frozen behavior, not trained-round RD.

### C. Legacy lag-1 OMR/TMR

OM/TM and OMR/TMR were experiment-only codecs, not the older indexed-codec study. M means
training across historical targets paired with their capture-time SI. The first generation used
the current and preceding target. It stopped before a complete matched 10-round comparison;
the archived matched analysis covers seven complete rounds 6–12 (210 rows/method).

Full-SI rANS OMR minus TMR: trained −3.63%; frozen +9.01%; partial overall +7.31%.
After symmetric removal of clients 1/5/11: trained −3.74%; frozen −5.40%; partial overall −5.18%.
OMR failed for clients 5/11, TMR for client1. Exclusions reduced the matched population to 189.
This reversal supports concentrated frozen-prior failures, not an intrinsic temporal advantage.
Neither partial overall result is a one-trained-plus-nine-frozen protocol result.

### D. Marginal gate and two historical rounds

The user requested all cold/warm quantizer and prior candidates be accepted only if the total
hard-symbol prior rate is no greater than the empirical marginal rate of those same symbols,
before the existing lowest-loss argmin. This general acceptance change is retained. Required
accepted quotas remain strict; exhaustion fails clearly, never deploying a rejected candidate.
Replay semantic identity includes the gate, so old outputs cannot be silently reused.

The M mixture was expanded to current/previous/second-previous targets with shares 50/30/20.
Each target was paired with its own capture-time SI; current SI stayed live. Both joint training
and fixed-symbol prior scratch/fine-tuning used the mixture. Historical symbols were computed
with the final encoder, not with the historical encoder. The weighted prior marginal comparison
used weighted per-round marginals, not a pooled cross-round marginal. These M-specific training
extensions have now been removed from active core code and preserved in the source snapshot.

OMR recorded 314/900 rows (start7 complete, start12 14 trained clients); TMR only 2/900 before
an actual host OOM. OMR later stopped after frozen client21 showed full-SI stages around
1.96–2.34 bits. Round11 full prior 2.6475 exceeded marginal 1.5724, while rANS total was 2.6397:
this implicated prior generalization, not exceptional coder overhead or symbol decoding errors.

The direct OMR/TMR comparison has only two trained clients: −0.65% full-SI rANS, no matched
frozen comparison. OR beat revised OMR by 1.98% on 14 trained rows at start12 (2.34% on 12
after exclusion). Neither result establishes a complete protocol winner, a general M penalty
on O, or an M benefit on T. The supplemental OR-start5/OMR-start7 frozen comparison has
different training placements/history and is not a controlled ablation.

### E. Sampled training diagnostic

Clients 0/5/11 and seeds 43/44/45 compared T, O and O initialized from an exact T-in-O embedding.
It retained joint priors and compared current-only versus multi-round fixed-symbol prior fitting.
The sample used 16,384 training and 4,096 held-out coordinates, only six steps/epoch versus
roughly 88 production steps, and raw shared histories rather than decoded warm-pool histories.
Its rANS flush overhead was about 3.21 bits/float: these rates are not production payload rates.

Multi-round prior fitting helped several sampled RD curves, but O had a severe train/held-out
gap. The diagnostic recommended full-vector confirmation and chronological selection, not
deployment of the sampled winner. See `results/training_diagnostic/runs/final_analysis.md`.

### F. Full-vector fallback confirmation and retry experiment

This used production update budgets and a real 30-client TMR warm-pool baseline at start7,
followed by nine F rounds. All candidate diagnostics shared the baseline's decoded history.
Diagnostic clients were 0/5/11/21. Candidates were T, zero-extended T-in-O, cold O trained from
scratch, and warm O initialized from T. Joint priors were retained alongside fine-tuned priors;
the T-in-O candidate also retained T's fine-tuned temporal prior. Actual 2048-lane rANS was
decoded and checked losslessly with sequential decoder-available context.

The first run failed at baseline client21: only one of four attempts passed for one warm parent,
while two accepted candidates were required. A fresh experiment increased quantizer rejection
slots from two to six: maximum warm attempts 4→8 per parent, cold 5→9. Quotas, acceptance,
prior retries and optimizer defaults were unchanged. This allowed the run to finish. The extra
retry setting is experiment-only and is NOT promoted to the ordinary OR/TR defaults.

Round8 was frozen chronological validation, used to select a prior by hard NLL and then an O
encoder by normalized-MSE-plus-native-NLL. Selection was persisted before rounds9–16. This
uses raw validation targets and oracle SI; it is not a deployable worker-only policy. It selected
cold O/fine prior for clients0/5 and T-initialized O/fine prior for clients11/21. Joint priors
were never selected. All alternatives were evaluated, not only winners.

The retry8 run completed at 2026-10-03 04:34 Brussels time, after about 15h17m: baseline
300/300 rows, validation four clients, future evaluation 4×8=32 client-rounds, 960 future
stage/policy rows. It is a common-history mechanism confirmation, NOT an independently evolving
all-client OMR/TMR protocol sweep. The diagnostic curves omit the trained round and use round8
for selection, so the 32-row result is not a 10-round protocol average.

| Frozen diagnostic comparison against T full-SI rANS | All matched clients | Symmetric outlier exclusion |
|---|---:|---:|
| Cold O with fine-tuned prior | −6.58% | −6.58% |
| T-initialized O with fine-tuned prior | +40.04% | −8.05% |
| Chronologically selected O | **−7.32%** | **−7.32%** |

The selected full-SI prior gap was −7.36%; native transport gap −9.72%. No selected candidate
had a flagged stage. Unselected T-initialized O client0 reached 3.389 bits/stage; its joint
prior also crossed 1.05. Comparisons involving those alternatives exclude client0 from both
methods across all eight future rounds, retaining 24 observations. Other curves retain 32.
Baseline TMR client14, outside the diagnostic set, reached 6.195 full-SI bits/stage and
6.415 native temporal bits/stage. There were no dead coder stages or symbol errors.

Selection avoided one failure without future-data leakage. It is promising evidence for a
safeguard, not proof of an incremental full-protocol win over ordinary OR. Additional fitting
and validation costs also prevent treating the selected curve as a free replacement for OR.

## 4. What remains active, and what not to infer

- Active: stable four-layer/half-lr/140-epoch defaults; ordinary O/OR/T/TR; five-slot warm pool;
  current-target fixed-symbol prior fine-tuning; same-symbol marginal gate; identity/provenance;
  temporal/full-SI rANS accounting; detached supervision and live plotting infrastructure.
- Active rejection allowance remains two slots. Cold/warm accepted quotas remain three/two;
  the experiment-only retry8 protocol and quota extension are archived. Prior budgets unchanged.
- Removed from active imports and core: M codec registration, historical-target runner hooks,
  weighted prior mixture parameters and codec prior-fit hooks used only by those mixtures.
- Not established: M improves T generally; M degrades O generally; OR is the best possible
  oracle architecture; marginal acceptance solves future failures; outlier exclusion is a fix.
- Established scope: complete original OR/TR windows favor OR; frozen-prior failures affect
  both families; M/selection alternatives have not shown a decisive controlled incremental win.

## 5. Archive layout, preserved data and reproducibility

`source/` mirrors the ORIGINAL repository-relative locations of experiment code/tests
and the exact supporting core/replay files as they existed before cleanup. `plans/` now owns
the archived experiment plans, original OR/TR plan snapshots and shared FL config. These are inert
historical snapshots, not active plugins. Their imports, commands, relative plan paths and
registration side effects intentionally preserve the original layout. Do not import them into
the retained runner. To reproduce, use a separate worktree at
`de470eadedef896c34984ca7fbdc3e1453b17971`, overlay the selected historical `source/` files at
their original paths and `plans/` into `Experiments/Protocol_test/plans/`, then use a fresh output
root and the archived command in that worktree. The plan JSON and stored manifests retain
their original output paths for provenance; map these to a fresh root rather than overwriting
archived results. Active ordinary OR/TR templates remain in the live plans directory. Removed
tensor artifacts must be recomputed; the source snapshot alone does not recover their contents.
Review the full support snapshot, not just `experimental_codecs.py`: weighted priors and runner
capture hooks were part of the semantics. Old lag-1 plans require their historical implementation
or explicitly matching weights; the old stopped confirmation plan is not accepted by the retry8
worker. No compatibility wrappers or silent rerun fallbacks are provided.

`results/` contains versioned copies of compact derived CSVs, final reports, manifests and
completion markers. Original derived artifacts also remain with the retained run evidence. The
copied reports contain historical commands/source paths and scope labels; this handoff owns the
final interpretation and complete-window comparison contract.

Closed run folders have been MOVED here into `runs/`, alongside their archived `plans/`.
They remain gitignored; compact evidence above is versioned. The old live `Runs_exp/` container
is empty and reserved for new experiments. Stored logs/manifests retain original paths rather
than rewriting historical provenance. Experimental tensor artifacts have been removed, along
with the one shared trace checkpoint listed above; datasets and other caches were not moved
or deleted. `runs/` retains about 19 MiB of evidence/logs/plots, locally and gitignored.

| Path under this archive's `runs/` | Contents/status |
|---|---|
| `or_svhn_30c_f9_warm/`, `tr_svhn_30c_f9_warm/` | Complete original OR/TR, 900 rows each |
| `omr_svhn_30c_f9_warm/`, `tmr_svhn_30c_f9_warm/` | Stopped lag-1 runs; partial matched audit only |
| `omr_svhn_30c_f9_w503020_gate/` | Revised OMR, 314 rows; stopped frozen-prior failures |
| `tmr_svhn_30c_f9_w503020_gate/` | Revised TMR, two rows; OOM stopped |
| `training_diagnostic/` | Completed sampled diagnostic, attempts/selection analysis; tensors removed |
| `full_vector_confirmation/` | Failed first confirmation, 21 baseline rows; checkpoints removed |
| `full_vector_confirmation_retry8/` | Successful confirmation, all baseline/evaluation results |
| `pairing_audit_2026-10-02/` | Matched historical RD audit and symmetric exclusions |
| `_global_analysis/`, `_global_analysis_control/` | Closed global plots/watcher status |
| `stopped_2026-09-29/` | Original stopped old-default outputs and plots |

Primary numerical evidence: `results/pairing_audit_2026-10-02/{stage_points,comparison_gaps,exclusions,suspect_stages}.csv`.
Confirmation evidence: `results/full_vector_confirmation_retry8/runs/`, including the full rate
table, all stage points, selections and failed alternatives. The raw `models/selection_client*.json`
files establish chronological choices; `attempts/round*.json` preserve every evaluated policy.
Final full matrices are in `results/full_vector_confirmation_retry8/full_si_comparison_matrices.md`.
See `VALIDATION.md` for closure checks.
Original manifests retain git HEAD and dirty-source fingerprints from execution; cleanup does
not rewrite them. Starting source HEAD was de470ea; the feature branch descended from local
master f792d73, through 8dad1f0, stable-default commit 4a0d772, and diagnostic archival de470ea.

## 6. Operational lessons and next justified work

For current launch commands and the exact text-table reporting contract, read
`Experiments/Protocol_test/README.md`, section "Running and reporting a new experiment".
Use a fresh ordinary OR/TR plan/output, launch through the detached supervisor, and report
full square equal-distortion percentage matrices with/without symmetric exclusions. Alongside
them include marginal, joint/pre-finetuning prior, temporal/full-SI fine-tuned prior,
native/temporal/full-SI rANS, and WMAPE dB (raw percent). Standard scalar marginal/fine-tuned
prior totals are not stage breakdowns. Frozen prior measurements mean fixed trained weights
evaluated on refreshed target/SI, not that fine-tuning happened again. Main averages always
cover the complete trained+9-frozen window; phase splits are separate diagnostics.

Run multi-hour work through the detached supervisor, never an interactive session. Keep the
live plotter enabled. It records health at 1/5/30 minutes, then every 90 minutes. Stop via the
specific user systemd unit, never `pkill`. The parallel OMR/TMR attempt caused a confirmed host
OOM despite apparently plausible RSS estimates; later confirmation ran alone. Watch actual
host memory and SwapFree. The custom worker's supervisor generic completed-case count remained
zero because it does not write the standard case marker; exit code zero, its own complete.json,
and 300 validated baseline rows establish completion.

The next justified test is a controlled complete OR/TR replay investigating per-round prior
fine-tuning or another decoder-legitimate fallback for frozen failures. Keep its added cost and
SI availability explicit, test failures on all clients/seeds, and compare complete 10-round
averages. Do not reintroduce M complexity solely because of a partial, selected, or excluded
curve. No such new experiment was authorized/launched as part of closing this branch.

The first cleanup verified 23 acceptance/plan/record tests; the user subsequently removed the
two active test files as no longer needed. Their original experimental test snapshots remain
historical evidence under `source/`. Archive imports remain removed from active replay and
historical tests excluded from routine discovery. The closure merges into
local master, deletes only the merged feature branch, and does not push or reconcile the
substantially diverged origin/master. Compact evidence is committed; large run data are not.
