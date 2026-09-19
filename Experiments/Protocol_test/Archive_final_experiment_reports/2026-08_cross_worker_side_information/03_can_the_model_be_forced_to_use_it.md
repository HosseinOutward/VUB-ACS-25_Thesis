# Can the model be forced to use the other workers' columns?

Setup and scope: [00_setup.md](00_setup.md).

The premise behind all five: the model may ignore the cross-worker columns because its own past is so
much easier to use. So either widen the training problem, or take the easy option away. All replace
the O round on the same schedule; only the quantizer is perturbed, while the prior retrains on full
server SI exactly as for O and T, so any difference is attributable to the quantizer.

Results are quoted against **T**, so positive means it still beats non-WZ. Plain O is **+4.87%**.

---

## T6 — Offset training, dense offsets (1,2,3)
[run12_longf_om_dense123](results/records/run12_longf_om_dense123/compression_records.csv)

**Idea.** Widen the training set with the round's own problem rebuilt as it stood at earlier rounds.
Payload and frozen handoff are unchanged; only the training rows grow.

**How.** [`OM_RoundCodec`](experiment/experiment_codecs/research_round_codecs.py) with
`train_offsets=3`.

**Result.** **+2.96%** vs T — i.e. **worse than plain O** by 0.010 bits.

**Verdict.** No gain; roughly noise-level damage.

---

## T7 — Offset training, gapped offsets (1,3,5,7)
[run11_longf_om_gap1357](results/records/run11_longf_om_gap1357/compression_records.csv)

**Idea.** Blocks cost memory, not time, so spreading the offsets reaches 7 rounds back for 4 blocks
instead of 7 — a longer horizon for the same cost.

**How.** `train_offsets` extended to accept an explicit sequence.

**Result.** **-0.70%** vs T — worse than plain O by 0.031 bits, and worse on frozen rounds than on
the training round, so the damage compounds.

**Verdict.** Reaching further back is monotonically worse. The delta distribution drifts, so an older
problem is not the same problem and the extra data is mislabelled. Contradicts commit `6e36558`,
which settled on OM at bins=2.

---

## T8 — Blank the worker's own columns during training
[run15_longf_od30](results/records/run15_longf_od30/compression_records.csv) (0.3),
[run16_longf_od60](results/records/run16_longf_od60/compression_records.csv) (0.6)

**Idea.** Remove the crutch on a share of training rows and the model must read the other workers.

**How.** [`OD_RoundCodec`](experiment/experiment_codecs/research_round_codecs.py). Own columns
located at `[client_id*window, +window)` in the SERVER_ONLY
view, so nothing is reordered. Blanked on training rows only — inference never blanks. The withheld
state is signalled by a learned `nn.Embedding(2,4)` rather than a raw bit, so the network can
represent it however it finds useful.

**Result.** **-0.01%** at 0.3, **-1.97%** at 0.6.

**Verdict.** Dose-dependent damage. Forcing the model off its own past does not make it find
cross-worker signal.

---

## T9 — Stop the flag becoming a gate
[run17_longf_od30_fo20](results/records/run17_longf_od30_fo20/compression_records.csv)

**Idea.** If "withheld" always coincides with blanked columns, the model can treat the flag as a gate
and stop reading those columns whenever it is set — so the flag stops being usable at inference.

**How.** `own_si_flag_only=0.2` adds rows carrying the withheld flag *while the columns are still
present*, breaking the correlation. Measured split verified at 30.5 / 19.9 / 49.6%.

**Result.** **+1.69%** vs T — recovers about a third of the dropout loss.

**Verdict.** The design intuition was right: the flag *had* been acting as a gate. A validated fix
inside a failed direction.

---

## T10 — Scramble the worker's own columns instead of blanking
[run23_longf_od_scr40](results/records/run23_longf_od_scr40/compression_records.csv) (0.4),
[run22_longf_od_scr70](results/records/run22_longf_od_scr70/compression_records.csv) (0.7)

**Idea.** Zeros are out of distribution, so a model can detect a blanked row and route around it —
blanking may be a weaker attack than it looks. Permuted values look real and carry no information.

**How.** `own_si_scramble` permutes the own columns across rows. No flag, deliberately — there is no
telltale to signal.

**Result.** **-4.17%** at 0.4, **-13.48%** at 0.7.

**Verdict.** Scrambling hurts **3-6x more than blanking**, confirming the model had been detecting
the zeros. Being genuinely misled about its own past is very expensive, and no cross-worker
substitute exists to fall back on.
