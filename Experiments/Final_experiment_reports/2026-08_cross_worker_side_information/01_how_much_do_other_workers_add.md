# How much do other workers' past updates add over a worker's own?

Setup and scope: [00_setup.md](00_setup.md).

---

## T1 — O against T: the measurement itself
[run5_longf_o_elem](results/records/run5_longf_o_elem/compression_records.csv) vs
[run7_longf_t_elem](results/records/run7_longf_t_elem/compression_records.csv)

**Idea.** T conditions on the worker's own 5 past reconstructions, O on all 25. They are otherwise
identical, so their gap *is* the value of the distributed side information.

**How.** Same replay, same schedule, same hyperparameters. Gap = horizontal distance between the two
RD curves, pooled over all 45 client-rounds.

**Result.** O beats T by **+5.12%** (0.027 bits), about 8x the noise floor.

**Verdict.** Real, reproducible, and thin for 5x the conditioning width.

---

## T2 — No side information at all
[run9_longf_n_nosi](results/records/run9_longf_n_nosi/compression_records.csv)

**Idea.** Establish what the pipeline costs without any past data, so the gain has a real baseline.

**How.** New [`N_RoundCodec`](experiment/experiment_codecs/research_round_codecs.py): the quantizer
trains marginal, and the prior is denied server SI via
`retrained_prior_uses_server_si=False`, with `refresh_frozen_si=False` so the frozen rounds cannot
smuggle it back. Built because the recorded `marginal_rate` is *not* a valid no-WZ baseline — it
prices bins an SI-trained quantizer produced, which are deliberately ambiguous.

**Result.** N is **+401%** worse than T. At matched distortion, 4-8x more rate.

**Verdict.** Conditional coding against past data is overwhelmingly justified. Almost all of it comes
from the worker's own past, not from the other workers.

---

## T3 — Which half of the pipeline the gain comes from
derived from [run9](results/records/run9_longf_n_nosi/compression_records.csv) and
[run5](results/records/run5_longf_o_elem/compression_records.csv)

**Idea.** The pipeline has two stages that could be responsible: the SI-aware quantizer, and the
conditional entropy coder.

**How.** Three rates at matched distortion — N's quantizer with a marginal prior, O's bins with a
marginal prior, O's bins with the conditional prior.

**Result.** At 21% wmape: 1.818 / 1.701 / 0.447 bits. The quantizer contributes 6.5%, the conditional
prior takes it to 75.4%.

**Verdict.** About 91% of the gain is the entropy coder. WZ binning without a conditional prior is
nearly worthless — awkward, since CLAUDE.md treats the entropy coder as "not the topic".

---

## T4 — Seed noise floor
[run8_longf_o_seed44](results/records/run8_longf_o_seed44/compression_records.csv)

**Idea.** Nothing above is interpretable without knowing what two identical runs differ by.

**How.** O retrained at seed 44 in a *separate cache directory* — required, because the cache keys on
content, not on seed.

**Result.** **0.0061 bits.**

**Verdict.** The threshold every later result is judged against.

---

## T5 — Does a frozen quantizer go stale, or does its side information?
[run10_longf_o_rawsi](results/records/run10_longf_o_rawsi/compression_records.csv)

**Idea.** O's lead grows over the 8 frozen rounds. Either the model ages, or the side information it
reads degrades as reconstructions replace raw deltas in the history.

**How.** Re-run O with `--replace-commits-with-raw`, which pins the SI window to raw deltas so it
cannot degrade. Compare each run against *its own* round 12, since the two diverge from client 1
onward.

**Result.** RD cost of 8 frozen rounds: **-0.013 bits** with drifting SI, **-0.036** with SI pinned.
Both *improve*.

**Verdict.** The frozen quantizer does not go stale — rate falls faster than distortion rises. What
degrades is SI quality, worth ~0.024 bits over 8 rounds. Distortion feeds back into future rate, so
reconstruction quality compounds.
