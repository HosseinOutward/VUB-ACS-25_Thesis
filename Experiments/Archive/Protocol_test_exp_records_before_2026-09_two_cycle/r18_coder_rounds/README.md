# R18 coder-round benchmark — O vs T with real wire transfer

Ran 2026-08-27/28. Two datasets x four cases x 30 clients, one training round replayed at absolute
round 5 of a saved six-round ResNet-18 trace (11,181,642 coordinates per update).

Plans: `../../coder_round_plans/{eurosat64,imagenette160}.json`
Codecs and factory: `../../coder_rounds.py`
Traces: `_/data/federated_learning/r18_{eurosat64,imagenette160}_30c` (180 float32 deltas each,
built by `../../generate_sparse_trace.py`).

Reproduce: `.venv/bin/python -m Experiments.Protocol_test.sparse_replay
Experiments/Protocol_test/coder_round_plans/<dataset>.json`. Completed cases carry `complete.json`
and are skipped, so a rerun only redoes what is missing. Plots: `plot_sparse_results` runs
automatically after each case; the per-codec curves come from
`python -m Experiments.Protocol_test.plot <records.csv> --out <png>`.

## The question

`T` and `O` differ only in decoder side information — 5 client-local temporal vectors versus the
150-vector server-wide view. Comparing their *estimated* rates makes `O` look better on both axes.
The comparison that decides anything is `O_oracle_sw` vs `T_rans`: each family paying for the
conditional entropy coder it would really use.

| | eurosat64 | imagenette160 |
|---|---|---|
| **`O_oracle_sw` vs `T_rans`** rate gap at matched distortion | **+17.70%** (+0.223 b/el) | **+12.68%** (+0.171 b/el) |
| distortion gap at matched rate | +0.53 dB | +0.33 dB |

`T_rans` wins on both. The oracle SI is genuinely worth something — `O_oracle_sw` reaches -4.04 dB
at stage 1 where `T_rans` reaches -3.38 (eurosat64) — but the Slepian-Wolf transport costs more than
that advantage is worth. Roughly three quarters of its overhead is the rate plan rather than LDPC
coding (below), so this verdict is contingent on that plan, not on side information.

## Which rate is the reference

`prior_stage_rates` — the **retrained** diagnostic prior — is the correct reference.
`training_prior_stage_rates` is trained jointly with the quantizer, so it is handicapped by the main
training loop and over-reads the rate. Both shipped transports nevertheless *condition on* the
handicapped one, so an operational rate splits in two:

| | `T_rans` | `O_oracle_sw` |
|---|---|---|
| excess over the prior it codes with | +0.0078 / +0.0077 b/el | +0.583 / +0.739 b/el |
| extra because that prior is handicapped | +0.0105 / +0.0077 b/el | +0.125 / +0.013 b/el |
| **total over the retrained prior** | **+1.20% / +0.97%** | **+53.9% / +52.2%** |

(eurosat64 / imagenette160.) For rANS about half of an already-tiny gap is the prior it was handed,
not the coder. Conditioning either coder on a retrained prior is a real win, and a large one for O.

## Per case

| | eurosat64 | imagenette160 |
|---|---|---|
| `T` rate (estimated) / final WMAPE | 1.5196 / 45.50 | 1.5895 / 43.58 |
| `T_rans` operational / WMAPE / payload | 1.5379 / 45.50 / 2.49 MB | 1.6049 / 43.58 / 2.58 MB |
| `O` rate (estimated) / final WMAPE | 1.3885 / 44.50 | 1.5203 / 39.86 |
| `O_oracle_sw` operational / WMAPE / payload | 2.0955 / 46.30 / 3.10 MB | 2.2729 / 42.54 / 3.17 MB |
| `T_rans` bit-exact clients | 30/30 | 30/30 |
| `O_oracle_sw` symbol BER | 5.8e-3, 0/30 exact | 4.6e-3, 0/30 exact |

`T_rans` is verified exact: distortion is identical to `T` on every client, which is the only real
evidence available since `RansRoundMixin` records `symbol_error_rate=0.0` by assumption without
checking. It also halves what the protocol puts on the wire today (zstd-compressed raw bins).

**The O family is not a matched pair.** `T_rans` reuses `T`'s quantizer on 30/30 clients; `O_oracle_sw`
reuses `O`'s only on **client 0**. `Oracle_RoundCodec.can_decode_where` is `SERVER_ONLY`, so from
client 1 onward each client trains on the previous clients' reconstructions *from this case*, which
are transport-dependent — so the cache key diverges and the quantizer is retrained. `O_oracle_sw`
versus `O` therefore mixes transport cost with a fresh quantizer draw, and since ~8% of clients lose
a stage to collapse on different clients in each draw, that noise is large: among the four clients
whose SW payload decoded exactly, client 23 collapsed under `O` but not under `O_oracle_sw`, and
client 18 failed under `O_oracle_sw` but not under `O`. Read the `O_oracle_sw` numbers as a
system-level result, not as an isolated coder measurement. `replace_selected_with_raw: true` is the
fix for future O-family A/Bs — see `r18_osw_prior/`.

`O_oracle_sw` decodes with 4.6-5.8e-3 symbol errors — inside its 1e-2 target, but it means the
transport is lossy and costs up to 2.7 WMAPE points on top of the rate.

## Why the SW overhead is a plan artifact

Measured against the prior it conditions on, the per-stage excess is a near-constant ~0.2 b/el on
every plane: [0.153, 0.235, 0.195] and [0.206, 0.245, 0.289]. A per-plane cost proportional to
nothing is not coding loss. `OracleSWRoundMixin.encode_staged_indices` builds a plan from each
plane's own entropy and then searches a **single common row-backoff added to every plane at once**;
the hardest plane sets it and the low-entropy planes are charged the same. Stage 1 clips to
`block_length` and ships uncompressed on nearly every client. A per-plane backoff is the first thing
to try, and it is what would decide whether O can beat T at all.

## Per-client cost (seconds, 11.18M coordinates)

| | train | encode | decode | total |
|---|---|---|---|---|
| `T` | 29 | 12 | 5 | 41 |
| `T_rans` | 2 (shared cache) | 31 | 20 | 53 |
| `O` | 136 | 76 | 54 | 271 |
| `O_oracle_sw` | 132 (retrains; see above) | ~210 | ~105 | 413-465 |

Envelope: ~4.9 GB host anon and ~14 GB GPU per worker at peak; two workers, one per GPU, fit in a
12 G scope each.

## Profile caveats

`bins_per_plane=2|num_planes=3|train_epochs=20|train_sample_size=200000`, one candidate each.
20 epochs is a floor, not a preference: at 5 epochs the quantizer collapses stages 2-3 on ~27% of
clients (zero rate, WMAPE stops improving); at 20 epochs that falls to ~8%. `bins_per_plane=2` is
forced by the oracle SW alphabet and is **not** re-tuned — see the `bins-per-plane-hyperparam-transfer`
note before reading absolute rates. `T`/`T_rans` run with `retrained_prior_uses_server_si=False` so
the diagnostic prior sees the same 5-vector SI the coder conditions on; `O`/`O_oracle_sw` keep the
default, which for them is the same `SERVER_ONLY` view either way.

## Other record directories here

`r18_coder_probe*` are single-client timing/memory probes. `r18_coder_rounds_interrupted_*` and
`r18_coder_rounds_crashed_osw_rank2` are abandoned attempts kept only as evidence (5-epoch collapse,
the server-SI prior cost, the rank-2 crash). None feed any aggregate; all are safe to delete.
