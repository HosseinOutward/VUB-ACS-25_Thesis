# Common setup, and what these reports do not settle

Shared by every report in this folder.

The preserved records, resolved configurations and rerun arguments are indexed in the
[replay manifest](results/run_manifest.md).

## The measurement

Wyner-Ziv here means side information the encoder cannot see. Two kinds exist: a worker's **own**
past reconstructions (temporal prediction, not distributed) and the **other workers'** (genuinely
distributed). The `T` round conditions on its own 5, `O` on all 25, so the quantity measured
throughout is **how much O beats T**.

Common replay: rounds 12-20, 5 workers, one training round at 12 then 8 frozen,
`--history-source raw_deltas`, bins=3, planes=3. Rate is the retrained conditional prior rate,
distortion is stage WMAPE in dB, and a gap is the horizontal distance between two RD curves.

**Noise floor 0.0061 bits** — [run8](results/records/run8_longf_o_seed44/compression_records.csv),
O retrained at seed 44 in a separate cache directory. A
separate directory is required because the cache keys on content, not on seed.

**Only 5 independent training events per run.** Rounds 14-20 reuse round 12's model, so 45
client-rounds is not 45 samples. This is the binding limit on every number.

## Scope

These test **one implementation** of cross-worker conditioning: the other workers' past
reconstructions supplied as extra columns of a per-coordinate side-information vector, read by the
RNN quantizer and by the retrained conditional prior, at one operating point. Every negative result
is a result about that, not about distributed source coding in general.

Untested, and any could behave differently:

- **Conditioning on something other than past reconstruction values** — above all the sparsity mask,
  where workers demonstrably do agree.
- **Other presentations of the same columns** — the model reads the 20 cross-worker columns as 20
  fixed slots although workers are exchangeable; pooled or permutation-invariant conditioning is a
  different mechanism even if a linear probe says it carries no more information.
- **Other architectures or operating points** — one RNN, one bins/planes setting, one schedule.
- **Cross-worker information used anywhere but per coordinate** — block, layer or whole-vector
  conditioning were not tried.

## Standing caveats

O vs T confounds the SI set with `can_decode_where`, which also changes which ledger the frozen
rounds refresh from. The no-side-information baseline uses hyperparameters tuned for the WZ setting,
so its 401% is an upper bound. Probes are least-squares, hence lower bounds on what a network could
extract. Per-coordinate effects must be rate-weighted: `fc.weight` gains 0.745 bits/coordinate but is
0.046% of the model and supplies 1.4% of O's gain.
