# Slepian–Wolf studies

These are synthetic/BSC and quantizer-integration studies of the staged Slepian–Wolf transports.
Rates below are operational bits per original value (uplink unless stated), including the wire
overheads reported by each benchmark. They are not measurements on real FL deltas.

## Benchmarks

| benchmark (what it measures) | exact command | result file |
|---|---|---|
| `paper_replication_benchmark.py`: analytical Gaussian-CEO rates, LLR/packing checks, and optional finite-length LDPC checks | `.venv/bin/python -m pytest -q -s Experiments/slepian_wolf/paper_replication_benchmark.py` | no file; pytest output |
| `cross_paper_sw_benchmark.py`: local BP against the paper's neural SW-ECCT decoder over common BSC streams | `.venv/bin/python -m Experiments.slepian_wolf.cross_paper_sw_benchmark --positions 240000 --output Experiments/slepian_wolf/results/cross_paper.csv` | `results/cross_paper.csv`, `results/cross_paper_summary.csv` |
| `neural_sw_comparison_benchmark.py`: fixed-rate and per-block oracle neural-decoder comparison | `.venv/bin/python -m Experiments.slepian_wolf.neural_sw_comparison_benchmark --blocks 10000 --output Experiments/slepian_wolf/results/neural_sw_comparison.csv` | `results/neural_sw_comparison.csv` |
| `practical_sw_benchmark.py`: one-shot, fixed-rate pilot, rate-adaptive LDPCA, and oracle SW on staged synthetic Gaussian indices | `.venv/bin/python Experiments/slepian_wolf/practical_sw_benchmark.py --bins 2 --stages 3 --seed 0` (repeat with `--seed 1` and `--seed 2`; use `--bins 4 --stages 2 --reconst-ld 160` for the second configuration) | `results/practical_sw_{2b3s,4b2s}_seed{0,1,2}.{json,csv}` |
| `quantizer_sw_integration_benchmark.py`: original versus value/Gray-ordered oracle labels, wire rate, symbol errors and reconstruction MSE | `.venv/bin/python Experiments/slepian_wolf/quantizer_sw_integration_benchmark.py --bins 4 --stages 2 --value-ordered-mapping --target-ber 0.0001 --output Experiments/slepian_wolf/results/quantizer_sw_integration_4b2s_mapped_strict.json` | `results/quantizer_sw_integration_*.json` |

The unit tests are:

```text
.venv/bin/python -m pytest -q Experiments/Protocol_test/tests \
  Experiments/slepian_wolf/test_staged_transport.py \
  Experiments/slepian_wolf/test_oracle_sw_mixin.py \
  Experiments/slepian_wolf/test_practical_sw.py
```

## Numeric verdicts that remain valid

The practical controller's tuned defaults are `pilot_blocks=20`, `bootstrap_margin=0.20`,
`safety_rows=72`, `verification_bits=16`; the LDPCA defaults used in the study are
`block_length=2400`, `ladder_step=24`, and 32-bit ladder hashes. Across three seeds, LDPCA was
10.8%, 7.6%, 14.5% below the pilot for 2×3 stages (mean 11.0%), and 15.9%, 5.4%, 13.6% below it
for 4×2 (mean 11.6%). Its excess over prior cross-entropy was +23.8–24.2% (2×3) and +17.6–28.4%
(4×2), versus the pilot's +34.3–45.3%; the practical pilot was 9–28% below the cheating oracle,
so the oracle is not a lower bound for this different transport.

The paper checks still reproduce 0.3631 (Figure 7 worker 10; analytical 0.3621) and 0.2640
bit/symbol at the one-bit Figure 10 point (73.6% saving). In strict 4×2 quantizer integration,
value/Gray mapping changed wire rate 2.0767→2.0058 (−3.41%) and MSE 0.00303951→0.00302879
(−0.35%); at target BER 1%, it was 1.8720 wire rate and +17.0% MSE, a tradeoff. These are
single-study results; oracle selection uses the reported vector and is optimistic.

`neural_sw_comparison_benchmark.py` intentionally keeps its private `sw_coder._bp_decode` import:
it is live comparison code. Optional paper/ neural/ cross checks require their artifacts and are
skipped unless `SW_PAPER_RUN_LDPC=1`, `SW_CROSS_RUN=1`, or `SW_NEURAL_RUN=1` is set.
