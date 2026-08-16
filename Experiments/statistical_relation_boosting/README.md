# Side-information predictability experiment

This experiment asks one narrow question: how well can decoder-available side information predict
the sign of the next client delta? It reads cached uncompressed deltas and fits the same small
histogram-gradient-boosting classifier under two conditions:

- `own_history`: only the target client's previous deltas;
- `decoder_si`: every client's previous deltas, plus current-round deltas from clients the decoder
  has already reconstructed.

The difference isolates the contribution of cross-client side information. Because the input is
lossless cached history, this is an upper bound for a real lossy protocol whose decoder retains
reconstructions.

```bash
.venv/bin/python Experiments/statistical_relation_boosting/experiment/run.py \
  --start-round 12 --end-round 20 --history-rounds 3

.venv/bin/python Experiments/statistical_relation_boosting/experiment/plot.py
```

The runner writes `results/sign_predictability.csv`; the plotter writes
`results/sign_predictability.png`. `cross_entropy_bits` is the conditional rate for the three sign
symbols, and `bits_gain` is its reduction relative to the training split's marginal sign prior.

All previous result CSVs, plots, histories, learning-rate files, and logs remain unchanged in
`results/`. The former federated-training, calibration, regression analysis, plotting, README, and
report sources are preserved under
`Experiments/review_before_simplification/statistical_relation_boosting/` for review.
