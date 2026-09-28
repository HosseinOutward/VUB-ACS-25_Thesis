# Pre-run settings verification

Verified directly against `Experiments/brent_quantizer_check_paper/quantizer_check.py` on
2026-09-23. The runner also repeats this comparison at every config load and aborts on drift.

Preserved exactly:

- Gaussian generation order, `DATA_SIZE=2,000,000`, `NOISE_POWER=0.01`, and seed `repeat × 42`;
- production `WZCancerProtocol`, reconstruction history, `simulate_compression`, full-vector
  normalization, hard-symbol stage reconstruction, and production prior/marginal estimates;
- architectures `2 bins × 3 planes, λ=120` and `4 bins × 2 planes, λ=160`;
- 180 epochs, 200,000 samples/epoch, batch 1,000, LR 0.003, LR step 40, tau 1.0,
  one quantizer candidate, shared heads, hard prior context, categorical probabilities, fused
  optimizer, hidden dimension 100, and 3 layers;
- independent held-out generation and parent/candidate evaluation on identical held-out tensors.

Declared unavoidable/requested differences:

- 12 independent repeats instead of Brent's 2, retaining its seed rule;
- Oracle (`O`) only, matching the Brent command's default codec selection;
- shifted retraining/evaluation SI is the corresponding Brent SI plus exactly `0.1`;
- LR/tau differ only in the named requested variants: LR `0.0003`, tau `0.1`, or both;
- trained parents are cached and checkpointed to disk;
- the parent gets one additional frozen production evaluation on the same held-out target with
  unshifted SI, followed by shifted SI, to measure whether the perturbation is large enough;
- raw rows include production timing fields, failures, and rejection status; WMAPE-dB curves use
  `Experiments/Protocol_test/rd_metrics.py` in addition to retaining MSE and all Brent rate fields.

Bad convergence is declared before running as an invalid/non-overlapping complete hard-symbol
curve or a trained-prior equal-distortion rate gap greater than 0.05 bit/symbol relative to the
paired full-scratch curve. Failures, incomplete curves, and no-overlap rejections remain separate
counts and are never silently discarded.
