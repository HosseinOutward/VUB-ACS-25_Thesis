# Pre-run settings verification

Verified on 2026-09-24 against `Experiments/Protocol_test/sparse_replay.py`, its seed-44 SVHN
plans, production `WZCancerProtocol`, and the registered `OracleRansRoundCodec`.

- Trace: deterministic identity FL, SVHN/ResNet-18, seed 42069, five total clients, 23 rounds,
  otherwise matching the Protocol-test SVHN configuration.
- Measurement: only client 0 is compressed; all five clients still train and advance the raw FL
  trajectory.
- Placement: parent OR training at round 12; rounds 13–21 have no codec calls; frozen parent and
  retraining comparisons use the exact raw round-22 gradient.
- SI: raw gradients from rounds 10–11 at parent training and rounds 20–21 at evaluation, for all
  five clients: exactly ten decoder-visible columns in both cases.
- Quantizer: production defaults with 2 bins × 3 stages. The parent and full scratch reference use
  120 epochs; reduced scratch and naïve warm start use 30 epochs. Normalization uses the real
  `StateDictManager.get_slices()` layer partition from Protocol-test rather than the earlier
  synthetic experiment's single full-vector slice.
- Entropy coding: production OR transport with 2,048 rANS lanes. Reported rate is cumulative
  `coder_stage_rates`, including its stage-zero fixed overhead. Distortion is hard-symbol stage
  WMAPE in dB.
- Repeats: twelve paired round-22 fits on the same trace, with training seeds `42069 + repeat`.
  The cached round-12 parent is fixed across repeats. It retains the production default of two
  candidate fits; every round-22 condition uses exactly one candidate so initialization is the
  only scratch-versus-warm difference.
- RD comparison: `Experiments/Protocol_test/rd_metrics.py`, piecewise-linear equal-distortion rate
  gap against the paired 120-epoch scratch curve. A missing overlap or gap above 0.05 bit is a
  declared bad convergence.

The unavoidable difference from stock `sparse_replay.py` is the deliberately non-contiguous codec
schedule: the FL trajectory advances through skipped rounds, but protocol history is explicitly
reseeded from raw rounds 20–21 before the round-22 operation.
