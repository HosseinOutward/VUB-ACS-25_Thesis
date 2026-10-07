CUDA precision is selected at execution boundaries, independently of AMP (`mixed_precision`).
Constructing an FL model or an RNN does not change process precision.

`Trace` version 3 uses `NumericalPolicy(float32_matmul_precision="high",
cudnn_allow_tf32=True, cudnn_deterministic=True, cudnn_benchmark=False)` for setup
and each client update. This preserves the previous CUDA FL TF32 policy. The scope
ends before any trace yield. Trace identities and continuation artifacts include
the policy, PyTorch/CUDA/cuDNN versions and GPU, invalidating older cache matches.
Distributed FL workers also establish the TF32-enabled policy at their entry points.

`WZcfgQuant.numerical_policy` explicitly controls quantizer and conditional-prior
fitting and neural inference. Its default uses `high` matmul precision and enables
cuDNN TF32. To reproduce TF32-off computation, pass:

```python
NumericalPolicy(float32_matmul_precision="highest", cudnn_allow_tf32=False)
```

The scopes restore the caller's matmul precision, cuDNN TF32/determinism/benchmark
and FP16/BF16 reduction flags, including on exceptions. Matmul `allow_tf32` aliases
`float32_matmul_precision`: restoring the latter also restores the former without
collapsing a saved `medium` mode to `high`. These are process-wide PyTorch settings;
different policies must run sequentially or in separate processes.

Protocol snapshots and compression records include the effective runtime.
Quantizer training provenance is separate from evaluation and prior fine-tuning;
model artifacts include both quantizer and native-prior provenance. A cache producer
must include `cfg.numerical_policy.runtime(device, mixed_precision=cfg.mixed_precision)`
alongside data, weights, configuration and seed in its explicit `CacheEntry` match.
Imported baseline weights retain their source artifact's provenance; an unrecorded
historical setting remains unknown rather than being assigned the importer’s policy.

Run `.venv/bin/python -m pytest FL_code/FL_core/test_numerical_policy.py -q` for
restoration, cache and same-GPU reproducibility checks. CUDA checks compare all 17
epoch weight hashes, final loss, symbols, reconstructions and three stage prior rates
for both TF32 policies and two seeds. The small O fixture bypasses candidate screening
while running the real warm fitting loop; production sampling, batches, epochs and
candidate selection are unchanged.
