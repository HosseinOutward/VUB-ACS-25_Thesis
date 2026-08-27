# Slepian–Wolf integration handoff

## Core API

```python
codec = SlepianWolfCodec(SWCoderConfig())
plan = codec.initial_plan(num_stages, alphabet)
payload = codec.encode_syndromes(bins, plan)  # encoder: no SI or prior
result = codec.decode_syndromes(payload, plan, prior_provider, side_info)
decoded_bins = result.symbols
next_plan = result.next_plan
```

`prior_provider(decoded_stages, side_info, stage)` must return probabilities shaped `(positions, alphabet)`. It may use only information available at the decoder.

## Round-codec mixin

```python
class MyCodec(OracleSWRoundMixin, MyStagedIndexCodec):
    pass
```

The wrapped codec must produce `payload_content == (bins, PreprocessMetadata)` and expose the trained quantizer used by the existing WZ codecs. The mixin replaces `bins` with `OracleSWPayload`; decoding restores bins before calling the quantizer reconstruction path.

`OracleSWRoundMixin` is deliberately cheating: it full-rate bootstraps the entropy plan, trial-decodes increasing common row backoffs, and sends the first tested plan whose one-sided 95% BER bound is below `sw_target_ber`. Exact search uses `sw_search_step=1`; experiments may use a coarser explicit step. The selection data and evaluation data are the same. Do not use this as a practical FL rate controller.

### Why it is oracle/cheating

A normal one-way SW encoder sees symbols and a previously agreed rate plan; side information, the conditional prior and decoding outcome exist only at the decoder. This mixin instead performs decoder work while encoding: it accesses the current decoder side information and learned prior, knows the true source symbols, repeatedly decodes candidate rates, measures their true BER, and chooses a passing rate on that same vector. It then transmits the selected plan. This gives an optimistic per-vector lower operational rate and bypasses the real problem of choosing a rate before the decoder observes success or failure. `PracticalSWRoundMixin` (see `HANDOFF_PRACTICAL.md`) solves that problem instead of skipping it, and measures lower than this mixin on every configuration tried, so do not read the oracle rate as a bound on what a real controller can reach.

## Optional value mapping

```python
class MyMappedCodec(
    OracleValueOrderedIndexMixin,
    OracleSWRoundMixin,
    MyStagedIndexCodec,
):
    pass
```

The mapping mixin forces each stage symbol through the quantizer decoder, orders symbols by mean reconstruction contribution and assigns that order along a Gray-code path. It permutes the prior consistently, transmits the validated mapping and reverses labels before reconstruction.

This is also oracle-only: it evaluates forced-symbol reconstructions on the current vector to design that vector's mapping. A practical version should learn a fixed mapping on calibration data and share/cache its identifier. A bijection does not change ideal conditional entropy; it can change finite-length LDPC behavior and the distortion caused by residual errors.

Use final serialized bits/value and reconstruction distortion as the primary result. BER is a reliability proxy, not the rate–distortion objective.
