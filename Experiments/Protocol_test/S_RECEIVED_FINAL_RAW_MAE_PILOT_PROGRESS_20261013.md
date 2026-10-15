# Received-prefix final raw-MAE pilot

## Question and matched design

Test whether changing only the final-S reconstruction objective from MSE to raw-domain MAE improves full-vector/full-SI rANS rate-distortion, while holding the normalized-MAE acquisition fit fixed. The fixed, predeclared fit seeds are 43, 44, and 45; all three will be reported without seed selection.

The candidate uses normalized-MAE acquisition, 3.6M acquired coordinates, 140 acquisition-refiner epochs, 28 final-S epochs, received-prefix bins, and a frozen final-S encoder. The new `sample_final_distortion="raw_mae"` setting applies only to the final received-prefix S fit. The acquisition/previous codec explicitly retains MSE final distortion. The final raw-MAE weights use the received packet's normalization scales and the mean absolute value of its decoded teacher sample; no current-target labels or metadata are used.

## Plan and references

- Plan: `plans/s_received_final_raw_mae_pilot_20261013.json`
- Output: `_/experiments/s_received_final_raw_mae_pilot_20261013`
- Target and starting state: `_/experiments/frozen_starting_svhn_seed43_rtx3090_20261007`
- Case IDs: `current_e1_mae_received_prefix_bins_final_raw_mae_seed43`, `current_e1_mae_received_prefix_bins_final_raw_mae_seed44`, and `current_e1_mae_received_prefix_bins_final_raw_mae_seed45`
- Matched normalized-MAE control: read-only `current_e1_mae_received_prefix_bins` in `s_received_mae_five_seed_20261012`
- O and old `S_e1`/MSE: read-only end-to-end context in `s_dedicated_refiner_coldfit_five_seed_20261011`; its final fit used a trainable encoder and reencode-teacher labels, so it is not a causal control.

The primary report will average cumulative rates and raw WMAPE across the same three seeds before dB conversion and RD comparison. It will include the full directed full-SI rANS matrix, stage tables, paired-seed gaps, sample WMAPE and acquisition costs as FYI, and nonmonotonicity/symbol-error flags. Acquired sample costs remain charged in the whole-vector rates. No best-three summary or new baseline fits are planned.

Before reporting the objective comparison, the postprocessor verifies equal acquisition-state hashes, coordinates, decoded sample symbol bytes, all packet normalization/outlier metadata, and recorded sample WMAPE/rate/overhead/metadata fields. The previous normalized-MAE outputs preserve their acquired sample object inside `payload.bin`; it was verified readable for seeds 43–45. Raw serialized gzip/pickle wrapper bytes may differ despite identical decoded symbols and metadata, so the report will state that wrapper-byte equality is not claimed. It also verifies the final candidate's recorded raw-MAE provenance and baseline fit-target identity.

## Outcome

The detached unit `fl-wz-s-received-final-raw-mae-pilot-20261013` completed all three cases with return code 0, three record rows, no error markers, and idle GPUs. The full report is `_/experiments/s_received_final_raw_mae_pilot_20261013/final_analysis/matched_rd_comparison.md`. Postprocessor plus plane-sweep tests pass (40 tests); codec tests pass (66, from codec owner). Curves are monotone and there are no symbol errors or unrankable pairs.

On the untrimmed seed-43–45 cohort, final raw-MAE is 1.561% worse than the matched normalized-MAE final-MSE control in full-SI rANS RD; the recomputed reverse entry is −1.537%. Per-seed directed gaps are +1.442%, +1.478%, and +1.760%. The raw-averaged stage WMAPE and cumulative full-SI rANS values are shown in the report. Sample WMAPE is identical by seed, averaging 33.602171%; full-vector rates charge all identical sample symbols and costs.

For every seed, acquisition/refiner hashes, sample indices, decoded coded-symbol bytes, normalization/outlier metadata, charged sample rate/overhead/metadata, and wrapper length match the normalized-MAE control. The final full-S encoder hash is identical across seeds and both final objectives. Serialized wrappers are not byte-identical: equal-size gzip/pickle containers include a gzip timestamp and a PyTorch tensor storage-key string that changes with allocated storage. No wire-length or semantic packet difference was found.

## Next controlled pilot (prepared; not yet launched)

The predeclared next comparison scales only the acquisition refiner's existing `reconst_ld`, leaving normalized raw-MAE acquisition loss, baseline-derived raw denominator/scales, final-MSE S fit, and sample fit settings unchanged. The experiment-only plan field records the multiplier, base `reconst_ld`, and effective value; the ordinary final S codec retains its base `reconst_ld`. The worker regression verifies the distinction.

- `plans/s_received_raw_mae_acq_ld05_seed43_20261013.json`: seed 43, multiplier 0.5.
- `plans/s_received_raw_mae_acq_ld075_seed43_20261013.json`: seed 43, multiplier 0.75.

Both use the same retained seed-43 reference, 3.6M sample, 140 acquisition epochs, 28 final epochs, received-prefix bins, and frozen encoder. They are intended to run sequentially, not concurrently, and are not launched yet.
