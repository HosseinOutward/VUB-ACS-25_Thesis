# Plan: one reconstruction history, compressed by default

Builds on commit "history: add optional compressed reconstruction-history storage", which added
`optimized_history.py` beside the old ledger as an opt-in feature. That opt-in shape is the problem
this plan removes.

> **Later storage revision:** the implementation now normalizes every model-layer slice before
> 18-bit prefix truncation. Factors are cached per client for five rounds and retained with each
> vector for decoder-side inversion. Chunks then choose raw prefixes, byte-aligned direct codes, or
> split-packed codes according to retained size. The older decisions below document the migration
> that preceded this revision and are not the current storage contract.

## 1. Goal

A researcher reading `FL_code/cancer_protocol/` should see one ledger with one storage format, and
be able to tell from names and file placement which part is *what history means* and which part is
*how it is stored*, without reading a configuration to find out which path a run took. No option
selects between an optimized and an unoptimized history.

End state, one module `FL_code/cancer_protocol/history.py`:

| Class | Role |
|---|---|
| `HistoryVector` | Parent. Plain float32 CPU vector. Defines the *meaning* of a history value: `numel()`, ordered `iter_chunks()`, `materialize()`. |
| `CompressedHistoryVector(HistoryVector)` | The storage optimization: per-chunk dictionary plus integer codes. Same reads, same order. |
| `HistoryEntry` | Unchanged dataclass; `value: HistoryVector`. |
| `ReconstructionHistory` | Parent. All ledger logic: access rules, per-client depth, `copy`, `replace_latest`, `view`. Retains values through one hook, `_retain(value) -> HistoryVector`. |
| `CompressedReconstructionHistory(ReconstructionHistory)` | Overrides `_retain` only. What the protocol always builds. |

The parents are never selected by a protocol option. They are the reference implementation, used
directly by small experiments that want exact history (`Taguchi_oracle_search.py`,
`quantizer_check.py`). `_WZProtocol` always builds the compressed child.

Gone: `optimized_history.py`, `HistoryStore`, `_DiskSession` and all disk support, the
`HistoryStorageConfig` switches, the `history_*` protocol options, the normalization-factor side
channel through the codecs, and the `torch.Tensor | StoredHistoryVector` unions.

## 2. Decisions (settled; not options)

1. **No configuration.** `HistoryStorageConfig` is deleted outright. Storage constants live in
   `history.py`: `_PREFIX_BITS`, `_CHUNK_ELEMENTS = 1 << 20`.
2. **Format: leading-bit prefix plus exact dedup.** Commit truncates each value with
   `float_prefix(value, _PREFIX_BITS)`, then stores each chunk as a float32 dictionary of its
   distinct values plus one integer code per coordinate (`uint8`/`uint16`/`uint32`, the narrowest
   that indexes the dictionary). Decode is `dictionary[codes]` — one gather, on CPU or GPU.
3. **No normalization.** Prefix truncation keeps the float32 exponent, so small values and layer
   scale differences are handled by construction. This deletes `_normalize_factors`,
   `_apply_normalization`, `take_history_normalization_factors`,
   `_WZRoundCodec._decoded_history_norm_factors`, the `staged_transport.py` `getattr(metadata,
   "norm_factors", None)` line, the `normalization_factor_source` snapshot field, and the
   `sd_slices` dependency of history. It also removes two failure modes: float16 overflow to `inf`
   above 65504, and `normalization_params` asserting on a constant layer.
4. **No disk.** `_write_disk`, `manifest.json`, mmap, session directories and the `history_root`
   argument through `create_protocol`, `BaseProtocol`, the server and `sparse_replay` are deleted.
   The compressed form halves host memory, and the real peak is the GPU side-information matrix,
   which §5 addresses directly.
5. **No bit packing, no escape codes, no rank checkpoints.** Codes stay byte-aligned so the same
   arrays decode on the GPU with a plain gather. `_pack_codes`, `_codes_at`, the analytic width
   search, the exception arrays and `read_range` all go.

### Evidence

One `refactor_smoke` delta, 855,770 coordinates, `|mean| = 2.7e-4`:

| representation | distinct values | code width | bytes/value |
|---|---|---|---|
| float32 | 454,103 (53.1%) | — | 4.00 |
| prefix 20 | 45,352 (5.3%) | 16 bit | ~2.05 |
| prefix 18 | 14,153 (1.7%) | 14 bit | ~1.82 |
| float16 + normalize | — | 16 bit | 2.00 |

A plain float16 cast without normalization puts 25% of these values in subnormals, so half
precision without normalization is not an option; prefix truncation is.

`_PREFIX_BITS = 18` gives a 9-bit mantissa, one bit coarser than normalized float16, at a smaller
size. Start there and confirm by RD (§6). If RD moves materially, raise the constant to 20 and
re-measure; it is a constant, not a switch.

Open: distinct counts at ResNet-18 scale, where a 1 M-element chunk covers more layers. If a chunk
exceeds 65536 distinct prefixed values the code dtype simply widens to `uint32`, so nothing breaks;
the size table just moves.

## 3. Commit sequence

Three commits, each importable and reviewable on its own.

### C1 — collapse the structure, no value change
- `git mv FL_code/cancer_protocol/optimized_history.py FL_code/cancer_protocol/history.py`.
- Move `HistoryEntry` and `ReconstructionHistory` out of `cancer_protocol.py` into it. Update
  imports in `cancer_protocol.py`, `round_codecs.py`, `sparse_replay.py`, `quantizer_check.py`,
  `Taguchi_oracle_search.py` and the benchmark. No re-export from `cancer_protocol`. The
  `TYPE_CHECKING` import in `round_codecs.py` becomes a normal import, which removes that cycle.
- Add `HistoryVector` (about 20 lines) wrapping an owned float32 CPU tensor: `numel()`,
  `iter_chunks()`, `materialize()`.
- Delete all disk code, `HistoryStorageConfig`, the `history_*` protocol options and the
  normalization side channel (§2.3, §2.4).
- `_retain` on the parent returns `HistoryVector(reconst.clone())`; the protocol still builds the
  parent here, so reconstructions, symbols and rates stay bit-identical to HEAD and the existing
  model-training cache still hits. That is what makes this commit verifiable.

### C2 — compressed storage becomes what the protocol builds
- Rewrite the storage internals as §2.2: prefix, per-chunk dictionary, byte-aligned codes. Delete
  the packing, escape and checkpoint machinery.
- `CompressedReconstructionHistory._retain` returns `CompressedHistoryVector`; `_WZProtocol` builds
  it unconditionally.
- Consumers take `HistoryVector`: replace the unions and the `HistoryValue` alias in
  `wz_quantizer.py` and `round_codecs.py`; `normalize_side_info_tensor` drops its `isinstance`
  branch and always iterates `iter_chunks()`. Plain-tensor callers wrap once at the edge
  (`bin_coder/rans_coder.py`, `bin_coder/sw_coder.py`, `extra_training_problems` overrides,
  experiment `train_model` calls).
- `sparse_replay.py`: bump `_EXPERIMENT_SEMANTICS_VERSION`, and reword the oracle-assumption text so
  it no longer describes storage as optional.
- This commit changes every SI value and therefore invalidates every cached quantizer training.
  That happens exactly once, here.

### C3 — the side-information matrix stays compressed on the GPU
Today `normalize_side_info_tensor` builds a dense `(vector_size, si_count)` float32 matrix on the
GPU: 6.7 GB for 150 vectors at 11.18 M coordinates, built up to three times per O client-round.
Chunked host reads do not touch that; they only bound the host peak while filling it.

Instead, upload each SI column in its stored form (dictionary plus codes) and decode on demand:

- A `SideInfoMatrix` object owns the per-column dictionaries and codes on the GPU and exposes what
  the training and inference code already uses: `matrix[row_indices]` and `matrix[start:stop]`
  returning float32 `(rows, si_count)`, plus `shape`, `device`, `dtype`.
- Decode per batch is `dictionary[codes[rows]]`, then the target's per-layer affine applied to the
  gathered rows. Callers (`wz_model_training_loop`, `decode_stages_prepared`, `PriorCalculator`,
  `_stack_training_pairs`) keep their current expressions and never learn where the bytes live.
- `unique_row_groups` folds the **codes** instead of `float_prefix` of decoded values. Equal codes
  mean equal values by construction, so grouping gets cheaper and stays exact.
- Expected: about 3.4 GB instead of 6.7 GB per SI matrix, and no full-matrix float32 upload.
- The target normalization moves from CPU to GPU, so SI values shift by an ulp against HEAD. That
  is acceptable only because C2 already invalidated those caches; it must land after C2, not before.

## 4. What stays as it is
- The ledger's access rules, per-client depth, `copy` sharing immutable entries, and
  `replace_latest`.
- `config_snapshot` recording the implementation qualname plus the storage constants, which is what
  replay identity reads.
- The `vector_size` assertions on SI, now through `numel()`.

## 5. The model-training cache key

Decision: the key covers the training target, not the SI contents. `cache.py` therefore loses the
`ContentHashable` protocol and `_update_typed_hash`, history vectors need no digest support at all,
and the per-client-round SI hashing cost disappears.

Two collisions have to be closed by hand, because `si_size` is derived from the SI tuple itself
(`si_size=len(side_information)`) and `c_cfg.training_cache_payload` pins `max_side_info_count: 5`
as a historical constant:

1. **T vs O, same round and client.** Both train on the same raw delta with the same `c_cfg`; only
   the SI set differs (5 temporal vectors vs 150 server ones). Keep `si_size` in the key explicitly
   so these cannot share an entry.
2. **Two cases in one plan.** `cache_root` is per plan, shared by every case, so two schedules
   replaying the same `(round, client)` see the same `x_raw` and may see the same `si_size` while
   their histories differ — transport-dependent reconstructions make this concrete. `si_size` alone
   does not separate them.

Closing (2) without hashing contents uses the case's resolved schedule identity: the protocol name,
warmup tokens, and routine tokens are carried by `training_cache_discriminator` in the quantizer
configuration. Together with the explicit `si_size`, this separates T from O and separates cases
sharing a `cache_root`, without scanning retained history values.

## 6. Verification
- **C1:** smoke sparse replay (T and O, `refactor_smoke`) is bit-identical to HEAD and gets cache
  hits from HEAD's cache.
- **C2:** parent and child agree on `iter_chunks` order and `materialize()` up to the prefix
  truncation; storage WMAPE and bytes per value reported for the real trace; RD for T and O against
  C1's exact history, as mean rate difference at equal distortion (`rd_metrics.py`). That number
  justifies `_PREFIX_BITS`.
- **C3:** decoded SI batches equal C2's dense matrix exactly; measured peak VRAM for one O
  client-round against the 13.97 GiB / 6.99 GiB baselines; per-client train/encode/decode timings
  do not regress.
- **Tests:** update `tests/test_history.py`, fold `tests/test_optimized_history.py` into it,
  `tests/test_wz_history_consumers.py`, `tests/test_codec_lifecycle.py` (its `history_*` protocol
  strings go away entirely) and `Experiments/Protocol_test/tests/test_sparse_replay.py`; run
  `pytest -q`.
- **Leftovers:** `rg -n "optimized_history|HistoryStore|HistoryValue|StoredHistoryVector|entry\.tensor|history_normalize|history_deduplicate|history_chunk_elements|history_disk|take_history_normalization|history_root" FL_code Experiments tests --glob '!**/final_experiment_reports/**'`
  returns nothing. Delete `COMPRESSED_HISTORY_IMPLEMENTATION_PLAN.md`.

## 7. Out of scope
- Sharing dictionaries across entries, approximate dedup, recovery across restarts.
- Memoizing the SI matrix so an O client-round builds it once instead of three times. Worth doing,
  but it is a codec-lifetime question (`last_trained_codec` pins one codec per client), not a
  storage one.
