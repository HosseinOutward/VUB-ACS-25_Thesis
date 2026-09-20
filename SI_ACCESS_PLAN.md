# Plan: make retained history cheap to *read*, not just cheap to *hold*

Follow-on to `HISTORY_MIGRATION_PLAN.md`. That plan solved storage size. This one solves the
access pattern that size bought.

Revision note: this replaces an earlier draft whose ordering was *flat format -> memoize matrix ->
maybe a prefetcher*. The measurements below moved two things: the dominant cost is not the decode
kernel but four **full-vector dense materializations** per client-round that the earlier draft did
not list at all, and the transcode it proposed is 4.5x more expensive than it assumed. §6 records
what changed and why.

## 0. Status

R1, R2 and R3 landed together, not as the three commits §6 sequences: the lazy view, the flat
device format and the per-client-round column cache are one change in `wz_quantizer.py`.

R4 was not built. R5 was not built either — `_stack_training_pairs` keeps its dense block behind
a size assertion that names `extra_training_problems()`, which still ships `()`.

The grouping shift §4 R1 warns about did **not** appear on `plans/smoke.json`: every rate and
distortion column matches HEAD exactly, and only serialized sizes move, by the ~0.1% pickle jitter
`[[serialized-size-pickle-jitter]]` already documents. The quantizer cache was therefore left in
its `quantizers` directory. At 150 columns the shift is still possible, so a study that reuses a
populated quantizer cache across this change should bump its `cache_dir`.

## 1. What was measured

Two sources.

**(a) Access-pattern trace, exact.** `SideInfoMatrix.__getitem__` instrumented over one real
`DedupedWZQuantizerCancer` cycle (train -> encode -> training prior -> retrained prior -> decode),
600k coordinates, 6 SI columns, 1 candidate attempt. Rows decoded per call site, as a multiple of
the vector:

| call site | requests | rows decoded | x vector |
|---|---|---|---|
| `wz_quantizer.py:711` `_fit_prepared`, `side_info[:vector_size]` | 1 | 600,000 | **1.00** |
| `wz_quantizer.py:791` candidate check, `side_info[:vector_size]` | 1 (per attempt) | 600,000 | **1.00** |
| `prior_code.py:242` `side_info[representatives]` | 1 | 599,991 | **1.00** |
| `wz_quantizer.py:1083` deduped decode, `side_info[representatives]` | 1 | 599,991 | **1.00** |
| `wz_quantizer.py:335` training epoch draw | 6 | 300,000 | 0.50 |
| collision fixups | 2 | 18 | 0.00 |
| | | | **4.5x total** |

Four of those are single calls returning one dense `(rows, si_count)` float32 tensor. At the
`resnet18_svhn_30c_or_tr_f9_v2` shape — 11,181,642 coordinates, 150 SI columns — that is
**6.25 GiB each**, and `quantizer_train_repeats=2` makes the candidate check two to four of them.
The deduped sites allocate the gather *and* hold the source, so they peak at twice their result.

**(b) Cost of the current format**, synthetic ledger at the real shape (11,181,642 coordinates,
60 normalization slices, 11 chunks/column, layer-scale heterogeneity), measured at 20 columns on
the RTX 3090 and scaled by column count, because every cost is per column:

| quantity | at 20 cols | at 150 cols |
|---|---|---|
| host ledger | 433 MiB, 2.03 B/value | 3.17 GiB (real trace: 1.80 B/value -> 2.81 GiB) |
| `SideInfoMatrix` VRAM | 0.46 GiB | 3.48 GiB |
| matrix upload | 0.12 s | 0.9 s, **x4 per O client-round** |
| random draw, 550k rows | 53 ms | **396 ms** |
| one contiguous 100k slice | 17.4 ms | 131 ms (**x112 for a full sweep**) |
| one full-vector code-key pass | 336 ms | 2.5 s (`unique_row_groups` runs two) |
| one training epoch, RNN only (88 x 6250, si=150) | — | 1.68 s (earlier measurement) |

So: a 550k-row epoch draw is 24% on top of the epoch it feeds, and every full-vector consumer pays
~8 s of decode *and* a 6.25 GiB allocation.

Storage is not the problem. 2.81 GiB resident for the whole ledger is near the floor.

## 2. Why it is slow

`SideInfoMatrix._decode_column` (`wz_quantizer.py:128`) walks chunks inside a walk over columns,
selecting rows with a boolean mask:

```python
for chunk in column.chunks:
    selected = (rows >= chunk.start) & (rows < chunk.stop)
    if selected.any():                                   # <- device sync
        decoded[selected] = chunk.decode_rows(rows[selected] - chunk.start)   # <- nonzero, sync
```

1. **~3300 device syncs per gather.** `if selected.any()` synchronizes; `rows[selected]` runs
   `nonzero`, whose output size must reach the host — another sync. 150 columns x 11 chunks. A
   contiguous 100k slice touches at most two chunks and still pays all 1650 mask evaluations.
2. **Boolean `index_put` instead of a gather**, which is far more expensive than writing a
   gathered result.
3. **Row-invariant work repeated per column**: `self._slice_ids[rows]` and the factor gathers are
   recomputed identically 150 times.
4. **`torch.stack` of 150 tensors** copies the whole output a second time.

Attribution at 20 columns / 4 chunks, 25.0 ms total: masks + `.any()` 3.8 ms, per-column
normalization gathers 3.3 ms, `stack` 3.1 ms, the actual packed decode **0.8 ms**. The decode is
~3% of the cost; the rest is dispatch.

The adaptive per-chunk layout (`raw` / `direct` / `packed`) is what forces the Python loop: each
chunk has its own dictionary and its own code width, so there is no single indexable array.

## 3. Why the callers are the bigger half

The compressed matrix is 3.48 GiB. Every one of the four sites in §1(a) converts it straight back
into the 6.25 GiB dense matrix `HISTORY_MIGRATION_PLAN.md` §C3 was written to remove. They do it
because `batch_loop` already batches at `INFERENCE_BATCH = 100_000`, but the *selection* happens
one level above it, and selection is eager:

- `side_info[:self.vector_size]` picks the leading stacked block and returns it dense.
- `side_info[representatives]` picks the deduplicated rows and returns them dense, so the
  compressed form survives exactly as long as it takes to undo it.

Nothing here needs the whole matrix at once. Each consumer feeds a 100k-row RNN batch. What is
missing is a *row selection that stays lazy* until `batch_loop` asks for a batch — the "dataloader"
in the request, minus the threads. That single change removes the 6.25 GiB peaks, removes ~8 s of
decode per site, and is a precondition for any prefetching being worth measuring.

`_stack_training_pairs` (`wz_quantizer.py:738`) has the same dense allocation and is genuinely
dormant (`extra_training_problems()` ships `()`), which is why it is last here, not first.

## 4. The plan

### R1 — a lazy row selection (do this first)

Add one small view type and use it at the four sites:

```python
matrix.rows(index)   # -> _SideInfoRows(matrix, index), O(1), no decode
```

`_SideInfoRows` exposes what consumers already use — `shape`, `device`, `dtype`, `to`,
`__getitem__`, `code_rows`, `iter_code_columns` — and decodes `matrix[index[start:stop]]` only when
`batch_loop` asks. `__getitem__` stays eager: that is what `batch_loop` needs at the leaf.

Sites: `_fit_prepared:711`, `_train_finite_candidate_and_retrieve_best:791`,
`DedupedPriorCalculator.compute_prior_from_network:242,246`,
`DedupedWZQuantizerCancer.decode_stages_prepared:1083,1087`.

Effect, independent of R2: peak drops from 6.25 GiB (12.5 GiB on the deduped sites) to
`100_000 x 150 x 4 B = 60 MiB` per live batch. Decode time is unchanged — the same rows are still
decoded, just batched.

**One behavioral change, and it is not free.** `_fit_prepared:711` and `:791` currently hand
`unique_row_groups` a *dense* block, which groups on a 24-bit float prefix; a view routes the same
call to `iter_code_columns`, which groups on storage codes plus chunk and slice identity. Code
grouping is exact but strictly finer: it never merges equal values living in different chunks, and
it never merges two stored values whose affine-normalized float32 results collide. Group counts —
and therefore `training_prior_stage_rates`, a cached training output — can shift. The quantizer
cache key deliberately excludes SI contents, so **old entries stay live and would disagree with
fresh ones**. Land R1 as its own commit, report group counts and rates before/after on a smoke
case, and if they differ, bump the study's `cache_dir` rather than mixing the two.

### R2 — separate the storage format from the device access format

Host keeps today's adaptive packing unchanged; 1.80 B/value is worth having and nothing in §1
blames it.

`SideInfoMatrix` stops holding `_CompressedHistoryChunk` objects and transcodes each column once,
on the GPU, into a flat random-access form:

- one flat `int16` array of chunk-local ranks covering the whole vector,
- the column's per-chunk dictionaries concatenated, plus a per-chunk offset vector,
- chunk id is `rows >> _CHUNK_SHIFT` — chunks are uniform `_CHUNK_ELEMENTS`, so no `bucketize`,
  no mask, no loop. Derive the shift from the constant; do not hardcode 20.

Decode becomes `dictionary[(ranks[rows].int() & 0xFFFF) + offsets[rows >> shift]]`: three gathers,
zero syncs, one Python iteration per column instead of per column per chunk. Row-invariant work is
hoisted out of the column loop and columns are written into a preallocated `(rows, columns)` output
instead of being stacked.

Measured against a prototype of exactly this, 20 columns, RTX 3090, scaled to 150:

| | current | flat | speedup |
|---|---|---|---|
| 550k random rows | 396 ms | **57 ms** | 7.0x |
| one contiguous 100k slice | 131 ms | **20 ms** | 6.5x |
| one full-vector code-key pass | 2.52 s | **0.25 s** | 10x |
| VRAM | 3.48 GiB | 3.23 GiB flat + dictionaries | ~parity |
| decoded values | — | **bit-identical**, max abs diff 0.0, boundary rows included | |
| code keys | — | **equal**, so grouping is unchanged | |
| build | 0.9 s upload | 0.9 s upload + **6.3 s transcode** | |

Two implementation constraints the earlier draft missed, both found by building it:

- **`uint16` is not indexable on CUDA** in this torch (`"index_cuda" not implemented for
  'UInt16'`). Store `int16` and mask with `& 0xFFFF` on gather. Same bytes, no kernel.
- **Per-column dtype, not a global assert.** A `raw` chunk has no dictionary — its ranks are
  identity over `1 << 20` positions and never fit 16 bits — and a chunk whose prefixed dictionary
  exceeds 65536 entries does not either (measured `code_bits` was 14-15, but that is data, not a
  guarantee). Pick `int16` per column when every chunk fits and `int32` otherwise. An experiment at
  a new model shape must degrade to 4 B/value, not crash mid-run.

### R3 — transcode once per client-round

6.3 s of transcode makes R2 a loss if it is paid four times per client-round (25 s), so R2 and R3
land together or R3 first. An O round builds four matrices over the *same* SI vectors; only
`metadata.norm_factors` differs, and that is a cheap per-slice affine applied at decode time.

Split the object: `SideInfoColumns` holds the transcoded, target-independent payload;
`SideInfoMatrix` becomes `(columns, target factors)` and stays the only thing consumers see.

**Scope the cache to the client-round, not the codec.** The earlier draft proposed holding the
columns on the codec because `last_trained_codec` pins one codec per client — that is exactly
backwards. `cancer_protocol.py:42` keeps a dict of one codec **per client**, so a codec-lifetime
cache pins 30 x 3.23 GiB. `simulate_compression` (`codec.py:434`) is one client-round: create,
train, encode, decode, `after_decode`. Add a `BaseRoundCodec.release_round_resources()` no-op
called there in a `finally`, overridden by `_WZRoundCodec` to drop the columns. A two-entry cache
keyed by the SI tuple's identity (holding a strong reference, so ids cannot be reused) covers the
`train(A) / encode(A) / prior(B) / decode(A)` order that `_prior_side_information` produces.

Saves ~3.6 s of upload and ~19 s of transcode per O client-round, and removes the duplicate
resident matrices that `[[server-si-matrix-cost]]` measured as the 13.97 GiB peak.

### R4 — prefetch, only if it still shows up

After R1-R3 the numbers to beat are: 57 ms of SI decode against a 1.68 s training epoch (3%), and
20 ms against a ~464 ms RNN inference batch (4%). A prefetcher is a two-slot buffer, a side stream
and a `stream.wait_event` handshake for **3-4%**. Build it only if a profile after R3 still shows
the gap, and put it behind the `_SideInfoRows` iterator so both the epoch draw and `batch_loop`
get it from one place.

Do **not** take the variant that moves the flat ranks to pinned host memory to reclaim VRAM. The
transfer is cheap, but the gather that feeds it is not: measured **431 ms of host CPU** per 550k
draw at 150 columns (warm pinned buffer, `index_select` + H2D), spent on the same CPU that
`[[dedupe-decoding-quantizer]]` measured as the training loop's actual bottleneck — 39.7 ms of a
39.93 ms step was the CPU issuing kernel launches. It would not hide; it would contend.

### R5 — the remaining guards

- `_stack_training_pairs:738` allocates `(rows * (1 + extras), si_count)` float32 — 6.7 GiB the
  moment `extra_training_problems()` returns anything. After R1 it should build a stacked
  `_SideInfoRows` instead of materializing; if that is not worth it yet, assert the size with a
  message naming the hook.
- The deduped inference paths always pay a full grouping pass. The training loop already probes a
  sample first and skips grouping below `TRAINING_DEDUPLICATION_MIN_FRACTION`; the decode and prior
  paths do not. Measure the duplicate fraction at 150 columns on a real trace before adding the
  probe — `[[dedupe-decoding-quantizer]]` found 62% duplicate rows at 25 columns, and whether that
  survives 150 columns is unknown, not assumed.

## 5. Verification

- **Equality is the gate.** `matrix[rows]` must be bit-identical to today's for random rows,
  contiguous slices, chunk-boundary rows, the final partial chunk, **all three chunk layouts**
  including `raw`, and the empty-SI case. Extend `tests/test_wz_history_consumers.py`, which
  already covers each layout. The prototype passes this at max abs diff 0.0.
- **Grouping.** `iter_code_columns` / `code_rows` must return the same keys as today (verified
  equal in the prototype). Separately report the R1 grouping shift described above — it is the one
  place results may legitimately move.
- **Cache.** `model_training_hash` excludes SI contents (`FL_core/cache.py:94`) and R2/R3 change no
  decoded value, so quantizer and prior entries stay valid. Confirm with a cache-hit replay.
- **End to end.** One `refactor_smoke` sparse-replay case, records bit-identical to HEAD except the
  documented R1 shift; then one O client-round at ResNet-18 scale reporting per-phase timings and
  `torch.cuda.max_memory_allocated()` against §1.
- **Benchmark.** `Experiments/Protocol_test/benchmark_history.py` measures commit and
  `materialize`, neither of which is the hot path. Add the three cases from §1(b) — random draw,
  contiguous sweep, code-key pass — so these numbers live in the repo instead of this document.

## 6. Sequence

| commit | content | reviewable alone |
|---|---|---|
| R1 | `_SideInfoRows` lazy selection at the four sites | yes — peak VRAM, plus the grouping report |
| R3 | `SideInfoColumns` + per-client-round release hook | yes — upload count and peak VRAM |
| R2 | flat `int16` device format, hoisted normalization, preallocated output | yes — equality tests plus §1(b) timings |
| R5 | `_stack_training_pairs` guard | yes — small, independent |
| R4 | prefetcher, only if R1-R3 leave a measurable gap | yes |

R1 first because it is the only step that changes peak memory by itself, and because it is the
step whose result may move. R3 before R2 because R2 without R3 pays its 6.3 s transcode four times.

### What changed from the previous draft

1. **The four full-vector dense materializations were missing** and are the dominant cost; the
   draft's §4 named the dormant `_stack_training_pairs` instead. Evidence: §1(a).
2. **"Prefetching is nearly pointless" was the right conclusion for the wrong reason.** What the
   callers need is laziness, not latency hiding. R1 is the requested dataloader; R4 is the thread
   pool, and it is worth 3-4%.
3. **The transcode is 6.3 s at 150 columns, not 1.4 s** (measured on the cheap path, which copies
   `direct` codes without a gather). That makes R3 a precondition rather than a follow-up.
4. **R3's lifetime argument was inverted.** One codec per client is the hazard, not the scope.
5. **`uint16` fails for two reasons the draft did not have**: no CUDA gather in this torch, and
   `raw` chunks that cannot fit 16 bits at all. Per-column dtype.
6. **The pinned-host prefetch variant counted the transfer but not the gather** — 431 ms of CPU
   per draw, on the CPU that is already the bottleneck. Dropped.
7. **Added the grouping-shift caveat**, which is the one way this work can change research results.
