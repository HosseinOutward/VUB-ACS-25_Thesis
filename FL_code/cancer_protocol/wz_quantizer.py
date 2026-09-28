from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from FL_code.FL_core.cache import cache_call, model_training_hash
from FL_code.FL_core.utils import (
    OUTLIER_CHUNK,
    OutlierMetadata,
    PreprocessMetadata,
    create_training_progress_bar,
    normalization_params,
    outlier_metadata,
)

from .brent_wz_models import EncoderDecoderLayeredRNN
from .history import _CHUNK_ELEMENTS, CompressedHistoryVector, HistoryVector
from .prior_code import INFERENCE_BATCH, DedupedPriorCalculator, PriorCalculator


# Below this point, exact grouping costs more than the avoided RNN work.
TRAINING_DEDUPLICATION_MIN_FRACTION = 0.1
assert _CHUNK_ELEMENTS > 0 and _CHUNK_ELEMENTS & (_CHUNK_ELEMENTS - 1) == 0
_CHUNK_SHIFT = _CHUNK_ELEMENTS.bit_length() - 1


def _resolve_device(device: torch.device | str) -> torch.device:
    """Bind an unindexed CUDA device to the current one, so device identity compares exactly."""
    resolved = torch.device(device)
    return (torch.device("cuda", torch.cuda.current_device())
            if resolved.type == "cuda" and resolved.index is None else resolved)


def select_si_rows(
    side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
    index: slice | torch.Tensor,
) -> SideInfoMatrix | _SideInfoRows | torch.Tensor:
    """Select side-information rows lazily when the source can, eagerly for a dense tensor."""
    if isinstance(side_info, (SideInfoMatrix, _SideInfoRows)):
        return side_info.rows(index)
    return side_info[index]


@dataclass(frozen=True, slots=True)
class _SideInfoColumn:
    """One decoder-side-information column retained in its compact device representation."""

    ranks: torch.Tensor | None
    dictionaries: torch.Tensor | None
    chunk_offsets: torch.Tensor | None
    raw_chunks: torch.Tensor | None  # None when no chunk stores values verbatim
    dense: torch.Tensor | None
    source_norm_factors: torch.Tensor | None


class SideInfoColumns:
    """Target-independent flat side-information payload shared by round matrices.

    The payload is produced from CPU ``HistoryVector`` instances and consumed by
    ``SideInfoMatrix`` decoders; compressed columns retain device ranks, concatenated
    dictionaries, chunk offsets, and raw-chunk flags while dense columns retain values.
    """

    def __init__(
        self, side_information: Sequence[HistoryVector], vector_size: int,
        norm_slices: Sequence[slice], device: torch.device,
    ) -> None:
        assert all(value.numel() == vector_size for value in side_information), (
            f"Each side-information vector must contain {vector_size} values.")
        self.vector_size = vector_size
        self._device = device
        self._columns = tuple(self._encode(value, vector_size, norm_slices, device)
                              for value in side_information)

    @staticmethod
    def _encode(value: HistoryVector, vector_size: int, norm_slices: Sequence[slice], device: torch.device) -> _SideInfoColumn:
        if isinstance(value, CompressedHistoryVector):
            expected_ranges = tuple((item.start, item.stop) for item in norm_slices)
            assert value.normalization_ranges == expected_ranges, (
                "Stored history normalization slices differ from the current model slices.")
            chunks = tuple(value._iter_encoded_chunks())
            if chunks:
                # Keep one random-access rank and one concatenated value array per column.
                # Raw chunks use their local coordinate as a rank; this also makes the final
                # partial chunk follow exactly the same indexing path as full chunks.
                all_dictionary = all(chunk.codes is not None for chunk in chunks)
                narrow = all_dictionary and all(chunk.values.numel() <= 65_536 for chunk in chunks)
                rank_dtype = torch.int16 if narrow else torch.int32
                ranks = torch.empty(vector_size, dtype=rank_dtype, device=device)
                values: list[torch.Tensor] = []
                offsets = [0]
                raw = []
                for chunk in chunks:
                    chunk = chunk.to(device)
                    local = torch.arange(chunk.stop - chunk.start, device=device)
                    ranks[chunk.start:chunk.stop] = (
                        local.to(rank_dtype) if chunk.codes is None else chunk.ranks(local).to(rank_dtype))
                    raw.append(chunk.codes is None)
                    values.append(chunk.values)
                    offsets.append(offsets[-1] + chunk.values.numel())
                return _SideInfoColumn(
                    ranks, torch.cat(values),
                    torch.tensor(offsets, dtype=torch.long, device=device),
                    torch.tensor(raw, dtype=torch.bool, device=device) if any(raw) else None,
                    None, value.normalization_factors.to(device),
                )
        values = value.materialize()
        assert values.numel() == vector_size
        return _SideInfoColumn(None, None, None, None, values.to(device=device, dtype=torch.float32), None)

    @property
    def device(self) -> torch.device:
        """Return the device containing the flat column payload tensors."""
        return self._device

    @property
    def columns(self) -> tuple[_SideInfoColumn, ...]:
        """Return the target-independent per-column flat payloads."""
        return self._columns

    def to(self, device: torch.device) -> SideInfoColumns:
        """Return this payload on ``device`` without invalidating matrices sharing it."""
        if device == self._device:
            return self
        result = self.__class__.__new__(self.__class__)
        result.vector_size = self.vector_size
        result._device = device
        result._columns = tuple(_SideInfoColumn(
            None if column.ranks is None else column.ranks.to(device),
            None if column.dictionaries is None else column.dictionaries.to(device),
            None if column.chunk_offsets is None else column.chunk_offsets.to(device),
            None if column.raw_chunks is None else column.raw_chunks.to(device),
            None if column.dense is None else column.dense.to(device),
            None if column.source_norm_factors is None else column.source_norm_factors.to(device),
        ) for column in self._columns)
        return result


class SideInfoColumnsCache:
    """Encoded columns reused by every side-information matrix of one client-round.

    A client-round builds matrices over the same vectors in train(A), encode(A), prior(B),
    decode(A) order, so two least-recently-used entries cover it. Entries are matched by vector
    identity and keep those vectors alive, so no recycled object can alias a stale payload;
    ``_WZRoundCodec.release_round_resources`` drops them when the client-round ends.
    """

    _MAX_ENTRIES = 2

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple[HistoryVector, ...], SideInfoColumns] = OrderedDict()

    def get(
        self, side_information: Sequence[HistoryVector], vector_size: int,
        norm_slices: Sequence[slice], device: torch.device,
    ) -> SideInfoColumns:
        """Return the encoded columns for these exact side-information vectors, building on a miss."""
        device = _resolve_device(device)
        key = tuple(side_information)
        for stored_key in self._entries:
            if len(stored_key) == len(key) and all(left is right for left, right in zip(stored_key, key)):
                self._entries[stored_key] = self._entries[stored_key].to(device)
                self._entries.move_to_end(stored_key)
                return self._entries[stored_key]
        self._entries[key] = SideInfoColumns(key, vector_size, norm_slices, device)
        while len(self._entries) > self._MAX_ENTRIES:
            self._entries.popitem(last=False)
        return self._entries[key]

    def clear(self) -> None:
        """Release encoded columns while leaving model and protocol state untouched."""
        self._entries.clear()


class SideInfoMatrix:
    """GPU-resident compressed side information with batched row decoding.

    A matrix stores one history vector per column. Compressed vectors use flat device ranks,
    concatenated dictionaries, and chunk metadata to decode only requested rows; dense
    ``HistoryVector`` values use a device tensor directly. Gathered values are restored with
    their stored source factors, then normalized with current target factors on the device.
    """

    def __init__(
        self,
        side_information: Sequence[HistoryVector],
        vector_size: int,
        norm_factors: torch.Tensor,
        norm_slices: Sequence[slice],
        device: torch.device,
        columns: SideInfoColumns | None = None,
    ) -> None:
        self._shape = (vector_size, max(len(side_information), 1))
        self._device = _resolve_device(device)
        self._dtype = torch.float32
        self._norm_factors = norm_factors.to(self._device, dtype=torch.float32)
        self._norm_slices = tuple(slice(*value.indices(vector_size)) for value in norm_slices)
        self._slice_ids = torch.full((vector_size,), -1, dtype=torch.int32, device=self._device)
        for index, value_slice in enumerate(self._norm_slices):
            self._slice_ids[value_slice] = index
        self._column_payload = columns.to(self._device) if columns is not None else SideInfoColumns(
            side_information, vector_size, self._norm_slices, self._device)
        self._columns = self._column_payload.columns

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(vector_size, side_information_count)`` like a dense matrix."""
        return self._shape

    def rows(self, index: slice | torch.Tensor | Sequence[int] | int) -> _SideInfoRows:
        """Return a lazy view selecting rows without decoding their values."""
        if isinstance(index, slice):
            start, stop, step = index.indices(self._shape[0])
            assert step == 1, "SideInfoMatrix only supports contiguous slices."
            return _SideInfoRows(self, slice(start, stop))
        row_indices = (
            torch.tensor([index], device=self._device) if isinstance(index, int)
            else torch.as_tensor(index, dtype=torch.long, device=self._device)
        )
        assert row_indices.ndim == 1 and bool(((row_indices >= 0) & (row_indices < self._shape[0])).all()), (
            f"SideInfoMatrix row indices must be in [0, {self._shape[0]}).")
        return _SideInfoRows(self, row_indices)

    @property
    def device(self) -> torch.device:
        """Return the device retaining dictionaries, codes, and target normalization."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the decoded matrix dtype."""
        return self._dtype

    def to(self, device: torch.device | str) -> SideInfoMatrix:
        """Move encoded columns and normalization metadata to ``device`` in place."""
        target = _resolve_device(device)
        if target == self._device:
            return self
        self._column_payload = self._column_payload.to(target)
        self._columns = self._column_payload.columns
        self._norm_factors = self._norm_factors.to(target)
        self._slice_ids = self._slice_ids.to(target)
        self._device = target
        return self

    @staticmethod
    def _gather_ranks(
        column: _SideInfoColumn, rows: torch.Tensor, chunk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return chunk-local ranks and their positions in the concatenated dictionary."""
        assert column.ranks is not None and column.chunk_offsets is not None
        ranks = column.ranks[rows].to(torch.int32)
        if column.ranks.dtype is torch.int16:
            ranks &= 0xFFFF  # int16 is only storage; ranks are unsigned and may exceed 32767.
        return ranks, column.chunk_offsets[chunk_ids] + ranks

    def _decode_column(
        self, column: _SideInfoColumn, rows: torch.Tensor, chunk_ids: torch.Tensor,
        slice_ids: torch.Tensor, target_factors: torch.Tensor,
    ) -> torch.Tensor:
        if column.dense is not None:
            decoded = column.dense[rows]
        else:
            assert column.dictionaries is not None
            _, positions = self._gather_ranks(column, rows, chunk_ids)
            decoded = column.dictionaries[positions]
        if column.source_norm_factors is not None:
            source_factors = column.source_norm_factors[slice_ids]
            decoded.mul_(source_factors[:, 0]).add_(source_factors[:, 1])
        return decoded.sub_(target_factors[:, 1]).div_(target_factors[:, 0])

    def _decode_codes(self, column: _SideInfoColumn, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unsigned local keys and physical chunk ids for selected rows."""
        if column.dense is not None:
            keys = column.dense[rows].view(torch.int32).to(torch.int64) & 0xFFFF_FFFF
            return keys, torch.zeros(rows.shape, dtype=torch.int64, device=rows.device)
        chunk_ids = rows >> _CHUNK_SHIFT
        ranks, positions = self._gather_ranks(column, rows, chunk_ids)
        keys = ranks.to(torch.int64)
        if column.raw_chunks is not None:
            # A raw chunk has no dictionary, so its rank is a coordinate; key it by value bits.
            assert column.dictionaries is not None
            values = column.dictionaries[positions].view(torch.int32).to(torch.int64) & 0xFFFF_FFFF
            keys = torch.where(column.raw_chunks[chunk_ids], values, keys)
        return keys, chunk_ids

    def _group_codes(self, column: _SideInfoColumn, rows: torch.Tensor) -> torch.Tensor:
        """Add target-normalization slice identity to compact codes used for row grouping."""
        codes, chunk_ids = self._decode_codes(column, rows)
        chunk_count = 1 if column.chunk_offsets is None else column.chunk_offsets.numel() - 1
        region_ids = self._slice_ids[rows].to(torch.int64) * chunk_count + chunk_ids
        return region_ids * (1 << 32) + codes

    def __getitem__(self, rows: slice | torch.Tensor | Sequence[int] | int) -> torch.Tensor:
        """Decode selected rows as a float32 ``(rows, columns)`` tensor."""
        row_indices = self.rows(rows)._resolved_indices()
        if not self._columns:
            return torch.zeros((row_indices.numel(), 1), device=self._device, dtype=self._dtype)
        result = torch.empty((row_indices.numel(), len(self._columns)), device=self._device, dtype=self._dtype)
        chunk_ids = row_indices >> _CHUNK_SHIFT
        slice_ids = self._slice_ids[row_indices].long()
        target_factors = self._norm_factors[slice_ids]
        for index, column in enumerate(self._columns):
            result[:, index] = self._decode_column(column, row_indices, chunk_ids, slice_ids, target_factors)
        return result

    def iter_code_columns(self, rows: torch.Tensor | None = None) -> Iterator[torch.Tensor]:
        """Yield compact row keys for deduplication without decoding compressed columns."""
        selected_rows = (
            torch.arange(self._shape[0], device=self._device) if rows is None else rows
        )
        if not self._columns:
            yield torch.zeros(selected_rows.shape, device=self._device, dtype=torch.int64)
            return
        yield from (self._group_codes(column, selected_rows) for column in self._columns)

    def code_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Return transient compact keys for selected rows without decoding SI values."""
        if not self._columns:
            return torch.zeros((rows.numel(), 1), device=self._device, dtype=torch.int64)
        return torch.stack([self._group_codes(column, rows) for column in self._columns], dim=1)


class _SideInfoRows:
    """Lazy selected-row view over a ``SideInfoMatrix`` consumed by batched callers."""

    def __init__(self, matrix: SideInfoMatrix, indices: slice | torch.Tensor) -> None:
        self._matrix = matrix
        self._indices = indices

    def _resolved_indices(self) -> torch.Tensor:
        if isinstance(self._indices, slice):
            return torch.arange(
                self._indices.start, self._indices.stop, device=self.device, dtype=torch.long
            )
        return self._indices

    @property
    def shape(self) -> tuple[int, int]:
        """Return the selected row count and matrix column count without decoding."""
        row_count = (
            self._indices.stop - self._indices.start
            if isinstance(self._indices, slice) else self._indices.numel()
        )
        return (row_count, self._matrix.shape[1])

    @property
    def device(self) -> torch.device:
        """Return the device on which selected indices and matrix data reside."""
        return self._matrix.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the decoded matrix dtype."""
        return self._matrix.dtype

    def to(self, device: torch.device | str) -> _SideInfoRows:
        """Move the underlying matrix and selected indices to ``device`` in place."""
        self._matrix.to(device)
        if isinstance(self._indices, torch.Tensor):
            self._indices = self._indices.to(self._matrix.device)
        return self

    def rows(self, index: slice | torch.Tensor | Sequence[int] | int) -> _SideInfoRows:
        """Return a nested lazy view relative to this selection."""
        if isinstance(index, slice):
            start, stop, step = index.indices(self.shape[0])
            assert step == 1, "_SideInfoRows only supports contiguous slices."
            if isinstance(self._indices, slice):
                return _SideInfoRows(
                    self._matrix,
                    slice(self._indices.start + start, self._indices.start + stop),
                )
            relative = torch.arange(start, stop, device=self.device)
        elif isinstance(index, int):
            relative = torch.tensor([index], device=self.device)
        else:
            relative = torch.as_tensor(index, dtype=torch.long, device=self.device)
        assert relative.ndim == 1 and bool(((relative >= 0) & (relative < self.shape[0])).all()), (
            f"_SideInfoRows row indices must be in [0, {self.shape[0]}).")
        if isinstance(self._indices, slice):
            return _SideInfoRows(self._matrix, self._indices.start + relative)
        return _SideInfoRows(self._matrix, self._indices[relative])

    def __getitem__(self, index: slice | torch.Tensor | Sequence[int] | int) -> torch.Tensor:
        """Decode selected rows eagerly at the leaf of a batched consumer."""
        selected = self.rows(index)
        return selected._matrix[selected._resolved_indices()]

    def iter_code_columns(self) -> Iterator[torch.Tensor]:
        """Yield compact keys for selected rows without decoding side-information values."""
        yield from self._matrix.iter_code_columns(self._resolved_indices())

    def code_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Return compact keys for rows relative to this view."""
        selected = (
            self._indices.start + rows
            if isinstance(self._indices, slice) else self._indices[rows]
        )
        return self._matrix.code_rows(selected)


def new_rnn_model(
    num_planes: int,
    bins_per_plane: int,
    side_info_size: int,
    marginal: bool,
    hidden_dim: int,
    layers: int,
    shared_heads: bool = True,
) -> EncoderDecoderLayeredRNN:
    """Create the RNN network shared by the WZ quantizer and the conditional prior model."""
    return EncoderDecoderLayeredRNN(
        num_planes=num_planes,
        bins_per_plane=bins_per_plane,
        side_info_size=max(1, side_info_size),
        input_dim=1,
        layers=layers,
        hidden_dim=hidden_dim,
        marginal=marginal,
        shared_encoder=shared_heads,
        shared_decoder=shared_heads,
    )


def batch_loop(
    func: Callable[[int, int], torch.Tensor],
    model: EncoderDecoderLayeredRNN,
    input_size: int,
    batch_size: int,
    cat_dim: int | None = None,
) -> torch.Tensor:
    """Run an inference callback over contiguous batches on the best device and concatenate the outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.inference_mode():
        batches = [func(start, min(start + batch_size, input_size)) for start in range(0, input_size, batch_size)]
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(batches, dim=cat_dim if cat_dim is not None else (1 if batches[0].ndim == 3 else 0))


def _sample_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    """Average per-sample values while preserving duplicate multiplicities when supplied."""
    return values.mean() if sample_weights is None else (values * sample_weights).sum() / sample_weights.sum()


def float_prefix(values: torch.Tensor, match_bits: int) -> torch.Tensor:
    """Return float32 values truncated to a shared leading-bit prefix."""
    assert values.dtype == torch.float32
    assert 1 <= match_bits <= 32
    return (
        values if match_bits == 32
        else (values.view(torch.int32) & -(1 << (32 - match_bits))).view(torch.float32)
    )


def wz_model_training_loop(
    compute_loss: Callable[..., torch.Tensor],
    model: EncoderDecoderLayeredRNN,
    parameters: Iterator[Parameter],
    training_inputs: tuple[torch.Tensor, ...],
    side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
    c_cfg: WZcfgQuant,
    batch_size: int = 50_000,
    label: str = "Training",
    metrics_fn: Callable[
        [EncoderDecoderLayeredRNN, tuple[torch.Tensor, ...], SideInfoMatrix | _SideInfoRows | torch.Tensor], str
    ] | None = None,
    index_sampler: Callable[[int], torch.Tensor] | None = None,
    **loss_fn_options,
) -> float:
    """Train one model, adaptively collapsing duplicate rows and preserving their loss mass.

    ``index_sampler`` draws each epoch's row indices; the default samples every row uniformly.
    """
    model.to('cuda').train()
    training_inputs = tuple(value.to('cuda') for value in training_inputs)
    assert training_inputs
    assert all(value.shape[0] == training_inputs[0].shape[0] for value in training_inputs)
    side_info = side_info.to('cuda')

    input_size = training_inputs[0].shape[0]
    total_samples = min(c_cfg.train_sample_size, input_size)
    batch_draws = min(batch_size, total_samples)
    expected_index_duplicate_fraction = 1 - (
        input_size * (1 - (1 - 1 / input_size) ** batch_draws) / batch_draws
        if input_size > 1 else 1
    )
    row_groups = None
    if c_cfg.deduplicate_training:
        blocks = (*training_inputs, side_info)
        probe_size = min(batch_size, input_size)
        probe_indices = (
            torch.arange(probe_size, device=side_info.device) * input_size // probe_size
        )
        probe_groups = unique_row_groups(
            tuple(_rows_for_grouping(block, probe_indices) for block in blocks), c_cfg.training_match_bits)[1]
        content_duplicate_fraction = 1 - (int(probe_groups.max()) + 1) / probe_size
        if content_duplicate_fraction >= TRAINING_DEDUPLICATION_MIN_FRACTION:
            representatives, row_groups, collisions = unique_row_groups(
                blocks, c_cfg.training_match_bits)
            # Colliding rows are not provably equal, so they must not share a weighted group.
            row_groups[collisions] = representatives.numel() + torch.arange(
                collisions.numel(), device=row_groups.device)
        elif expected_index_duplicate_fraction >= TRAINING_DEDUPLICATION_MIN_FRACTION:
            row_groups = torch.arange(input_size, device=side_info.device)

    # Both training losses are scaled by 1/K relative to their previous forms.
    optimizer_epsilon = 1e-8 / model.num_planes
    optimizer = torch.optim.Adam(
        parameters,
        fused=c_cfg.fused_optimizer,
        lr=c_cfg.lr,
        weight_decay=0,
        eps=optimizer_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c_cfg.lr_step, gamma=0.3)
    scaler = torch.amp.GradScaler("cuda", enabled=c_cfg.mixed_precision)

    total_batches = (total_samples + batch_size - 1) // batch_size
    pbar = create_training_progress_bar(
        c_cfg.train_epochs * total_batches,
        desc=label,
        disable=not c_cfg.training_progress_bar,)

    for epoch in range(c_cfg.train_epochs):
        epoch_started = perf_counter()
        indices = (
            torch.randint(input_size, (total_samples,), device=training_inputs[0].device)
            if index_sampler is None
            else index_sampler(total_samples)
        )
        # Decode the sampled SI once per epoch. In O rounds this replaces 32 minibatch passes over
        # 150 chunked history columns with one pass, while preserving the exact sampled rows and
        # per-minibatch duplicate weighting below.
        epoch_inputs = tuple(value[indices] for value in training_inputs)
        epoch_side_info = side_info[indices]
        epoch_row_groups = None if row_groups is None else row_groups[indices]
        epoch_loss = 0.0
        epoch_model_samples = 0
        for start in range(0, total_samples, batch_size):
            batch_slice = slice(start, start + batch_size)
            input_batch = tuple(value[batch_slice] for value in epoch_inputs)
            si_batch = epoch_side_info[batch_slice]
            sample_weights = None
            if epoch_row_groups is not None:
                _, inverse, counts = torch.unique(
                    epoch_row_groups[batch_slice], return_inverse=True, return_counts=True
                )
                if counts.numel() < input_batch[0].shape[0]:
                    representatives = torch.full_like(counts, input_batch[0].shape[0])
                    representatives.scatter_reduce_(
                        0,
                        inverse,
                        torch.arange(input_batch[0].shape[0], device=inverse.device),
                        reduce="amin",
                    )
                    input_batch = tuple(value[representatives] for value in input_batch)
                    si_batch = si_batch[representatives]
                    sample_weights = counts.to(training_inputs[0].dtype)
            epoch_model_samples += input_batch[0].shape[0]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=c_cfg.mixed_precision):
                loss = compute_loss(
                    model, input_batch, si_batch, epoch, c_cfg,
                    sample_weights=sample_weights, **loss_fn_options
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if c_cfg.training_progress_bar:
                pbar.set_postfix({"loss": f"{loss.item():.2f}"})
                pbar.update(1)
        scheduler.step()
        epoch_loss /= total_batches
        if c_cfg.training_log_every_epochs and (
            (epoch + 1) % c_cfg.training_log_every_epochs == 0 or epoch + 1 == c_cfg.train_epochs
        ):
            epoch_seconds = perf_counter() - epoch_started
            metrics = ""
            if metrics_fn is not None:
                model.eval()
                with torch.inference_mode():
                    metrics = f", {metrics_fn(model, training_inputs, side_info)}"
                model.train()
            print(
                f"{label} epoch {epoch + 1}/{c_cfg.train_epochs}: "
                f"loss={epoch_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.3e}, "
                f"seconds={epoch_seconds:.2f}, samples_per_second={total_samples / epoch_seconds:.0f}, "
                f"model_samples={epoch_model_samples}/{total_samples}{metrics}",
                flush=True,
            )

    pbar.close()
    model.cpu()
    torch.cuda.empty_cache()
    return epoch_loss


def _stacked_problem_sampler(
    weights: tuple[float, ...], block_size: int, device: torch.device
) -> Callable[[int], torch.Tensor]:
    """Draw rows from equally sized stacked problems in proportion to `weights`, refreshed per call."""
    assert len(weights) > 1 and all(weight > 0 for weight in weights), (
        f"Every stacked problem needs a positive weight; got {weights}.")
    distribution = torch.tensor(weights, device=device, dtype=torch.float32)
    distribution /= distribution.sum()

    def sample(count: int) -> torch.Tensor:
        return torch.multinomial(distribution, count, replacement=True) * block_size + torch.randint(
            block_size, (count,), device=device)

    return sample


class WZcfgQuant(BaseModel):
    """Quantizer and training configuration consumed by WZQuantizerCancer."""
    # best config for FL delta (laplace like with b=0.356 but sharp peak around 0)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _TRAINING_CACHE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "bins_per_plane",
        "num_planes",
        "norm_slices",
        "outlier_threshold",
        "marginal_loss",
        "train_epochs",
        "reconst_ld",
        "train_sample_size",
        "train_batch_size",
        "lr",
        "lr_step",
        "tau",
        "prior_train_repeats",
        "fused_optimizer",
        "mixed_precision",
        "source_perturbation",
        "shared_heads",
        "prior_context",
        "prior_probabilities",
        "hidden_dim",
        "layers",
    })

    bins_per_plane: int
    num_planes: int
    norm_slices: Sequence[slice] | None = None
    outlier_threshold: float | None = None
    marginal_loss: bool = False
    pretrain_pth_dir: Path = Path("FL_code/data/pre_trained_pth")

    train_epochs: int = 120
    reconst_ld: float = 72.31601255189986
    train_sample_size: int = 550_000
    train_batch_size: int = 6_250
    lr: float = 0.0056728196862653865
    lr_step: int = 43
    tau: float = 5.5
    warm_training: bool = False
    warm_retrain_repeats: int = Field(default=2, ge=1, strict=True)
    quantizer_train_repeats: int = 3
    prior_train_repeats: int = 3
    prior_retrain_epochs: int = Field(default=30, ge=1, strict=True)
    # Distinguishes stochastic training outcomes for otherwise-identical (target, SI, config)
    # inputs, e.g. a sparse-replay case seeding torch's RNG per client-round. Omitted from the
    # cache payload when unset so every existing cache entry keeps its current key.
    training_seed: int | None = None
    training_cache_discriminator: str | tuple[str, ...] | None = None

    training_progress_bar: bool = False
    training_log_every_epochs: int | None = None
    fused_optimizer: bool = True
    mixed_precision: bool = False
    deduplicate_training: bool = True
    training_match_bits: int = Field(default=26, ge=1, le=32)
    source_perturbation: Literal["none", "draw_discard", "add"] = "none"
    shared_heads: bool = True
    prior_context: Literal["hard", "soft"] = "hard"
    prior_probabilities: Literal["categorical", "gumbel"] = "categorical"
    hidden_dim: int = 128
    layers: int = 6
    cache_dir: Path | None = None
    cache_overwrite: bool = False

    def warm_retraining_config(self, train_epochs: int | None = None) -> WZcfgQuant:
        """Return the shared lower-rate, lower-temperature configuration for warm fitting."""
        return self.model_copy(update={
            "train_epochs": train_epochs if train_epochs is not None else max(1, round(self.train_epochs * 0.25)),
            "lr": self.lr * 0.3,
            "tau": 0.5,
        })

    def training_cache_payload(self) -> dict[str, Any]:
        """Return the model-semantic options used to identify cached training results."""
        payload = self.model_dump(include=self._TRAINING_CACHE_FIELDS) | {
            # These pinned historical values preserve the established baseline cache namespace.
            "max_side_info_count": 5,
            "pretrain_pth_dir": Path("FL_code/data/pre_trained_pth"),
            "training_progress_bar": False,
            "training_log_every_epochs": None,
            "tf32": False,
        }
        # Two was the omitted historical default. Every other value, including today's default,
        # must remain explicit so it cannot reuse a legacy two-repeat cache entry.
        if self.quantizer_train_repeats != 2:
            payload["quantizer_train_repeats"] = self.quantizer_train_repeats
        if self.warm_training:
            payload["warm_training"] = True
            payload["warm_retrain_repeats"] = self.warm_retrain_repeats
        if self.training_seed is not None:
            payload["training_seed"] = self.training_seed
        if self.training_cache_discriminator is not None:
            payload["training_cache_discriminator"] = self.training_cache_discriminator
        return payload


class WZQuantizerCancer:
    """Learned Wyner-Ziv quantizer for one flattened Cancer protocol update vector."""

    prior_calculator: ClassVar[type[PriorCalculator]] = PriorCalculator
    candidate_failure_tolerance: ClassVar[int] = 2

    @property
    def device(self) -> torch.device:
        """Return the accelerator used by quantizer inference when available."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getattr__(self, name: str) -> Any:
        """Read missing quantizer attributes from c_cfg when they are configuration fields."""
        c_cfg = self.__dict__.get("c_cfg")
        if c_cfg is not None and name in type(c_cfg).model_fields:
            return getattr(c_cfg, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}.")

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent instance attributes from silently shadowing c_cfg fields."""
        c_cfg = self.__dict__.get("c_cfg")
        assert name == "c_cfg" or c_cfg is None or name not in type(c_cfg).model_fields, (
            f"{name!r} is a WZcfgQuant field; update c_cfg instead of shadowing it on the quantizer.")
        super().__setattr__(name, value)

    def _new_coding_model(self, side_info_size: int) -> EncoderDecoderLayeredRNN:
        """Create the coding model used for initial and retry training attempts."""
        return new_rnn_model(
            num_planes=self.num_planes,
            bins_per_plane=self.bins_per_plane,
            side_info_size=side_info_size,
            marginal=self.marginal_loss,
            shared_heads=self.shared_heads,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
        )

    def _cached_training_quantizer_class(self) -> type[WZQuantizerCancer]:
        """Return the quantizer implementation whose parameters are cached."""
        return WZQuantizerCancer

    def __init__(
        self,
        c_cfg: WZcfgQuant,
        si_size: int,
        side_info_columns_cache: SideInfoColumnsCache | None = None,
    ) -> None:
        self.c_cfg: WZcfgQuant = c_cfg
        self.no_side_info: bool = si_size == 0
        assert self.marginal_loss or not self.no_side_info, (
            "si_size=0 requires marginal_loss=True; no-SI quantizers must be explicit.")

        self.side_info_size: int = max(si_size, 1)

        self.coding_model: EncoderDecoderLayeredRNN = self._new_coding_model(self.side_info_size)

        self.side_info_list_used: list[HistoryVector] | None = None
        # A quantizer built outside a round codec owns its cache and releases it with itself.
        self.side_info_columns_cache = side_info_columns_cache or SideInfoColumnsCache()
        self.vector_size: int | None = None
        self.quantizer_prior_stage_rates: tuple[float, ...] | None = None
        self.quantizer_prior_rate: float | None = None
        self._warm_start_states: tuple[dict[str, torch.Tensor], ...] = ()
        self.warm_retrained_states: tuple[dict[str, torch.Tensor], ...] = ()
        self.warm_retrained_losses: tuple[float, ...] = ()
        self.warm_deployed_parent: int | None = None
        # Set while stacking, whenever the caller weights the stacked problems unevenly.
        self.training_index_sampler: Callable[[int], torch.Tensor] | None = None

    def load_warm_start_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load one prior-round parent state for warm retraining."""
        self.load_warm_start_states((state,))

    def load_warm_start_states(self, states: Sequence[Mapping[str, torch.Tensor]]) -> None:
        """Load the ordered parent states used by independent warm-retraining lineages."""
        assert self.warm_training, "A warm-start state requires warm_training=True."
        assert not self._warm_start_states, "Warm-start states can only be loaded once per quantizer."
        assert states, "Warm training requires at least one parent state."
        copied: list[dict[str, torch.Tensor]] = []
        for parent_index, state in enumerate(states):
            try:
                self.coding_model.load_state_dict(state)
            except RuntimeError as error:
                raise AssertionError(
                    f"Warm parent {parent_index} is incompatible with the warm-training architecture."
                ) from error
            copied.append({
                name: value.detach().cpu().clone()
                for name, value in self.coding_model.state_dict().items()
            })
        self._warm_start_states = tuple(copied)
        self.coding_model.load_state_dict(self._warm_start_states[0])

    def training_config(self) -> WZcfgQuant:
        """Return the effective optimizer and temperature settings for this fit."""
        if not self.warm_training:
            return self.c_cfg
        return self.c_cfg.warm_retraining_config()

    @staticmethod
    def training_metrics(
        model: EncoderDecoderLayeredRNN,
        training_inputs: tuple[torch.Tensor, ...],
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
    ) -> str:
        """Summarize hard-code stage distortion and cumulative rate on a fixed training sample."""
        x = training_inputs[0][:10_000]
        y = side_info[:x.shape[0]]
        bins, codes = model.encode(x)
        reconstructions = model.decode(codes, y)
        priors = model.get_priors(codes, y)
        sample_indices = torch.arange(x.shape[0], device=x.device)
        plane_rates = [
            -torch.log2(prior[sample_indices, symbols] + 1e-12).mean().item()
            for prior, symbols in zip(priors, bins, strict=True)
        ]
        return (
            f"hard_mse={[round(F.mse_loss(reconstruction, x).item(), 6) for reconstruction in reconstructions]}, "
            f"cumulative_rate={[round(float(rate), 4) for rate in np.cumsum(plane_rates)]}"
        )

    @staticmethod
    def compute_loss(
        model: EncoderDecoderLayeredRNN,
        training_input: tuple[torch.Tensor],
        si_batch: torch.Tensor,
        epoch: int,
        c_cfg: WZcfgQuant,
        sample_weights: torch.Tensor | None,
        wmspe_denom: float,
        perturb_generator: torch.Generator,
    ) -> torch.Tensor:
        """Compute the stage-averaged reconstruction-plus-rate loss for one training batch."""
        x_batch, = training_input
        if c_cfg.source_perturbation != "none":
            perturbation = torch.randn(
                x_batch.shape, device=x_batch.device, dtype=x_batch.dtype, generator=perturb_generator
            ) * (1e-5 * x_batch.abs().mean())
            if c_cfg.source_perturbation == "add":
                x_batch = x_batch + perturbation
        progress = epoch / c_cfg.train_epochs
        tau = c_cfg.tau * np.exp(progress * np.log(0.1 / c_cfg.tau))
        reconstructions, bins, soft_codes, priors = model(x_batch, si_batch, tau=tau)

        loss = torch.zeros((), device=x_batch.device)
        sample_indices = torch.arange(x_batch.shape[0], device=x_batch.device)
        for plane in range(model.num_planes):
            distortion = _sample_mean(
                F.mse_loss(
                    reconstructions[plane], x_batch, reduction="none"
                ).flatten(start_dim=1).mean(dim=1),
                sample_weights,
            )

            posterior_prob = soft_codes[plane][sample_indices, bins[plane]]
            prior_prob = priors[plane][sample_indices, bins[plane]]
            rate = _sample_mean(
                torch.log((posterior_prob + 1e-12) / (prior_prob + 1e-12)),
                sample_weights,
            )

            new_loss = (c_cfg.reconst_ld / wmspe_denom) * distortion + rate
            loss = loss + new_loss / model.num_planes
        return loss

    def train_model(
        self,
        x_raw: torch.Tensor,
        si_raw_list: Sequence[HistoryVector] | None,
        extra_pairs: Sequence[tuple[torch.Tensor, Sequence[HistoryVector]]] = (),
        problem_weights: tuple[float, ...] | None = None,
    ) -> None:
        """Train repeated quantizer attempts and keep the finite attempt with the lowest loss.

        ``extra_pairs`` adds further (target, side information) problems to the training rows only;
        the coded vector, its side information, and every diagnostic stay those of ``x_raw``.
        ``problem_weights`` gives one sampling weight per stacked problem, the coded one first.
        """
        assert not self.warm_training or self._warm_start_states, (
            "warm_training=True requires load_warm_start_states() before training.")
        if self.cache_dir is None:
            self._train_model_uncached(x_raw, si_raw_list, extra_pairs, problem_weights)
            return

        training_result = cache_call(
            _train_quantizer_parameters,
            self.cache_dir / "quantizers",
            model_training_hash,
            self.cache_overwrite,
            # Inference-only subclasses share this training implementation and
            # restore the resulting state into their own instance below.
            quantizer_class=self._cached_training_quantizer_class(),
            c_cfg=self.c_cfg,
            x_raw=x_raw,
            side_information=tuple(si_raw_list or ()),
            side_info_columns_cache=self.side_info_columns_cache,
            # Passed only when used, so single-problem training keeps the cache keys it
            # was written with; the hash covers every keyword it receives.
            **({"extra_training_pairs": tuple(
                (extra_x, tuple(extra_si)) for extra_x, extra_si in extra_pairs)} if extra_pairs else {}),
            **({"problem_weights": problem_weights} if problem_weights is not None else {}),
            **({"warm_start_states": self._warm_start_states} if self._warm_start_states else {}),
        )
        if self._warm_start_states:
            state_dict, quantizer_prior_stage_rates, warm_states, warm_losses, deployed_parent = training_result
        else:
            # Preserve the established two-item cold-cache payload and its existing entries.
            state_dict, quantizer_prior_stage_rates = training_result
            warm_states, warm_losses, deployed_parent = (), (), None
        # Cache only portable training results; bind the current decoder-available
        # side information to this fresh quantizer instance.
        assert self.side_info_list_used is None
        if self.no_side_info:
            assert not si_raw_list
            self.side_info_list_used = []
        else:
            assert si_raw_list
            self.set_side_information(si_raw_list)
        self.vector_size = x_raw.numel()
        self.coding_model.load_state_dict(state_dict)
        self.quantizer_prior_stage_rates = quantizer_prior_stage_rates
        self.quantizer_prior_rate = sum(quantizer_prior_stage_rates)
        self.warm_retrained_states = warm_states
        self.warm_retrained_losses = warm_losses
        self.warm_deployed_parent = deployed_parent

    def _train_model_uncached(
        self,
        x_raw: torch.Tensor,
        si_raw_list: Sequence[HistoryVector] | None,
        extra_pairs: Sequence[tuple[torch.Tensor, Sequence[HistoryVector]]] = (),
        problem_weights: tuple[float, ...] | None = None,
    ) -> None:
        assert self.side_info_list_used is None
        if self.no_side_info:
            assert not si_raw_list, "Marginal quantizer training expects no side information."
            self.side_info_list_used = []
        else:
            assert si_raw_list
            self.set_side_information(si_raw_list)

        x_prep, metadata = self.preprocess_x(x_raw)
        x_prep, side_info = self._stack_training_pairs(
            x_prep, self.normalize_side_info_tensor(metadata, self.side_info_list_used or []),
            extra_pairs, problem_weights)
        self._fit_prepared(x_prep, side_info)

    def _fit_prepared(
        self, x_prep: torch.Tensor, side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor
    ) -> None:
        """Fit the coding model to normalized rows and record its own training rate estimate."""
        assert self.vector_size is not None
        wmspe_denom = (x_prep.float().square().mean().item() / 2) + 1e-8
        self.coding_model = self._train_finite_candidate_and_retrieve_best(x_prep, side_info, wmspe_denom)

        # Diagnostics describe the problem this quantizer will code, which is the leading block.
        bins, _ = self.encode_prepared(x_prep[:self.vector_size], INFERENCE_BATCH)
        prior = self.prior_calculator.compute_prior_from_network(
            self.coding_model, bins, select_si_rows(side_info, slice(0, self.vector_size)))
        self.quantizer_prior_stage_rates = self.prior_calculator.compute_plane_rates(
            prior, bins, self.c_cfg.num_planes)
        self.quantizer_prior_rate = sum(self.quantizer_prior_stage_rates)

    def _stack_training_pairs(
        self,
        x_prep: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        extra_pairs: Sequence[tuple[torch.Tensor, Sequence[HistoryVector]]],
        problem_weights: tuple[float, ...] | None = None,
    ) -> tuple[torch.Tensor, SideInfoMatrix | _SideInfoRows | torch.Tensor]:
        """Append further training problems below the coded one, each normalized by its own target."""
        # `> 1` as well, so weights handed in without anything to weigh fail here rather than being
        # dropped while still counting toward the training cache key.
        assert problem_weights is None or len(problem_weights) == 1 + len(extra_pairs) > 1, (
            f"problem_weights has {len(problem_weights or ())} entries for the coded problem and "
            f"{len(extra_pairs)} extra ones.")
        if not extra_pairs:
            return x_prep, side_info

        rows = x_prep.shape[0]
        # Dense, unlike every other side-information consumer: the hook that reaches here is
        # dormant, so it is guarded rather than given its own lazy stacked view.
        stacked_bytes = rows * (1 + len(extra_pairs)) * side_info.shape[1] * 4
        assert stacked_bytes < 1 << 31, (
            f"extra_training_problems() would materialize {stacked_bytes / 1024 ** 3:.1f} GiB of "
            "dense side information; make the stacked block lazy before enabling it at this size.")
        # Filled block by block rather than concatenated, because holding the parts and the
        # concatenation at once doubles peak memory at full vector size.
        stacked_x = torch.empty(
            (rows * (1 + len(extra_pairs)), 1), device=x_prep.device, dtype=x_prep.dtype)
        stacked_si = torch.empty(
            (stacked_x.shape[0], side_info.shape[1]), device=side_info.device, dtype=side_info.dtype)
        stacked_x[:rows], stacked_si[:rows] = x_prep, side_info[:]
        del x_prep, side_info

        for index, (extra_x, extra_si) in enumerate(extra_pairs, start=1):
            assert len(extra_si) == self.side_info_size, (
                f"Extra training pair {index} has {len(extra_si)} side-information vectors; "
                f"the quantizer is built for {self.side_info_size}.")
            block = slice(index * rows, (index + 1) * rows)
            extra_prep, extra_metadata = self.preprocess_x(extra_x)
            stacked_x[block] = extra_prep
            stacked_si[block] = self.normalize_side_info_tensor(extra_metadata, extra_si)
        if problem_weights is not None:
            self.training_index_sampler = _stacked_problem_sampler(
                problem_weights, rows, stacked_x.device)
        return stacked_x, stacked_si

    def _train_finite_candidate_and_retrieve_best(
        self, x_prep: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor, wmspe_denom: float,
    ) -> EncoderDecoderLayeredRNN:
        if self._warm_start_states:
            states: list[dict[str, torch.Tensor]] = []
            losses: list[float] = []
            for parent_index, parent_state in enumerate(self._warm_start_states):
                model, loss = self._train_finite_candidate(
                    x_prep, side_info, wmspe_denom, parent_state, self.warm_retrain_repeats,
                    f"parent {parent_index}"
                )
                states.append({
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                })
                losses.append(loss)
                del model
            self.warm_retrained_states = tuple(states)
            self.warm_retrained_losses = tuple(losses)
            self.warm_deployed_parent = min(
                range(len(states)), key=self.warm_retrained_losses.__getitem__)
            self.coding_model.load_state_dict(self.warm_retrained_states[self.warm_deployed_parent])
            return self.coding_model
        return self._train_finite_candidate(
            x_prep, side_info, wmspe_denom, None, self.quantizer_train_repeats, "cold"
        )[0]

    def _train_finite_candidate(
        self,
        x_prep: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        wmspe_denom: float,
        initial_state: Mapping[str, torch.Tensor] | None,
        repeats: int,
        label: str,
    ) -> tuple[EncoderDecoderLayeredRNN, float]:
        """Train one lineage repeatedly and return its lowest-loss usable candidate."""
        failure_tolerance = self.candidate_failure_tolerance
        made_quants_and_stat: list[tuple[EncoderDecoderLayeredRNN, float]] = []
        rejected: list[str] = []
        for attempt_index in range(repeats + failure_tolerance):
            if len(made_quants_and_stat) == repeats:
                break

            model = self.coding_model if attempt_index == 0 and initial_state is None else self._new_coding_model(
                self.side_info_size)
            if initial_state is not None:
                model.load_state_dict(initial_state)
            perturb_generator = torch.Generator(device="cuda").manual_seed(torch.initial_seed())
            loss = wz_model_training_loop(
                self.compute_loss, model, model.parameters(),
                (x_prep,), side_info, self.training_config(), batch_size=self.train_batch_size,
                label=f"Quantizer {label} attempt {attempt_index + 1}",
                metrics_fn=self.training_metrics,
                index_sampler=self.training_index_sampler,
                wmspe_denom=wmspe_denom,
                perturb_generator=perturb_generator)

            if not np.isfinite(loss):
                rejected.append(f"attempt {attempt_index + 1}: non-finite loss")
                continue

            # Check through this quantizer's own encode/decode so the pass benefits from
            # row deduplication; the caller overwrites coding_model with the winner below.
            # Only the leading block is checked: it is the vector this quantizer will code.
            self.coding_model = model
            bins, _ = self.encode_prepared(x_prep[:self.vector_size], INFERENCE_BATCH)
            reconstruction = self.decode_prepared(
                bins, select_si_rows(side_info, slice(0, self.vector_size)), INFERENCE_BATCH)
            if not torch.isfinite(reconstruction).all():
                rejected.append(f"attempt {attempt_index + 1}: non-finite reconstruction")
            elif not self._candidate_is_usable(bins, reconstruction):
                symbol_counts = tuple(plane.unique().numel() for plane in bins)
                rejected.append(
                    f"attempt {attempt_index + 1}: unusable symbols per plane {symbol_counts}"
                )
            else:
                made_quants_and_stat.append((model, loss))
        assert len(made_quants_and_stat) == repeats, (
            f"Created {len(made_quants_and_stat)}/{repeats} required {label} quantizer candidates "
            f"after {repeats + failure_tolerance} attempts; rejected: {rejected}."
        )

        return min(made_quants_and_stat, key=lambda attempt: attempt[1])

    def _candidate_is_usable(
        self,
        bins: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> bool:
        """Return whether a finite trained candidate satisfies codec-specific structure."""
        return True

    def encoding_process(
        self, x_raw: torch.Tensor, batch_size: int = INFERENCE_BATCH,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PreprocessMetadata]:
        """Preprocess and quantize one raw vector into per-plane bin indices and soft code posteriors.

        Estimator subclasses that bypass the neural encoder return ``None`` soft codes.
        """
        x_prep, metadata = self.preprocess_x(x_raw)
        bins, soft_codes = self.encode_prepared(x_prep, batch_size)
        return bins, soft_codes, metadata

    def encode_prepared(self, x_prep: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode normalized model inputs into per-plane bin indices and soft code posteriors."""
        soft_code_batches: list[torch.Tensor] = []

        def encode_batch(start: int, end: int) -> torch.Tensor:
            bins, soft_codes = self.coding_model.encode(x_prep[start:end], force_softmax=True)
            soft_code_batches.append(torch.stack(soft_codes).cpu())
            return torch.stack(bins).cpu()
        bins = batch_loop(encode_batch, self.coding_model, x_prep.shape[0], batch_size, cat_dim=1)
        dtype = torch.uint8 if self.bins_per_plane <= 2**8 else torch.uint16
        return bins.to(dtype), torch.cat(soft_code_batches, dim=1)

    def decoding_stages_process(
        self,
        payload_content: tuple[torch.Tensor, PreprocessMetadata],
        batch_size: int = INFERENCE_BATCH,
    ) -> torch.Tensor:
        """Decode and postprocess the reconstruction produced by every refinement stage."""
        bins, metadata = payload_content
        assert self.vector_size is not None
        assert bins.shape == (self.num_planes, self.vector_size)
        side_info = self.side_info_tensor(metadata)
        normalized = self.decode_stages_prepared(bins, side_info, batch_size)
        return torch.stack([self.postprocess(stage, metadata) for stage in normalized])

    def decode_stages_prepared(
        self,
        bins: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Decode every normalized reconstruction stage from bins and side information."""
        def decode_batch(start: int, end: int) -> torch.Tensor:
            codes = [
                F.one_hot(plane.long().to(self.device), num_classes=self.bins_per_plane).float()
                for plane in bins[:, start:end]
            ]
            return torch.stack(self.coding_model.decode(codes, side_info[start:end])).cpu()

        return batch_loop(
            decode_batch, self.coding_model, bins.shape[1], batch_size, cat_dim=1
        ).squeeze(-1)

    def decode_prepared(
        self,
        bins: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Decode the final normalized reconstruction stage."""
        return self.decode_stages_prepared(bins, side_info, batch_size)[-1]

    def preprocess_x(
        self,
        x_raw: torch.Tensor,
        skip_outliers: bool = False,
        normalization_factors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PreprocessMetadata]:
        """Normalize a raw vector per slice, optionally extract outliers, and shape it for the model."""
        if self.vector_size is None:
            self.vector_size = x_raw.numel()
        assert self.vector_size == x_raw.numel(), f"Expected vector size {self.vector_size}, got {x_raw.numel()}."

        x_prep = x_raw.clone()
        assert normalization_factors is None or normalization_factors.shape == (len(self.norm_slices), 2)
        norm_factors: list[tuple[float, float]] = []
        for index, vector_slice in enumerate(self.norm_slices):
            scale, center = (
                normalization_params(x_prep[vector_slice])
                if normalization_factors is None
                else map(float, normalization_factors[index])
            )
            x_prep[vector_slice] = (x_prep[vector_slice] - center) / scale
            norm_factors.append((scale, center))

        outliers = OutlierMetadata((), None, np.array([], dtype=np.bool_))
        if self.outlier_threshold is not None and not skip_outliers:
            x_prep, outliers = outlier_metadata(x_prep, self.outlier_threshold)

        metadata = PreprocessMetadata(torch.tensor(norm_factors, dtype=torch.float32), outliers)
        return x_prep.to(self.device).unsqueeze(1).to(torch.float32).contiguous(), metadata

    def postprocess(self, recons_raw: torch.Tensor, metadata: PreprocessMetadata) -> torch.Tensor:
        """Invert outlier handling and per-slice normalization."""
        recons = recons_raw.clone()
        if metadata.outliers.positions:
            positions = torch.from_numpy(np.concatenate([
                chunk.astype(np.int64) + start
                for chunk, start in zip(
                    metadata.outliers.positions,
                    range(0, recons.numel(), OUTLIER_CHUNK),
                    strict=True,
                )
            ])).to(recons.device)
            signs = torch.from_numpy(metadata.outliers.signs).to(recons.device, torch.float32) * 2 - 1
            assert signs.numel() == positions.numel()
            assert metadata.outliers.scale is not None and self.outlier_threshold is not None
            outlier_scale = float(metadata.outliers.scale)
            recons[positions] = (recons[positions].abs() * outlier_scale + self.outlier_threshold) * signs

        for vector_slice, (scale, center) in zip(self.norm_slices, metadata.norm_factors, strict=True):
            recons[vector_slice] = recons[vector_slice] * scale + center
        return recons

    def side_info_tensor(self, source_metadata: PreprocessMetadata) -> SideInfoMatrix:
        """Normalize and stack the quantizer's stored side information."""
        assert self.side_info_list_used is not None, "Side information must be set before formatting."
        return self.normalize_side_info_tensor(source_metadata, self.side_info_list_used)

    def normalize_side_info_tensor(
        self,
        source_metadata: PreprocessMetadata,
        side_information: Sequence[HistoryVector],
    ) -> SideInfoMatrix:
        """Stream decoder-visible history into one target-normalized device matrix."""
        assert self.vector_size is not None, "Vector size must be known before side information is formatted."
        assert all(value.numel() == self.vector_size for value in side_information), (
            f"Each side-information vector must contain {self.vector_size} values.")

        return SideInfoMatrix(
            side_information, self.vector_size, source_metadata.norm_factors,
            self.norm_slices, self.device,
            self.side_info_columns_cache.get(
                side_information, self.vector_size, self.norm_slices, self.device),
        )

    def set_side_information(self, side_info: Sequence[HistoryVector]) -> None:
        """Set the decoder-available side-information vectors used by decoding and conditional priors."""
        assert not self.no_side_info, "A no-side-information quantizer cannot accept side information."
        assert len(side_info) == self.side_info_size, (
            f"Quantizer requires {self.side_info_size} side-information vectors, got {len(side_info)}.")
        assert self.vector_size is None or all(
            value.numel() == self.vector_size for value in side_info
        ), (
            f"Each side-information vector must contain {self.vector_size} values.")
        assert self.side_info_list_used is None, "Side information can only be set once per quantizer instance."
        self.side_info_list_used = list(side_info)


def _train_quantizer_parameters(
    quantizer_class: type[WZQuantizerCancer],
    c_cfg: WZcfgQuant,
    x_raw: torch.Tensor,
    side_information: tuple[HistoryVector, ...],
    extra_training_pairs: tuple[tuple[torch.Tensor, tuple[HistoryVector, ...]], ...] = (),
    problem_weights: tuple[float, ...] | None = None,
    side_info_columns_cache: SideInfoColumnsCache | None = None,
    warm_start_states: tuple[dict[str, torch.Tensor], ...] = (),
) -> tuple[dict[str, torch.Tensor], tuple[float, ...]] | tuple[
    dict[str, torch.Tensor], tuple[float, ...], tuple[dict[str, torch.Tensor], ...], tuple[float, ...], int
]:
    # Runs only on a cache miss, so the keyed seed is applied here, and forked so a hit and a miss
    # leave the caller's RNG identical for later stochastic steps in the same client-round.
    seeded = c_cfg.training_seed is not None
    with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=seeded):
        if seeded:
            torch.manual_seed(c_cfg.training_seed)
        quantizer = quantizer_class(
            c_cfg=c_cfg, si_size=len(side_information), side_info_columns_cache=side_info_columns_cache)
        if warm_start_states:
            quantizer.load_warm_start_states(warm_start_states)
        quantizer._train_model_uncached(x_raw, side_information, extra_training_pairs, problem_weights)
    assert quantizer.quantizer_prior_stage_rates is not None
    if not warm_start_states:
        return quantizer.coding_model.state_dict(), quantizer.quantizer_prior_stage_rates
    assert quantizer.warm_deployed_parent is not None
    return (
        quantizer.coding_model.state_dict(), quantizer.quantizer_prior_stage_rates,
        quantizer.warm_retrained_states, quantizer.warm_retrained_losses,
        quantizer.warm_deployed_parent,
    )


def _rows_for_grouping(block: torch.Tensor | SideInfoMatrix | _SideInfoRows, rows: torch.Tensor) -> torch.Tensor:
    """Select rows from a grouping block while preserving compact SI code columns."""
    if isinstance(block, (SideInfoMatrix, _SideInfoRows)):
        return block.code_rows(rows)
    return block.flatten(start_dim=1)[rows]


def unique_row_groups(
    blocks: Sequence[torch.Tensor | SideInfoMatrix | _SideInfoRows],
    match_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group rows agreeing on every column of every block, floats compared on a leading-bit prefix.

    Columns are folded into a single float64 key one at a time, because materializing the
    full-width row matrix costs gigabytes at full vector size. Distinct rows can in principle
    share a key, so rows that do not match their representative are returned for exact
    recomputation. Returns the representative row of each group, each row's group, and the
    colliding rows.
    """
    assert blocks
    device = blocks[0].device
    size = blocks[0].shape[0]
    assert all(
        block.shape[0] == size
        and (isinstance(block, (SideInfoMatrix, _SideInfoRows)) or block.ndim >= 2)
        for block in blocks
    )

    def columns() -> Iterator[torch.Tensor]:
        for block in blocks:
            if isinstance(block, (SideInfoMatrix, _SideInfoRows)):
                yield from block.iter_code_columns()
                continue
            flattened = block.flatten(start_dim=1)
            for index in range(flattened.shape[1]):
                column = flattened[:, index].contiguous()
                yield (
                    float_prefix(column, match_bits) if column.dtype == torch.float32
                    else column.to(torch.float64)
                )

    generator = torch.Generator(device=device).manual_seed(0)
    projection = torch.randn(
        sum(block.shape[1] if isinstance(block, (SideInfoMatrix, _SideInfoRows)) else block.flatten(start_dim=1).shape[1]
            for block in blocks),
        generator=generator, device=device, dtype=torch.float64,
    )

    key = torch.zeros(size, device=device, dtype=torch.float64)
    for weight, column in zip(projection, columns(), strict=True):
        key += column.to(torch.float64) * weight

    unique_keys, inverse = torch.unique(key, return_inverse=True)
    # amin, not a plain scatter: rows agreeing only on their prefix decode to slightly different
    # values, so an arbitrary duplicate-index winner would make repeated decodes disagree.
    representatives = torch.full((unique_keys.numel(),), size, dtype=torch.long, device=device)
    representatives.scatter_reduce_(0, inverse, torch.arange(size, device=device), "amin")

    reference = representatives[inverse]
    mismatched = torch.zeros(size, dtype=torch.bool, device=device)
    for column in columns():
        mismatched |= column != column[reference]
    return representatives, inverse, mismatched.nonzero(as_tuple=False).flatten()


class DedupedWZQuantizerCancer(WZQuantizerCancer):
    """WZ quantizer that reuses encoder, decoder, and prior results across float-prefix rows."""

    prior_calculator: ClassVar[type[PriorCalculator]] = DedupedPriorCalculator
    match_bits: ClassVar[int] = 24

    def encode_prepared(
        self,
        x_prep: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one representative per normalized float-prefix group."""
        representatives, inverse, collisions = unique_row_groups((x_prep,), self.match_bits)
        bins, soft_codes = super().encode_prepared(x_prep[representatives], batch_size)
        inverse_cpu = inverse.cpu()
        bins = bins[:, inverse_cpu]
        soft_codes = soft_codes[:, inverse_cpu]
        if collisions.numel():
            collision_bins, collision_codes = super().encode_prepared(
                x_prep[collisions], batch_size
            )
            collisions_cpu = collisions.cpu()
            bins[:, collisions_cpu] = collision_bins
            soft_codes[:, collisions_cpu] = collision_codes
        return bins, soft_codes

    def decode_stages_prepared(
        self,
        bins: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Decode every stage once per symbol-and-SI float-prefix group."""
        bins_device = bins.to(self.device, torch.long)
        representatives, inverse, collisions = unique_row_groups(
            (side_info, bins_device.T), self.match_bits
        )
        recons = super().decode_stages_prepared(
            bins_device[:, representatives], select_si_rows(side_info, representatives), batch_size
        ).to(self.device)[:, inverse]
        if collisions.numel():
            recons[:, collisions] = super().decode_stages_prepared(
                bins_device[:, collisions], select_si_rows(side_info, collisions), batch_size
            ).to(self.device)

        return recons.cpu()
