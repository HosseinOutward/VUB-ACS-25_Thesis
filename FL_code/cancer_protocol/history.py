"""Decoder-visible reconstruction history and its in-memory representations."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch

from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.utils import normalization_params


_PREFIX_BITS = 18
_CHUNK_ELEMENTS = 1 << 20
_COMPRESSION_WORKERS = min(4, os.cpu_count() or 1)
_MIN_PACKING_SAVINGS = 0.1
_NORMALIZATION_REFRESH_ROUNDS = 5
_STORAGE_FORMAT = "normalized_adaptive_split_packed_v2"


def float_prefix(values: torch.Tensor, match_bits: int = _PREFIX_BITS) -> torch.Tensor:
    """Truncate float32 values to a shared leading-bit prefix."""
    assert values.dtype is torch.float32
    assert 1 <= match_bits <= 32
    if match_bits == 32:
        return values
    return (values.view(torch.int32) & -(1 << (32 - match_bits))).view(torch.float32)


def _normalization_ranges(size: int, slices: Sequence[slice] | None) -> tuple[tuple[int, int], ...]:
    """Resolve contiguous layer slices covering one complete history vector."""
    if size == 0:
        return ()
    resolved = ((0, size, 1),) if slices is None else tuple(value.indices(size) for value in slices)
    assert all(step == 1 and start < stop for start, stop, step in resolved), (
        "History normalization slices must be non-empty contiguous ranges."
    )
    ranges = tuple((start, stop) for start, stop, _ in resolved)
    assert ranges and ranges[0][0] == 0 and ranges[-1][1] == size
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])), (
        "History normalization slices must cover the vector exactly once and in order."
    )
    return ranges


def _normalize_history(
    value: torch.Tensor,
    ranges: tuple[tuple[int, int], ...],
    supplied_factors: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each layer using supplied factors or freshly estimated scale and center."""
    if not ranges:
        return value.clone(), torch.empty((0, 2), dtype=torch.float32)
    if supplied_factors is None:
        def estimate(bounds: tuple[int, int]) -> tuple[float, float]:
            """Estimate one layer independently so refresh work can run concurrently."""
            start, stop = bounds
            layer = value[start:stop]
            minimum, maximum = torch.aminmax(layer)
            if minimum == maximum:
                return 1.0, float(minimum)
            magnitude = max(abs(float(minimum)), abs(float(maximum)))
            scale, center = normalization_params(layer / magnitude)
            return scale * magnitude, center * magnitude

        if len(ranges) <= 1 or _COMPRESSION_WORKERS == 1:
            factors = tuple(map(estimate, ranges))
        else:
            with ThreadPoolExecutor(max_workers=min(_COMPRESSION_WORKERS, len(ranges))) as executor:
                factors = tuple(executor.map(estimate, ranges))
        resolved = torch.tensor(factors, dtype=torch.float32)
    else:
        assert supplied_factors.shape == (len(ranges), 2)
        resolved = supplied_factors.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    assert bool(torch.isfinite(resolved).all()) and bool((resolved[:, 0] != 0).all()), (
        "History normalization factors must contain finite, non-zero scales and finite centers."
    )

    normalized = value.clone()
    for (start, stop), (scale, center) in zip(ranges, resolved, strict=True):
        normalized[start:stop].sub_(center).div_(scale)
    return normalized, resolved


class HistoryVector:
    """Owned CPU float32 history values consumed as decoder side information."""

    def __init__(self, value: torch.Tensor) -> None:
        assert (
            isinstance(value, torch.Tensor)
            and value.dtype is torch.float32
            and value.device.type == "cpu"
            and value.ndim == 1
        ), "History value must be a one-dimensional CPU float32 tensor."
        self._value: torch.Tensor = value.detach().contiguous().clone()

    @property
    def logical_size(self) -> int:
        """Return the number of values represented by this vector."""
        return self._value.numel()

    def numel(self) -> int:
        """Return the logical number of reconstructed values."""
        return self.logical_size

    @property
    def resident_nbytes(self) -> int:
        """Return resident bytes occupied by the owned float32 representation."""
        return self.logical_size * 4

    @property
    def nbytes(self) -> int:
        """Return resident bytes occupied by this history representation."""
        return self.resident_nbytes

    def iter_chunks(self) -> Iterator[tuple[slice, torch.Tensor]]:
        """Yield owned float32 chunks in their original coordinate order."""
        for start in range(0, self.logical_size, _CHUNK_ELEMENTS):
            stop = min(start + _CHUNK_ELEMENTS, self.logical_size)
            yield slice(start, stop), self._value[start:stop].clone()

    def materialize(self) -> torch.Tensor:
        """Return an owned float32 copy of the complete vector."""
        return self._value.clone()


@dataclass(frozen=True, slots=True)
class _CompressedHistoryChunk:
    """One raw or dictionary-coded coordinate range with device-side row access."""

    start: int
    stop: int
    values: torch.Tensor
    codes: torch.Tensor | None = None
    packed_high: torch.Tensor | None = None
    code_bits: int | None = None

    def __post_init__(self) -> None:
        """Validate the selected physical layout's tensor contract."""
        assert self.start <= self.stop and self.values.dtype is torch.float32 and self.values.ndim == 1
        if self.codes is None:
            assert self.packed_high is None and self.code_bits is None
            assert self.values.numel() == self.stop - self.start
            return
        assert self.codes.ndim == 1 and self.codes.numel() == self.stop - self.start
        assert self.codes.dtype in (torch.uint8, torch.uint16, torch.uint32)
        assert self.codes.device == self.values.device
        assert self.code_bits is not None and 1 <= self.code_bits <= _PREFIX_BITS
        assert self.packed_high is None or (
            self.codes.dtype is torch.uint8
            and self.packed_high.dtype is torch.uint8
            and self.packed_high.ndim == 1
            and self.packed_high.device == self.values.device
            and self.code_bits > 8
        )

    @property
    def layout(self) -> str:
        """Return the physical layout selected for this chunk."""
        if self.codes is None:
            return "raw"
        return "packed" if self.packed_high is not None else "direct"

    @property
    def dictionary(self) -> torch.Tensor | None:
        """Return dictionary values, or ``None`` for a raw prefixed chunk."""
        return None if self.codes is None else self.values

    @property
    def nbytes(self) -> int:
        """Return bytes retained by this chunk's tensor payloads."""
        tensors = (self.values, self.codes, self.packed_high)
        return sum(item.numel() * item.element_size() for item in tensors if item is not None)

    def to(self, device: torch.device) -> _CompressedHistoryChunk:
        """Copy this immutable chunk representation to ``device``."""
        return _CompressedHistoryChunk(
            self.start,
            self.stop,
            self.values.to(device),
            None if self.codes is None else self.codes.to(device),
            None if self.packed_high is None else self.packed_high.to(device),
            self.code_bits,
        )

    def ranks(self, indices: torch.Tensor) -> torch.Tensor:
        """Return dictionary ranks at local chunk coordinates."""
        assert self.codes is not None and self.code_bits is not None
        if self.packed_high is None:
            return self.codes.to(torch.int32)[indices]

        high_bits = self.code_bits - 8
        if indices.is_cuda:
            from .packed_history_cuda import packed_ranks
            return packed_ranks(self.codes, self.packed_high, indices, high_bits)

        offsets = indices * high_bits
        byte_indices, shifts = offsets >> 3, offsets & 7
        words = self.packed_high[byte_indices].to(torch.int32)
        words |= self.packed_high[byte_indices + 1].to(torch.int32) << 8
        if high_bits > 9:
            words |= self.packed_high[byte_indices + 2].to(torch.int32) << 16
        high = (words >> shifts) & ((1 << high_bits) - 1)
        return self.codes[indices].to(torch.int32) | (high << 8)

    def local_keys(self, indices: torch.Tensor) -> torch.Tensor:
        """Return stable unsigned local keys without decoding float values."""
        if self.codes is not None:
            return self.ranks(indices).to(torch.int64)
        bits = self.values[indices].view(torch.int32).to(torch.int64)
        return bits & 0xFFFF_FFFF

    def decode_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode float32 values at local chunk coordinates."""
        if self.codes is None:
            return self.values[indices]
        if self.packed_high is not None and indices.is_cuda:
            from .packed_history_cuda import packed_values
            assert self.code_bits is not None
            return packed_values(
                self.codes, self.packed_high, self.values, indices, self.code_bits - 8
            )
        return self.values[self.ranks(indices).long()]

    def decode_all(self) -> torch.Tensor:
        """Decode this complete CPU chunk."""
        if self.codes is None:
            return self.values.clone()
        if self.packed_high is None:
            return self.values[self.codes.to(torch.int64)]

        assert self.code_bits is not None
        high_bits = self.code_bits - 8
        packed = self.packed_high[:-2].numpy()
        bit_rows = np.unpackbits(
            packed, count=self.codes.numel() * high_bits, bitorder="little"
        ).reshape(self.codes.numel(), high_bits)
        high = bit_rows.astype(np.uint32) @ (np.uint32(1) << np.arange(high_bits, dtype=np.uint32))
        ranks = self.codes.numpy().astype(np.uint32) | (high << 8)
        return self.values[torch.from_numpy(ranks.astype(np.int64))]


def _pack_high_codes(codes: np.ndarray, code_bits: int) -> torch.Tensor:
    """Pack rank bits above the directly indexable low byte."""
    high_bits = code_bits - 8
    high = codes.astype(np.uint32, copy=False) >> 8
    shifts = np.arange(high_bits, dtype=np.uint32)
    bit_rows = ((high[:, None] >> shifts) & 1).astype(np.uint8)
    # Two sentinels let arbitrary-row decoding load a three-byte window without a boundary branch.
    packed = np.concatenate((np.packbits(bit_rows.reshape(-1), bitorder="little"), np.zeros(2, dtype=np.uint8)))
    return torch.from_numpy(packed)


def _compress_chunk(value: torch.Tensor, start: int, stop: int) -> _CompressedHistoryChunk:
    """Prefix one chunk and choose its smallest worthwhile random-access layout."""
    prefixed = float_prefix(value[start:stop])
    bits = prefixed.view(torch.int32).numpy()
    unique, inverse = np.unique(bits, return_inverse=True)
    code_bits = max(1, (unique.size - 1).bit_length())
    code_dtype = np.uint8 if code_bits <= 8 else np.uint16 if code_bits <= 16 else np.uint32
    direct_nbytes = unique.nbytes + inverse.size * np.dtype(code_dtype).itemsize
    packed_nbytes = (
        unique.nbytes + inverse.size + (inverse.size * (code_bits - 8) + 7) // 8 + 2
        if code_bits > 8 else None
    )
    raw_nbytes = prefixed.numel() * prefixed.element_size()
    use_packed = packed_nbytes is not None and packed_nbytes <= direct_nbytes * (1 - _MIN_PACKING_SAVINGS)

    if raw_nbytes <= min(direct_nbytes, packed_nbytes if use_packed else direct_nbytes):
        return _CompressedHistoryChunk(start, stop, prefixed)

    dictionary = torch.from_numpy(unique.astype(np.int32, copy=True)).view(torch.float32)
    if use_packed:
        low_codes = torch.from_numpy((inverse & 0xFF).astype(np.uint8))
        return _CompressedHistoryChunk(
            start, stop, dictionary, low_codes, _pack_high_codes(inverse, code_bits), code_bits
        )

    codes = torch.from_numpy(inverse.astype(code_dtype, copy=True))
    return _CompressedHistoryChunk(start, stop, dictionary, codes, code_bits=code_bits)


class CompressedHistoryVector(HistoryVector):
    """Layer-normalized, adaptively packed CPU history with the HistoryVector read contract."""

    def __init__(
        self,
        value: torch.Tensor,
        norm_slices: Sequence[slice] | None = None,
        normalization_factors: torch.Tensor | None = None,
    ) -> None:
        assert (
            isinstance(value, torch.Tensor)
            and value.dtype is torch.float32
            and value.device.type == "cpu"
            and value.ndim == 1
        ), "History value must be a one-dimensional CPU float32 tensor."
        source = value.detach().contiguous()
        self._logical_size: int = source.numel()
        self._normalization_ranges = _normalization_ranges(source.numel(), norm_slices)
        source, self._normalization_factors = _normalize_history(
            source, self._normalization_ranges, normalization_factors
        )
        bounds = tuple(
            (source, start, min(start + _CHUNK_ELEMENTS, source.numel()))
            for start in range(0, source.numel(), _CHUNK_ELEMENTS)
        )
        if len(bounds) <= 1 or _COMPRESSION_WORKERS == 1:
            self._chunks = tuple(_compress_chunk(*bound) for bound in bounds)
        else:
            with ThreadPoolExecutor(max_workers=min(_COMPRESSION_WORKERS, len(bounds))) as executor:
                self._chunks = tuple(executor.map(lambda bound: _compress_chunk(*bound), bounds))

    def __getstate__(self) -> dict[str, Any]:
        """Serialize unsigned direct codes through same-width signed torch dtypes."""
        chunks = tuple({
            "start": chunk.start,
            "stop": chunk.stop,
            "values": chunk.values,
            "codes": (
                chunk.codes.view(torch.int16) if chunk.codes is not None and chunk.codes.dtype is torch.uint16
                else chunk.codes.view(torch.int32) if chunk.codes is not None and chunk.codes.dtype is torch.uint32
                else chunk.codes
            ),
            "codes_dtype": None if chunk.codes is None else str(chunk.codes.dtype),
            "packed_high": chunk.packed_high,
            "code_bits": chunk.code_bits,
        } for chunk in self._chunks)
        return {
            "format": _STORAGE_FORMAT,
            "logical_size": self._logical_size,
            "normalization_ranges": self._normalization_ranges,
            "normalization_factors": self._normalization_factors,
            "chunks": chunks,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore adaptive chunks and narrow unsigned direct-code dtypes."""
        assert state["format"] == _STORAGE_FORMAT, f"Unknown compressed-history format: {state['format']!r}."
        dtypes = {"torch.uint8": torch.uint8, "torch.uint16": torch.uint16, "torch.uint32": torch.uint32}
        self._logical_size = state["logical_size"]
        self._normalization_ranges = state["normalization_ranges"]
        self._normalization_factors = state["normalization_factors"]
        self._chunks = tuple(
            _CompressedHistoryChunk(
                item["start"],
                item["stop"],
                item["values"],
                None if item["codes"] is None else item["codes"].view(dtypes[item["codes_dtype"]]),
                item["packed_high"],
                item["code_bits"],
            )
            for item in state["chunks"]
        )

    @property
    def logical_size(self) -> int:
        """Return the number of values represented by this vector."""
        return self._logical_size

    def _iter_encoded_chunks(self) -> Iterator[_CompressedHistoryChunk]:
        """Yield dictionary and code tensors to the colocated storage consumer."""
        yield from self._chunks

    @property
    def normalization_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return the layer ranges used by storage normalization."""
        return self._normalization_ranges

    @property
    def normalization_factors(self) -> torch.Tensor:
        """Return an owned copy of per-layer ``(scale, center)`` storage factors."""
        return self._normalization_factors.clone()

    @property
    def resident_nbytes(self) -> int:
        """Return bytes occupied by raw values, dictionaries, and code arrays."""
        return self._normalization_factors.numel() * self._normalization_factors.element_size() + sum(
            chunk.nbytes for chunk in self._chunks
        )

    def _restore_normalization(self, values: torch.Tensor, start: int) -> torch.Tensor:
        """Invert every stored layer transform overlapping one decoded range."""
        stop = start + values.numel()
        for (layer_start, layer_stop), (scale, center) in zip(
            self._normalization_ranges, self._normalization_factors, strict=True
        ):
            overlap_start, overlap_stop = max(start, layer_start), min(stop, layer_stop)
            if overlap_start < overlap_stop:
                values[overlap_start - start:overlap_stop - start].mul_(scale).add_(center)
        return values

    def iter_chunks(self) -> Iterator[tuple[slice, torch.Tensor]]:
        """Decode and yield compressed chunks in their original coordinate order."""
        for chunk in self._chunks:
            yield slice(chunk.start, chunk.stop), self._restore_normalization(
                chunk.decode_all(), chunk.start
            )

    def materialize(self) -> torch.Tensor:
        """Explicitly decode the complete vector, including prefix quantization."""
        if len(self._chunks) <= 1 or _COMPRESSION_WORKERS == 1:
            chunks = [chunk.decode_all() for chunk in self._chunks]
        else:
            with ThreadPoolExecutor(max_workers=min(_COMPRESSION_WORKERS, len(self._chunks))) as executor:
                chunks = list(executor.map(_CompressedHistoryChunk.decode_all, self._chunks))
        normalized = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
        return self._restore_normalization(normalized, 0)


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One reconstructed delta and the decoder visibility granted to its consumers."""

    value: HistoryVector
    round_id: int
    client_id: int
    access: Access
    round_name: str | None


class ReconstructionHistory:
    """Bounded decoder-visible ledger retaining exact float32 HistoryVector values."""

    def __init__(self, max_per_client: int | None) -> None:
        assert max_per_client != 0, "to disable reconstruction history, set max_per_client to None"
        self.max_per_client: int = max_per_client or 0
        self._server: dict[int, list[HistoryEntry]] = {}
        self._temporal: dict[int, list[HistoryEntry]] = {}
        self.frozen: bool = self.max_per_client == 0

    def config_snapshot(self) -> dict[str, Any]:
        """Describe the history implementation and fixed storage constants."""
        normalization_slices = (
            tuple((value.start, value.stop, value.step) for value in self._norm_slices)
            if isinstance(self, CompressedReconstructionHistory) and self._norm_slices is not None
            else None
        )
        return {
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "max_per_client": self.max_per_client,
            "prefix_bits": _PREFIX_BITS if isinstance(self, CompressedReconstructionHistory) else 32,
            "chunk_elements": _CHUNK_ELEMENTS,
            "storage_format": _STORAGE_FORMAT if isinstance(self, CompressedReconstructionHistory) else "float32",
            "normalization_refresh_rounds": (
                _NORMALIZATION_REFRESH_ROUNDS if isinstance(self, CompressedReconstructionHistory) else None
            ),
            "normalization_slices": normalization_slices,
        }

    def copy(self) -> ReconstructionHistory:
        """Copy ledger lists while sharing immutable entries and their payloads."""
        copied = copy.copy(self)
        copied._server = {client_id: entries.copy() for client_id, entries in self._server.items()}
        copied._temporal = {client_id: entries.copy() for client_id, entries in self._temporal.items()}
        return copied

    def _retain(self, value: torch.Tensor, round_id: int, client_id: int) -> HistoryVector:
        """Retain one committed reconstruction in the reference representation."""
        return HistoryVector(value)

    def commit(self, reconst: torch.Tensor, record: CompressionRecord, access: Access) -> None:
        """Retain a decoded reconstruction in each ledger permitted by its access policy."""
        assert not self.frozen, "ReconstructionHistory is frozen and cannot be modified."
        assert reconst.dtype is torch.float32 and reconst.device == torch.device("cpu") and reconst.ndim == 1
        if access is Access.NONE:
            return
        entry = HistoryEntry(
            value=self._retain(reconst, record.round_id, record.client_id),
            round_id=record.round_id,
            client_id=record.client_id,
            access=access,
            round_name=record.round_name_full,
        )
        self._add_entry(self._server, entry)
        if access is Access.TEMPORAL_TOO:
            self._add_entry(self._temporal, entry)

    def replace_latest(self, reconst: torch.Tensor, round_id: int, client_id: int) -> None:
        """Replace a client's latest entry while preserving its metadata and ledger access."""
        assert not self.frozen, "ReconstructionHistory is frozen and cannot be modified."
        assert reconst.dtype is torch.float32 and reconst.device == torch.device("cpu") and reconst.ndim == 1
        server_ledger = self._server.get(client_id)
        assert server_ledger and server_ledger[-1].round_id == round_id, (
            f"No latest reconstruction exists for round {round_id}, client {client_id}."
        )
        previous = server_ledger[-1]
        replacement = HistoryEntry(
            value=self._retain(reconst, round_id, client_id),
            round_id=previous.round_id,
            client_id=previous.client_id,
            access=previous.access,
            round_name=previous.round_name,
        )
        server_ledger[-1] = replacement
        if previous.access is Access.TEMPORAL_TOO:
            temporal_ledger = self._temporal.get(client_id)
            assert temporal_ledger and temporal_ledger[-1] is previous
            temporal_ledger[-1] = replacement

    def view(self, access: Access, client_id: int) -> dict[int, tuple[HistoryEntry, ...]]:
        """Return only history entries visible at the requested decoder location."""
        if access is Access.NONE:
            return {}
        if access is Access.TEMPORAL_TOO:
            assert client_id in self._temporal, f"No temporal reconstruction history exists for client {client_id}."
            return {client_id: tuple(self._temporal[client_id])}
        assert access is Access.SERVER_ONLY, f"Unknown access policy: {access!r}."
        return {visible_client: tuple(ledger) for visible_client, ledger in self._server.items()}

    def _add_entry(self, history: dict[int, list[HistoryEntry]], entry: HistoryEntry) -> None:
        ledger = history.setdefault(entry.client_id, [])
        ledger.append(entry)
        if len(ledger) > self.max_per_client:
            ledger.pop(0)


class CompressedReconstructionHistory(ReconstructionHistory):
    """ReconstructionHistory variant retaining every value in compressed chunk storage."""

    def __init__(self, max_per_client: int | None, norm_slices: Sequence[slice] | None = None) -> None:
        super().__init__(max_per_client)
        self._norm_slices = None if norm_slices is None else tuple(norm_slices)
        self._normalization_cache: dict[int, tuple[int, torch.Tensor]] = {}

    def copy(self) -> CompressedReconstructionHistory:
        """Copy ledger and normalization-cache mappings while sharing immutable payloads."""
        copied = super().copy()
        assert isinstance(copied, CompressedReconstructionHistory)
        copied._normalization_cache = self._normalization_cache.copy()
        return copied

    def _retain(self, value: torch.Tensor, round_id: int, client_id: int) -> HistoryVector:
        """Normalize with a periodically refreshed client factor set, then retain compactly."""
        cached = self._normalization_cache.get(client_id)
        refresh = cached is None or round_id < cached[0] or round_id - cached[0] >= _NORMALIZATION_REFRESH_ROUNDS
        stored = CompressedHistoryVector(
            value,
            self._norm_slices,
            None if refresh else cached[1],
        )
        if refresh:
            self._normalization_cache[client_id] = (round_id, stored.normalization_factors)
        return stored


__all__ = [
    "_PREFIX_BITS",
    "_CHUNK_ELEMENTS",
    "float_prefix",
    "HistoryVector",
    "CompressedHistoryVector",
    "HistoryEntry",
    "ReconstructionHistory",
    "CompressedReconstructionHistory",
]
