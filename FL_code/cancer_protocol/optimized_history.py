from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from itertools import islice
import math
import os
import shutil
import uuid
import weakref
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
import torch

from FL_code.FL_core.utils import normalization_params


class HistoryStorageConfig(BaseModel):
    """Representation used to retain decoder-visible reconstruction history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalize: bool = False
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    deduplicate: bool = False
    disk: bool = False
    chunk_elements: int = 1 << 20

    @field_validator("chunk_elements")
    @classmethod
    def _positive_chunk_elements(cls, value: int) -> int:
        assert value > 0, "chunk_elements must be positive."
        return value


_RANK_BLOCK_SIZE = 1 << 10
_DEDUP_WORKERS = min(8, os.cpu_count() or 1)

# Torch storage dtype, its equal-width unsigned bit-pattern dtype, and the portable NumPy dtype.
_STORAGE_DTYPES: dict[str, tuple[torch.dtype, torch.dtype, np.dtype]] = {
    "float32": (torch.float32, torch.uint32, np.dtype(np.uint32)),
    "float16": (torch.float16, torch.uint16, np.dtype(np.uint16)),
    "bfloat16": (torch.bfloat16, torch.uint16, np.dtype(np.uint16)),
}


def _pack_codes(codes: np.ndarray, code_bits: int) -> np.ndarray:
    """Pack fixed-width integer codes into little-endian bits."""
    codes = np.asarray(codes)
    assert codes.ndim == 1 and 1 <= code_bits <= 16
    if codes.size == 0:
        return np.empty(0, dtype=np.uint8)
    bits = ((codes.astype(np.uint32)[:, None] >> np.arange(code_bits, dtype=np.uint32)) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="little")


def _codes_at(packed: np.ndarray, indices: np.ndarray, code_bits: int) -> np.ndarray:
    """Read selected fixed-width little-endian codes without unpacking the full array."""
    packed = np.asarray(packed)
    indices = np.asarray(indices, dtype=np.int64)
    assert indices.ndim == 1 and 1 <= code_bits <= 16
    if indices.size == 0:
        return np.empty(0, dtype=np.uint8 if code_bits <= 8 else np.uint16)
    offsets = indices * code_bits
    byte_indices, shifts = offsets // 8, offsets % 8
    words = packed[byte_indices].astype(np.uint32, copy=True)
    for byte_offset in (1, 2):
        valid = byte_indices + byte_offset < packed.size
        if np.any(valid):
            words[valid] |= packed[byte_indices[valid] + byte_offset].astype(np.uint32) << (8 * byte_offset)
    dtype = np.uint8 if code_bits <= 8 else np.uint16
    return ((words >> shifts.astype(np.uint32)) & (2**code_bits - 1)).astype(dtype)


@dataclass(frozen=True, slots=True)
class _Chunk:
    """One physical encoded chunk and its coordinate interval."""

    start: int
    stop: int
    kind: Literal["raw", "dictionary"]
    values: np.ndarray
    codes: np.ndarray | None = None
    exceptions: np.ndarray | None = None
    rank_checkpoints: np.ndarray | None = None
    base_code_bits: int | None = None

    @property
    def nbytes(self) -> int:
        """Return encoded array bytes for this chunk."""
        arrays = (self.values, self.codes, self.exceptions, self.rank_checkpoints)
        return sum(array.nbytes for array in arrays if array is not None)


_CHUNK_ARRAY_NAMES: dict[str, tuple[str, ...]] = {
    "raw": ("values",),
    "dictionary": ("values", "codes", "exceptions", "rank_checkpoints"),
}


def _to_storage_bits(values: torch.Tensor, dtype: str) -> np.ndarray:
    """Quantize an owned contiguous chunk once and share its bit patterns without copying."""
    torch_dtype, bits_dtype, _ = _STORAGE_DTYPES[dtype]
    return values.to(torch_dtype).view(bits_dtype).numpy()


def _from_storage_bits(bits: np.ndarray, dtype: str) -> torch.Tensor:
    """Reinterpret owned bit patterns as CPU float32; float32 storage is not copied again."""
    torch_dtype, bits_dtype, _ = _STORAGE_DTYPES[dtype]
    return torch.from_numpy(bits).view(bits_dtype).view(torch_dtype).float()


def _build_dictionary(bits: np.ndarray, dtype: str, start: int, stop: int) -> _Chunk | None:
    """Build the byte-minimal exact two-level dictionary chunk, or None if it would not shrink.

    Each code width's encoded size is a closed form over the rank/frequency histogram, so the
    winning width is picked analytically before any array is built. 16-bit patterns are counted
    and ranked through a 65536-entry table, avoiding the O(n) int64 sort/inverse arrays that
    np.unique needs; 32-bit patterns cannot be tabled and keep np.unique.
    """
    if bits.size == 0:
        return None
    tabled = bits.dtype.itemsize == 2
    if tabled:
        table_counts = np.bincount(bits, minlength=1 << 16)
        unique = np.flatnonzero(table_counts)
        counts = table_counts[unique]
    else:
        unique, inverse, counts = np.unique(bits, return_inverse=True, return_counts=True)
    # Ties rank by value: encoded size depends only on counts, so any fixed tie order is exact.
    order = np.argsort(-counts, kind="stable")
    cumulative = np.cumsum(counts[order])

    value_itemsize = _STORAGE_DTYPES[dtype][2].itemsize
    block_count = math.ceil(bits.size / _RANK_BLOCK_SIZE)
    fixed_bytes = unique.size * value_itemsize + block_count * 4  # dictionary + checkpoints

    best_size, best_bits = None, 1
    for code_bits in range(1, 17):
        escape = 2**code_bits - 1
        if escape >= unique.size:
            exception_count, exception_max = 0, -1
        else:
            exception_count = int(bits.size - cumulative[escape - 1])
            exception_max = int(unique.size - 1 - escape)
        exception_itemsize = 1 if exception_max < 2**8 else 2 if exception_max < 2**16 else 4
        size = fixed_bytes + math.ceil(bits.size * code_bits / 8) + exception_count * exception_itemsize
        if best_size is None or size < best_size:
            best_size, best_bits = size, code_bits
    if best_size >= bits.nbytes:
        return None

    # Ranks are < unique.size, so they fit the bit-pattern dtype itself.
    ranks = np.empty(1 << 16 if tabled else unique.size, dtype=bits.dtype)
    if tabled:
        ranks[unique[order]] = np.arange(unique.size)
        ranked = ranks[bits]
    else:
        ranks[order] = np.arange(unique.size)
        ranked = ranks[inverse]
    escape = 2**best_bits - 1
    base = np.minimum(ranked, escape).astype(np.uint16 if best_bits > 8 else np.uint8)
    escaped = ranked >= escape
    if escaped.any():
        exception_max = int(ranked[escaped].max() - escape)
        exception_dtype = np.uint8 if exception_max < 2**8 else np.uint16 if exception_max < 2**16 else np.uint32
        exceptions = (ranked[escaped] - escape).astype(exception_dtype, copy=True)
    else:
        exceptions = np.empty(0, dtype=np.uint8)
    checkpoints = np.zeros(block_count, dtype=np.uint32)
    if block_count > 1:
        escaped_counts = np.add.reduceat(escaped.astype(np.uint32), np.arange(0, bits.size, _RANK_BLOCK_SIZE))
        checkpoints[1:] = np.cumsum(escaped_counts[:-1], dtype=np.uint32)
    value_dtype = _STORAGE_DTYPES[dtype][2]
    return _Chunk(
        start, stop, "dictionary",
        unique[order].astype(value_dtype, copy=True),
        _pack_codes(base, best_bits), exceptions, checkpoints, best_bits,
    )


def _normalize_factors(
    value: torch.Tensor,
    slices: Sequence[slice],
    supplied: torch.Tensor | None,
) -> tuple[tuple[int, int, float, float], ...]:
    """Resolve layer ranges and use decoder metadata or the quantizer's factor estimator."""
    ranges = tuple(layer.indices(value.numel()) for layer in slices)
    assert all(step == 1 and start < stop for start, stop, step in ranges), (
        "History normalization slices must be nonempty and contiguous.")
    if supplied is None:
        return tuple(
            (start, stop, *normalization_params(value[start:stop]))
            for start, stop, _ in ranges
        )
    assert supplied.shape == (len(ranges), 2), (
        f"Expected {len(ranges)} supplied history normalization factors, got {tuple(supplied.shape)}.")
    return tuple(
        (start, stop, float(scale), float(center))
        for (start, stop, _), (scale, center) in zip(ranges, supplied, strict=True)
    )


def _apply_normalization(chunk: torch.Tensor, start: int, factors: tuple[tuple[int, int, float, float], ...], inverse: bool) -> None:
    """Apply only parameter factors overlapping a chunk in place."""
    stop = start + chunk.numel()
    for factor_start, factor_stop, scale, center in factors:
        overlap_start = max(start, factor_start)
        overlap_stop = min(stop, factor_stop)
        if overlap_start >= overlap_stop:
            continue
        local_start, local_stop = overlap_start - start, overlap_stop - start
        segment = chunk[local_start:local_stop]
        if inverse:
            segment.mul_(scale).add_(center)
        else:
            segment.sub_(center).div_(scale)


def _remove_path(path: Path) -> None:
    """Remove exactly one generated disk entry after its final payload handle is released."""
    shutil.rmtree(path, ignore_errors=True)


class _DiskSession:
    """Keep one generated session alive while its store or any payload still refers to it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cleanup = weakref.finalize(self, _remove_path, path)


class StoredHistoryVector:
    """Immutable stored reconstruction read by protocol consumers in bounded ranges."""

    def __init__(
        self, chunks: tuple[_Chunk, ...], logical_size: int, dtype: str,
        factors: tuple[tuple[int, int, float, float], ...],
        disk_path: Path | None = None, disk_session: _DiskSession | None = None,
    ) -> None:
        self.logical_size = logical_size
        self._chunks = chunks
        self._dtype = dtype
        self._factors = factors
        self._disk_path = disk_path
        self._disk_session = disk_session
        self._cleanup = weakref.finalize(self, _remove_path, disk_path) if disk_path is not None else None

    def _decode_chunk(self, chunk: _Chunk, start: int, stop: int) -> torch.Tensor:
        local_start, local_stop = start - chunk.start, stop - chunk.start
        if chunk.kind == "raw":
            restored = _from_storage_bits(np.array(chunk.values[local_start:local_stop]), self._dtype)
        else:
            assert chunk.codes is not None and chunk.exceptions is not None
            assert chunk.rank_checkpoints is not None and chunk.base_code_bits is not None
            # A requested range is always contiguous: decode a single span starting at the
            # rank-checkpoint block containing local_start, then trim to the requested range.
            block = local_start // _RANK_BLOCK_SIZE
            block_start = block * _RANK_BLOCK_SIZE
            coordinates = np.arange(block_start, local_stop, dtype=np.int64)
            codes = _codes_at(chunk.codes, coordinates, chunk.base_code_bits).astype(np.int64)
            escape = 2**chunk.base_code_bits - 1
            escaped = codes == escape
            if escaped.any():
                exception_ranks = chunk.rank_checkpoints[block] + np.cumsum(escaped) - 1
                codes[escaped] = escape + chunk.exceptions[exception_ranks[escaped]].astype(np.int64)
            codes = codes[local_start - block_start:]
            restored = _from_storage_bits(chunk.values[codes], self._dtype)
        _apply_normalization(restored, start, self._factors, inverse=True)
        return restored

    def read_range(self, start: int, stop: int) -> torch.Tensor:
        """Restore only the requested half-open range as CPU float32 values."""
        assert isinstance(start, int) and isinstance(stop, int) and 0 <= start <= stop <= self.logical_size, (
            f"History read range [{start}:{stop}] is outside [0:{self.logical_size}]."
        )
        if start == stop:
            return torch.empty(0, dtype=torch.float32)
        pieces = [
            self._decode_chunk(chunk, max(start, chunk.start), min(stop, chunk.stop))
            for chunk in self._chunks
            if chunk.start < stop and chunk.stop > start
        ]
        result = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        assert result.numel() == stop - start
        return result

    def iter_chunks(self) -> Iterator[tuple[slice, torch.Tensor]]:
        """Restore configured physical chunks in order without complete-vector allocation."""
        for chunk in self._chunks:
            yield slice(chunk.start, chunk.stop), self.read_range(chunk.start, chunk.stop)

    def materialize(self) -> torch.Tensor:
        """Explicitly restore the complete vector."""
        chunks = [chunk for _, chunk in self.iter_chunks()]
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)

    def update_content_hash(self, digest: Any) -> None:
        """Hash logical float32 values in physical chunks without materializing the vector."""
        for _, values in self.iter_chunks():
            digest.update(values.numpy())

    def numel(self) -> int:
        """Return the logical number of reconstructed values."""
        return self.logical_size

    def normalization_factors(self) -> torch.Tensor | None:
        """Return reusable float32 scale/center pairs for replacing this stored value."""
        if not self._factors:
            return None
        return torch.tensor(
            [(scale, center) for _, _, scale, center in self._factors], dtype=torch.float32)

    @property
    def resident_nbytes(self) -> int:
        """Return encoded arrays and normalization metadata retained in resident memory."""
        metadata = len(self._factors) * 4 * np.dtype(np.float32).itemsize
        return metadata + (sum(chunk.nbytes for chunk in self._chunks) if self._disk_path is None else 0)

    @property
    def disk_nbytes(self) -> int:
        """Return bytes in this payload's manifest and chunk files."""
        if self._disk_path is None or not self._disk_path.exists():
            return 0
        return sum(path.stat().st_size for path in self._disk_path.rglob("*") if path.is_file())


class HistoryStore:
    """Create immutable stored vectors according to one history representation config."""

    def __init__(self, config: HistoryStorageConfig, sd_slices: Sequence[slice] | None, history_root: Path | None, protocol_name: str) -> None:
        self.config = config
        self._slices = tuple(sd_slices) if sd_slices is not None else None
        self._history_root = None if history_root is None else Path(history_root)
        self._protocol_name = protocol_name
        self._session: _DiskSession | None = None

    def _disk_session(self) -> _DiskSession:
        """Create this store's unique session lazily on its first retained disk payload."""
        if self._history_root is None:
            raise RuntimeError(
                f"History storage for protocol {self._protocol_name!r} requires "
                "history_root when disk=True."
            )
        if self._session is None:
            self._history_root.mkdir(parents=True, exist_ok=True)
            session_path = self._history_root / f"session_{uuid.uuid4().hex}"
            session_path.mkdir()
            self._session = _DiskSession(session_path)
        return self._session

    def _encode_chunk(
        self, value: torch.Tensor, factors: tuple[tuple[int, int, float, float], ...],
        start: int, stop: int,
    ) -> _Chunk:
        """Encode one independent physical chunk."""
        transformed = value[start:stop].clone()
        _apply_normalization(transformed, start, factors, inverse=False)
        bits = _to_storage_bits(transformed, self.config.dtype)
        dictionary = _build_dictionary(bits, self.config.dtype, start, stop) if self.config.deduplicate else None
        return dictionary or _Chunk(start, stop, "raw", bits)

    def _encode(self, value: torch.Tensor, factors: tuple[tuple[int, int, float, float], ...]) -> Iterator[_Chunk]:
        """Encode ordered chunks with bounded parallel work only when deduplication benefits."""
        bounds = iter(
            (start, min(start + self.config.chunk_elements, value.numel()))
            for start in range(0, value.numel(), self.config.chunk_elements)
        )
        if not self.config.deduplicate or _DEDUP_WORKERS == 1:
            for start, stop in bounds:
                yield self._encode_chunk(value, factors, start, stop)
            return

        with ThreadPoolExecutor(max_workers=_DEDUP_WORKERS) as executor:
            pending = deque(
                executor.submit(self._encode_chunk, value, factors, start, stop)
                for start, stop in islice(bounds, _DEDUP_WORKERS)
            )
            for start, stop in bounds:
                yield pending.popleft().result()
                pending.append(executor.submit(self._encode_chunk, value, factors, start, stop))
            while pending:
                yield pending.popleft().result()

    def _write_disk(
        self, chunks: Iterator[_Chunk], logical_size: int, factors: tuple[tuple[int, int, float, float], ...],
        *, round_id: int, client_id: int,
    ) -> StoredHistoryVector:
        """Write each chunk as it is encoded, publish the entry atomically, then map it read-only."""
        session = self._disk_session()
        name = f"r{round_id}_c{client_id}_{uuid.uuid4().hex}"
        temporary, final = session.path / f".tmp_{name}", session.path / name
        temporary.mkdir()
        try:
            # Only this metadata outlives a chunk; the encoded arrays are released once saved.
            manifest_chunks: list[dict[str, Any]] = []
            for index, chunk in enumerate(chunks):
                chunk_dir = temporary / f"chunk_{index:05d}"
                chunk_dir.mkdir()
                arrays = {array_name: getattr(chunk, array_name) for array_name in _CHUNK_ARRAY_NAMES[chunk.kind]}
                for array_name, array in arrays.items():
                    np.save(chunk_dir / f"{array_name}.npy", array, allow_pickle=False)
                manifest_chunks.append({
                    "start": chunk.start, "stop": chunk.stop, "layout": chunk.kind,
                    "base_code_bits": chunk.base_code_bits,
                    "checkpoint_block_size": _RANK_BLOCK_SIZE if chunk.kind == "dictionary" else None,
                    "arrays": {array_name: {"shape": list(array.shape), "dtype": str(array.dtype)} for array_name, array in arrays.items()},
                })
            manifest = {
                "format_version": 1, "logical_size": logical_size, "chunk_elements": self.config.chunk_elements,
                "stored_dtype": self.config.dtype, "normalize": self.config.normalize,
                "factors": [list(factor) for factor in factors], "chunks": manifest_chunks,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
            temporary.rename(final)
            loaded = tuple(
                _Chunk(meta["start"], meta["stop"], meta["layout"], base_code_bits=meta["base_code_bits"], **{
                    array_name: np.load(final / f"chunk_{index:05d}" / f"{array_name}.npy", mmap_mode="r", allow_pickle=False)
                    for array_name in meta["arrays"]
                })
                for index, meta in enumerate(manifest_chunks)
            )
            return StoredHistoryVector(loaded, logical_size, self.config.dtype, factors, final, session)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(final, ignore_errors=True)
            raise

    def create(
        self, value: torch.Tensor, *, round_id: int, client_id: int,
        normalization_factors: torch.Tensor | None = None,
    ) -> StoredHistoryVector:
        """Transform and retain one decoded CPU float32 vector."""
        assert isinstance(value, torch.Tensor) and value.dtype is torch.float32 and value.device.type == "cpu" and value.ndim == 1, (
            "History value must be a one-dimensional CPU float32 tensor."
        )
        assert not self.config.normalize or self._slices is not None, (
            f"History normalization for protocol {self._protocol_name!r} needs the state-dict manager's layer slices.")
        factors = (
            _normalize_factors(value, self._slices, normalization_factors)
            if self.config.normalize else ()
        )
        chunks = self._encode(value, factors)
        if self.config.disk:
            return self._write_disk(chunks, value.numel(), factors, round_id=round_id, client_id=client_id)
        return StoredHistoryVector(tuple(chunks), value.numel(), self.config.dtype, factors)


__all__ = ["HistoryStorageConfig", "HistoryStore", "StoredHistoryVector"]
