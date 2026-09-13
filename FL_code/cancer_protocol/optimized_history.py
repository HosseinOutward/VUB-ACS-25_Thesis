from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import uuid
import weakref

import numpy as np
import torch
from torch.utils._pytree import tree_map

from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.utils import GradientPreprocessMetadata, StateDictManager

from .cancer_protocol import HistoryEntry, ReconstructionHistory
from .wz_quantizer import float_prefix

RANK_BLOCK_SIZE = 2**10
PACKING_CHUNK = 2**20


def deduplicate_gradient_values(
    normalized: torch.Tensor,
    match_bits: int,
    storage_dtype: torch.dtype = torch.bfloat16,
    round_prefix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize values and group them by a rounded or truncated float32 leading-bit prefix."""
    assert normalized.dtype == torch.float32 and 1 <= match_bits <= 32
    assert storage_dtype in (torch.float16, torch.bfloat16)
    quantized = normalized.to(storage_dtype).to(torch.float32)
    if round_prefix and match_bits < 32:
        quantized = (
            quantized.view(torch.int32) + (1 << (31 - match_bits))).view(torch.float32)
    grouped = float_prefix(quantized, match_bits).to(storage_dtype)
    unique_bits, inverse = torch.unique(
        grouped.view(torch.uint16).to(torch.int32), sorted=True, return_inverse=True)
    return unique_bits.to(torch.uint16).view(storage_dtype), inverse


def _pack_codes(codes: np.ndarray, code_bits: int) -> np.ndarray:
    shifts = np.arange(code_bits, dtype=np.uint32)
    parts = [
        np.packbits(
            ((codes[start:end, None].astype(np.uint32) >> shifts) & 1).astype(np.uint8).reshape(-1),
            bitorder="little",
        )
        for start in range(0, codes.size, PACKING_CHUNK)
        for end in (min(start + PACKING_CHUNK, codes.size),)
    ]
    return np.concatenate(parts)


def _unpack_codes(packed: np.ndarray, size: int, code_bits: int) -> np.ndarray:
    dtype = np.uint8 if code_bits <= 8 else np.uint16
    codes = np.empty(size, dtype=dtype)
    for start in range(0, size, PACKING_CHUNK):
        end = min(start + PACKING_CHUNK, size)
        coordinates = np.arange(start, end, dtype=np.int64)
        codes[start:end] = _codes_at(packed, coordinates, code_bits)
    return codes


def _codes_at(packed: np.ndarray, indices: np.ndarray, code_bits: int) -> np.ndarray:
    bit_offsets = indices * code_bits
    byte_indices, shifts = bit_offsets // 8, bit_offsets % 8
    words = np.asarray(packed[byte_indices], dtype=np.uint32)
    for byte_offset in (1, 2):
        needed = shifts + code_bits > byte_offset * 8
        words[needed] |= np.asarray(
            packed[byte_indices[needed] + byte_offset], dtype=np.uint32
        ) << (byte_offset * 8)
    dtype = np.uint8 if code_bits <= 8 else np.uint16
    return ((words >> shifts) & (2**code_bits - 1)).astype(dtype)


@dataclass(frozen=True, slots=True)
class TwoLevelGradientIndex:
    """Packed frequent-value codes plus rare codes ordered by escape occurrence."""

    base_code_bits: int
    values: np.ndarray
    packed_codes: np.ndarray
    exceptions: np.ndarray
    rank_checkpoints: np.ndarray

    @property
    def nbytes(self) -> int:
        """Return the retained array bytes for this two-level index."""
        return sum(array.nbytes for array in (
            self.values, self.packed_codes, self.exceptions, self.rank_checkpoints))


def build_two_level_gradient_index(
    unique_values: torch.Tensor, inverse: torch.Tensor, base_code_bits: int
) -> TwoLevelGradientIndex:
    """Put frequent values in packed codes and rare values behind the final escape code."""
    assert 1 <= base_code_bits <= 16
    escape_code = 2**base_code_bits - 1
    counts = torch.bincount(inverse, minlength=unique_values.numel())
    order = torch.argsort(counts, descending=True, stable=True)
    old_to_rank = torch.empty_like(order)
    old_to_rank[order] = torch.arange(order.numel())
    ranked = old_to_rank[inverse]
    base_dtype = torch.uint8 if base_code_bits <= 8 else torch.int32
    numpy_base_dtype = np.uint8 if base_code_bits <= 8 else np.uint16
    base_codes = torch.where(ranked < escape_code, ranked, escape_code).to(
        base_dtype).numpy().astype(numpy_base_dtype, copy=False)
    rare_count = max(0, unique_values.numel() - escape_code)
    exception_dtype = np.uint8 if rare_count <= 2**8 else np.uint16 if rare_count <= 2**16 else np.uint32
    exceptions = (ranked[ranked >= escape_code] - escape_code).numpy().astype(
        exception_dtype, copy=False)

    block_counts = np.add.reduceat(
        base_codes == escape_code, np.arange(0, base_codes.size, RANK_BLOCK_SIZE))
    rank_checkpoints = np.zeros(block_counts.size, dtype=np.uint32)
    if block_counts.size > 1:
        rank_checkpoints[1:] = np.cumsum(block_counts[:-1], dtype=np.uint32)
    return TwoLevelGradientIndex(
        base_code_bits,
        unique_values[order].view(torch.uint16).numpy().copy(),
        _pack_codes(base_codes, base_code_bits),
        exceptions,
        rank_checkpoints,
    )


def _metadata_nbytes(metadata: GradientPreprocessMetadata) -> int:
    return (
        metadata.norm_factors.numel() * metadata.norm_factors.element_size()
        + metadata.outliers.positions.nbytes
        + (0 if metadata.outliers.scale is None else metadata.outliers.scale.nbytes)
        + metadata.outliers.packed_signs.nbytes
    )


def _unlink_files(paths: tuple[Path, ...]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


@dataclass(slots=True, weakref_slot=True)
class _DiskArrays:
    """Generated two-level index files owned by one packed history entry."""

    values: Path
    packed_codes: Path
    exceptions: Path
    rank_checkpoints: Path
    _cleanup: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cleanup = weakref.finalize(
            self,
            _unlink_files,
            (self.values, self.packed_codes, self.exceptions, self.rank_checkpoints),
        )


@dataclass(frozen=True, slots=True)
class _PackedGradient:
    """Deduplicated 16-bit floating-point storage reconstructed by a lazy history tensor."""

    manager: StateDictManager
    metadata: GradientPreprocessMetadata
    outlier_threshold: float | None
    storage_dtype: torch.dtype
    base_code_bits: int
    index: TwoLevelGradientIndex | _DiskArrays
    vector_size: int

    def _arrays(self) -> TwoLevelGradientIndex:
        if isinstance(self.index, _DiskArrays):
            return TwoLevelGradientIndex(
                self.base_code_bits,
                np.load(self.index.values, mmap_mode="r", allow_pickle=False),
                np.load(self.index.packed_codes, mmap_mode="r", allow_pickle=False),
                np.load(self.index.exceptions, mmap_mode="r", allow_pickle=False),
                np.load(self.index.rank_checkpoints, mmap_mode="r", allow_pickle=False),
            )
        return self.index

    def _value_indices(self, coordinates: np.ndarray | None = None) -> np.ndarray:
        arrays = self._arrays()
        escape_code = 2**arrays.base_code_bits - 1
        if coordinates is None:
            codes = _unpack_codes(
                arrays.packed_codes, self.vector_size, arrays.base_code_bits).astype(np.int64)
            escaped = codes == escape_code
            assert int(escaped.sum()) == arrays.exceptions.size
            codes[escaped] += arrays.exceptions.astype(np.int64)
            return codes

        flat_coordinates = coordinates.reshape(-1)
        codes = _codes_at(
            arrays.packed_codes, flat_coordinates, arrays.base_code_bits).astype(np.int64)
        escaped = codes == escape_code
        if not escaped.any():
            return codes
        escaped_coordinates = flat_coordinates[escaped]
        blocks = escaped_coordinates // RANK_BLOCK_SIZE
        exception_indices = np.empty(escaped_coordinates.size, dtype=np.int64)
        for block in np.unique(blocks):
            selected = blocks == block
            start = int(block) * RANK_BLOCK_SIZE
            end = min(start + RANK_BLOCK_SIZE, self.vector_size)
            block_coordinates = np.arange(start, end, dtype=np.int64)
            block_codes = _codes_at(
                arrays.packed_codes, block_coordinates, arrays.base_code_bits)
            ranks_before = np.cumsum(block_codes == escape_code) - (block_codes == escape_code)
            exception_indices[selected] = (
                arrays.rank_checkpoints[block]
                + ranks_before[escaped_coordinates[selected] - start]
            )
        codes[escaped] += arrays.exceptions[exception_indices].astype(np.int64)
        return codes

    def reconstruct(self, index: Any = slice(None)) -> torch.Tensor:
        """Restore only coordinates selected by one-dimensional tensor indexing."""
        arrays = self._arrays()
        if isinstance(index, slice) and index == slice(None):
            selected = np.asarray(
                arrays.values[self._value_indices()], dtype=np.uint16).copy()
            return self.manager.denormalize_delta(
                torch.from_numpy(selected).view(self.storage_dtype).to(torch.float32),
                self.metadata,
                self.outlier_threshold,
            )

        coordinates = np.arange(self.vector_size, dtype=np.int64)[index]
        coordinate_array = np.asarray(coordinates, dtype=np.int64)
        selected = np.asarray(
            arrays.values[self._value_indices(coordinate_array)], dtype=np.uint16).copy()
        normalized = torch.from_numpy(selected).view(self.storage_dtype).to(torch.float32)
        restored = self.manager.denormalize_delta(
            normalized,
            self.metadata,
            self.outlier_threshold,
            torch.as_tensor(coordinate_array, dtype=torch.long),
        )
        return restored.reshape(coordinate_array.shape)

    @property
    def resident_nbytes(self) -> int:
        """Return array and metadata bytes retained in RAM, excluding Python object overhead."""
        metadata_bytes = _metadata_nbytes(self.metadata)
        if isinstance(self.index, _DiskArrays):
            return metadata_bytes
        return metadata_bytes + self.index.nbytes

    @property
    def disk_nbytes(self) -> int:
        """Return generated on-disk array bytes for this entry."""
        if not isinstance(self.index, _DiskArrays):
            return 0
        return sum(path.stat().st_size for path in (
            self.index.values,
            self.index.packed_codes,
            self.index.exceptions,
            self.index.rank_checkpoints,
        ))

    @property
    def unique_count(self) -> int:
        """Return the number of float16 values stored in the deduplication dictionary."""
        return int(self._arrays().values.size)


class LazyGradientTensor(torch.Tensor):
    """Read-only float32 tensor facade that reconstructs packed history coordinates on demand."""

    __slots__ = ("_packed",)

    @staticmethod
    def __new__(cls, packed: _PackedGradient) -> LazyGradientTensor:
        return torch.Tensor._make_wrapper_subclass(
            cls,
            (packed.vector_size,),
            strides=(1,),
            storage_offset=0,
            dtype=torch.float32,
            layout=torch.strided,
            device=torch.device("cpu"),
            requires_grad=False,
        )

    def __init__(self, packed: _PackedGradient) -> None:
        self._packed = packed

    def __getitem__(self, index: Any) -> torch.Tensor:
        return self._packed.reconstruct(index)

    def __len__(self) -> int:
        return self._packed.vector_size

    def numel(self) -> int:
        """Return the logical vector size without materializing its values."""
        return self._packed.vector_size

    def size(self, dim: int | None = None) -> torch.Size | int:
        """Return the logical one-dimensional shape without materializing values."""
        shape = torch.Size((self._packed.vector_size,))
        return shape if dim is None else shape[dim]

    def materialize(self) -> torch.Tensor:
        """Reconstruct the complete float32 vector explicitly."""
        return self._packed.reconstruct()

    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Materialize the vector as a NumPy array."""
        return self.materialize().numpy(force=force)

    def tolist(self) -> list[float]:
        """Materialize the vector as a Python list."""
        return self.materialize().tolist()

    def __repr__(self) -> str:
        return (
            f"LazyGradientTensor(size=({self.numel()},), unique_values="
            f"{self._packed.unique_count}, device='cpu')"
        )

    @classmethod
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Materialize lazy operands when an operation cannot be answered from metadata alone."""
        if func._schema.is_mutable:
            raise TypeError("LazyGradientTensor is a read-only history view.")
        unwrap = lambda value: value.materialize() if isinstance(value, cls) else value
        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {}))


class OptimizedGradientHistory(ReconstructionHistory):
    """Bounded history stored as layer-normalized, prefix-grouped 16-bit values.

    The protocol commits ordinary CPU float32 deltas. Views expose read-only lazy float32 tensor
    facades, while the retained representation lives either in RAM or in generated `.npy` files.
    Frequent values use packed `base_code_bits` codes, the final code escapes to an ordered
    rare-code subset, and block rank checkpoints support lazy access without storing exception
    positions. `dedup_match_bits` controls the memory-error tradeoff; server and temporal ledgers
    share each immutable entry.
    """

    def __init__(
        self,
        max_per_client: int | None,
        state_dict_manager: StateDictManager,
        outlier_threshold: float | None = 8.0,
        storage_dir: Path | None = None,
        dedup_match_bits: int = 14,
        storage_dtype: torch.dtype = torch.bfloat16,
        base_code_bits: int = 9,
        round_prefix: bool = True,
    ) -> None:
        super().__init__(max_per_client, dtype=torch.float32)
        assert outlier_threshold is None or outlier_threshold > 0
        assert 1 <= dedup_match_bits <= 32
        assert storage_dtype in (torch.float16, torch.bfloat16)
        assert 1 <= base_code_bits <= 16
        assert isinstance(round_prefix, bool)
        self.state_dict_manager: StateDictManager = state_dict_manager
        self.outlier_threshold: float | None = outlier_threshold
        self.storage_dir: Path | None = storage_dir
        self.dedup_match_bits: int = dedup_match_bits
        self.storage_dtype: torch.dtype = storage_dtype
        self.base_code_bits: int = base_code_bits
        self.round_prefix: bool = round_prefix
        if storage_dir is not None:
            storage_dir.mkdir(parents=True, exist_ok=True)

    def commit(self, reconst: torch.Tensor, record: CompressionRecord, access: Access) -> None:
        """Normalize, prefix-group, two-level index, and commit one reconstructed delta."""
        assert not self.frozen, "OptimizedGradientHistory is frozen and cannot be modified."
        assert reconst.dtype == self.dtype and reconst.device == torch.device("cpu")
        assert reconst.ndim == 1 and reconst.numel() == self.state_dict_manager.param_count
        if access is Access.NONE:
            return

        normalized, metadata = self.state_dict_manager.normalize_delta(
            reconst, self.outlier_threshold)
        unique_values, inverse = deduplicate_gradient_values(
            normalized, self.dedup_match_bits, self.storage_dtype, self.round_prefix)
        two_level_index = build_two_level_gradient_index(
            unique_values, inverse, self.base_code_bits)

        if self.storage_dir is None:
            packed = _PackedGradient(
                self.state_dict_manager,
                metadata,
                self.outlier_threshold,
                self.storage_dtype,
                self.base_code_bits,
                two_level_index,
                reconst.numel(),
            )
        else:
            stem = f"r{record.round_id}_c{record.client_id}_{uuid.uuid4().hex}"
            paths = _DiskArrays(
                self.storage_dir / f"{stem}_values.npy",
                self.storage_dir / f"{stem}_codes.npy",
                self.storage_dir / f"{stem}_exceptions.npy",
                self.storage_dir / f"{stem}_ranks.npy",
            )
            for path, array in zip(
                (paths.values, paths.packed_codes, paths.exceptions, paths.rank_checkpoints),
                (
                    two_level_index.values,
                    two_level_index.packed_codes,
                    two_level_index.exceptions,
                    two_level_index.rank_checkpoints,
                ),
                strict=True,
            ):
                np.save(path, array, allow_pickle=False)
            packed = _PackedGradient(
                self.state_dict_manager,
                metadata,
                self.outlier_threshold,
                self.storage_dtype,
                self.base_code_bits,
                paths,
                reconst.numel(),
            )

        entry = HistoryEntry(
            tensor=LazyGradientTensor(packed),
            round_id=record.round_id,
            client_id=record.client_id,
            access=access,
            round_name=record.round_name_full,
        )
        self._add_entry(self._server, entry)
        if access is Access.TEMPORAL_TOO:
            self._add_entry(self._temporal, entry)

    def config_snapshot(self) -> dict[str, Any]:
        """Describe the lossy packed representation used for experiment provenance."""
        return super().config_snapshot() | {
            "outlier_threshold": self.outlier_threshold,
            "storage_dir": None if self.storage_dir is None else str(self.storage_dir),
            "dedup_match_bits": self.dedup_match_bits,
            "storage_dtype": str(self.storage_dtype),
            "base_code_bits": self.base_code_bits,
            "round_prefix": self.round_prefix,
        }

    def _packed_entries(self) -> tuple[_PackedGradient, ...]:
        entries = {
            id(entry.tensor): entry.tensor._packed
            for ledger in self._server.values()
            for entry in ledger
            if isinstance(entry.tensor, LazyGradientTensor)
        }
        return tuple(entries.values())

    @property
    def resident_nbytes(self) -> int:
        """Return retained array and metadata bytes in RAM without double-counting ledgers."""
        return sum(entry.resident_nbytes for entry in self._packed_entries())

    @property
    def disk_nbytes(self) -> int:
        """Return generated array bytes on disk without double-counting ledgers."""
        return sum(entry.disk_nbytes for entry in self._packed_entries())


__all__ = [
    "LazyGradientTensor",
    "OptimizedGradientHistory",
    "TwoLevelGradientIndex",
    "build_two_level_gradient_index",
    "deduplicate_gradient_values",
]
