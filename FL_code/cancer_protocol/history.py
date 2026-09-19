"""Decoder-visible reconstruction history and its in-memory representations."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from FL_code.FL_core.codec import Access, CompressionRecord


_PREFIX_BITS = 18
_CHUNK_ELEMENTS = 1 << 20


def float_prefix(values: torch.Tensor, match_bits: int = _PREFIX_BITS) -> torch.Tensor:
    """Truncate float32 values to a shared leading-bit prefix."""
    assert values.dtype is torch.float32
    assert 1 <= match_bits <= 32
    if match_bits == 32:
        return values
    return (values.view(torch.int32) & -(1 << (32 - match_bits))).view(torch.float32)


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
    """One compressed coordinate range, exposed for on-device lazy decoding."""

    start: int
    stop: int
    dictionary: torch.Tensor
    codes: torch.Tensor

    @property
    def nbytes(self) -> int:
        """Return bytes retained by this dictionary-coded chunk."""
        return self.dictionary.numel() * self.dictionary.element_size() + self.codes.numel() * self.codes.element_size()


def _compress_chunk(value: torch.Tensor, start: int, stop: int) -> _CompressedHistoryChunk:
    """Prefix, deduplicate, and index one physical history chunk."""
    prefixed = float_prefix(value[start:stop])
    bits = prefixed.view(torch.int32).numpy()
    unique, inverse, counts = np.unique(bits, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")

    ranks = np.empty(unique.size, dtype=np.int64)
    ranks[order] = np.arange(unique.size, dtype=np.int64)
    dictionary_bits = unique[order].astype(np.int32, copy=True)
    code_values = ranks[inverse]
    code_dtype = np.uint8 if unique.size <= 1 << 8 else np.uint16 if unique.size <= 1 << 16 else np.uint32
    dictionary = torch.from_numpy(dictionary_bits).view(torch.float32)
    codes = torch.from_numpy(code_values.astype(code_dtype, copy=True))
    return _CompressedHistoryChunk(start, stop, dictionary, codes)


class CompressedHistoryVector(HistoryVector):
    """Prefix-truncated, dictionary-coded CPU history with the HistoryVector read contract."""

    def __init__(self, value: torch.Tensor) -> None:
        assert (
            isinstance(value, torch.Tensor)
            and value.dtype is torch.float32
            and value.device.type == "cpu"
            and value.ndim == 1
        ), "History value must be a one-dimensional CPU float32 tensor."
        source = value.detach().contiguous()
        self._logical_size: int = source.numel()
        self._chunks: tuple[_CompressedHistoryChunk, ...] = tuple(
            _compress_chunk(source, start, min(start + _CHUNK_ELEMENTS, source.numel()))
            for start in range(0, source.numel(), _CHUNK_ELEMENTS)
        )

    def __getstate__(self) -> dict[str, Any]:
        """Serialize unsigned codes through same-width signed views supported by torch.save."""
        chunks = tuple((
            chunk.start,
            chunk.stop,
            chunk.dictionary,
            chunk.codes.view(torch.int16) if chunk.codes.dtype is torch.uint16
            else chunk.codes.view(torch.int32) if chunk.codes.dtype is torch.uint32
            else chunk.codes,
            str(chunk.codes.dtype),
        ) for chunk in self._chunks)
        return {"logical_size": self._logical_size, "chunks": chunks}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the narrow unsigned code dtype after torch deserialization."""
        dtypes = {"torch.uint8": torch.uint8, "torch.uint16": torch.uint16, "torch.uint32": torch.uint32}
        self._logical_size = state["logical_size"]
        self._chunks = tuple(
            _CompressedHistoryChunk(start, stop, dictionary, codes.view(dtypes[dtype]))
            for start, stop, dictionary, codes, dtype in state["chunks"]
        )

    @property
    def logical_size(self) -> int:
        """Return the number of values represented by this vector."""
        return self._logical_size

    def _iter_encoded_chunks(self) -> Iterator[_CompressedHistoryChunk]:
        """Yield dictionary and code tensors to the colocated storage consumer."""
        yield from self._chunks

    @property
    def resident_nbytes(self) -> int:
        """Return bytes occupied by dictionaries and narrow integer code arrays."""
        return sum(chunk.nbytes for chunk in self._chunks)

    def iter_chunks(self) -> Iterator[tuple[slice, torch.Tensor]]:
        """Decode and yield compressed chunks in their original coordinate order."""
        for chunk in self._chunks:
            yield slice(chunk.start, chunk.stop), chunk.dictionary[chunk.codes.to(torch.int64)]

    def materialize(self) -> torch.Tensor:
        """Explicitly decode the complete vector, including prefix quantization."""
        chunks = [values for _, values in self.iter_chunks()]
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)


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
        return {
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "max_per_client": self.max_per_client,
            "prefix_bits": _PREFIX_BITS if isinstance(self, CompressedReconstructionHistory) else 32,
            "chunk_elements": _CHUNK_ELEMENTS,
        }

    def copy(self) -> ReconstructionHistory:
        """Copy ledger lists while sharing immutable entries and their payloads."""
        copied = copy.copy(self)
        copied._server = {client_id: entries.copy() for client_id, entries in self._server.items()}
        copied._temporal = {client_id: entries.copy() for client_id, entries in self._temporal.items()}
        return copied

    def _retain(self, value: torch.Tensor) -> HistoryVector:
        """Retain one committed reconstruction in the reference representation."""
        return HistoryVector(value)

    def commit(self, reconst: torch.Tensor, record: CompressionRecord, access: Access) -> None:
        """Retain a decoded reconstruction in each ledger permitted by its access policy."""
        assert not self.frozen, "ReconstructionHistory is frozen and cannot be modified."
        assert reconst.dtype is torch.float32 and reconst.device == torch.device("cpu") and reconst.ndim == 1
        if access is Access.NONE:
            return
        entry = HistoryEntry(
            value=self._retain(reconst),
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
            value=self._retain(reconst),
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

    def _retain(self, value: torch.Tensor) -> HistoryVector:
        """Retain a decoded reconstruction as a CompressedHistoryVector."""
        return CompressedHistoryVector(value)


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
