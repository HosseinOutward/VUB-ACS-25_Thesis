from __future__ import annotations

import io

import pytest
import torch

from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.cancer_protocol.history import (
    _CHUNK_ELEMENTS,
    _PREFIX_BITS,
    CompressedHistoryVector,
    CompressedReconstructionHistory,
    HistoryVector,
    ReconstructionHistory,
    float_prefix,
)


def _record(round_id: int, client_id: int, round_name: str = "T") -> CompressionRecord:
    return CompressionRecord(round_id, client_id, round_name)


@pytest.mark.parametrize("history_type", [ReconstructionHistory, CompressedReconstructionHistory])
def test_commit_stores_a_copy_not_an_alias_of_the_caller_tensor(
    history_type: type[ReconstructionHistory],
) -> None:
    history = history_type(2)
    original = torch.zeros(4)
    history.commit(original, _record(0, 0), Access.TEMPORAL_TOO)

    original[:] = 99.0
    stored = history.view(Access.TEMPORAL_TOO, 0)[0][0].value.materialize()
    expected = float_prefix(torch.zeros(4)) if history_type is CompressedReconstructionHistory else torch.zeros(4)
    assert torch.equal(stored, expected)
    assert stored.data_ptr() != original.data_ptr()


def test_history_eviction_keeps_only_the_configured_depth() -> None:
    history = ReconstructionHistory(2)
    for round_id in range(4):
        history.commit(torch.full((1,), float(round_id)), _record(round_id, 0), Access.SERVER_ONLY)

    ledger = history.view(Access.SERVER_ONLY, 0)[0]
    assert len(ledger) == 2
    assert [entry.round_id for entry in ledger] == [2, 3]


def test_replace_latest_mutates_the_ledger_entry_in_place_for_temporal_too() -> None:
    history = ReconstructionHistory(2)
    original = torch.zeros(4)
    history.commit(original, _record(4, 2), Access.TEMPORAL_TOO)

    replacement = torch.ones(4)
    history.replace_latest(replacement, round_id=4, client_id=2)
    replacement[:] = -1.0  # mutating after the call must not affect the stored value

    server_entry = history.view(Access.SERVER_ONLY, 99)[2][-1]
    temporal_entry = history.view(Access.TEMPORAL_TOO, 2)[2][-1]
    assert server_entry is temporal_entry
    assert len(history.view(Access.SERVER_ONLY, 99)[2]) == 1, "replacement must not append a new entry"
    assert torch.equal(server_entry.value.materialize(), torch.ones(4))
    assert server_entry.round_id == 4 and server_entry.access is Access.TEMPORAL_TOO
    assert server_entry.round_name == "T"


def test_replace_latest_does_not_grant_temporal_access_to_a_server_only_entry() -> None:
    history = ReconstructionHistory(2)
    history.commit(torch.zeros(4), _record(1, 5), Access.SERVER_ONLY)

    # The replacement value happens to be a raw delta, but SERVER_ONLY access must be preserved:
    # a worker must not gain a temporal view of a reconstruction it was never entitled to decode.
    history.replace_latest(torch.ones(4), round_id=1, client_id=5)

    server_entry = history.view(Access.SERVER_ONLY, 99)[5][-1]
    assert server_entry.access is Access.SERVER_ONLY
    assert torch.equal(server_entry.value.materialize(), torch.ones(4))
    with pytest.raises(AssertionError, match="No temporal reconstruction history"):
        history.view(Access.TEMPORAL_TOO, 5)


def test_replace_latest_requires_the_latest_round_to_match() -> None:
    history = ReconstructionHistory(2)
    history.commit(torch.zeros(4), _record(0, 0), Access.SERVER_ONLY)

    with pytest.raises(AssertionError, match="No latest reconstruction"):
        history.replace_latest(torch.ones(4), round_id=1, client_id=0)


def test_copy_is_independent_but_shares_existing_entries() -> None:
    history = ReconstructionHistory(1)
    history.commit(torch.zeros(4), _record(0, 0), Access.TEMPORAL_TOO)
    copied = history.copy()
    first_entry = copied.view(Access.SERVER_ONLY, 0)[0][0]

    history.commit(torch.ones(4), _record(1, 0), Access.TEMPORAL_TOO)

    assert history.view(Access.SERVER_ONLY, 0)[0][0].round_id == 1
    assert copied.view(Access.SERVER_ONLY, 0)[0][0] is first_entry


def test_history_vector_owns_float32_cpu_values_and_fixed_chunks() -> None:
    value = torch.arange(_CHUNK_ELEMENTS + 3, dtype=torch.float32)
    stored = HistoryVector(value)
    value[0] = -1

    chunks = list(stored.iter_chunks())
    assert [(part.start, part.stop) for part, _ in chunks] == [
        (0, _CHUNK_ELEMENTS), (_CHUNK_ELEMENTS, _CHUNK_ELEMENTS + 3)
    ]
    assert stored.numel() == value.numel()
    assert stored.materialize()[0] == 0
    assert all(part.device.type == "cpu" and part.dtype is torch.float32 for _, part in chunks)


def test_compressed_history_prefixes_values_and_exposes_encoded_chunks() -> None:
    value = torch.tensor([0.0, -0.0, 1.0, 1.0, 2.0], dtype=torch.float32).repeat(100)
    stored = CompressedHistoryVector(value)
    expected = float_prefix(value, _PREFIX_BITS)

    torch.testing.assert_close(stored.materialize(), expected)
    chunk = next(stored._iter_encoded_chunks())
    assert chunk.layout == "direct"
    assert chunk.dictionary is not None and chunk.dictionary.dtype is torch.float32
    assert chunk.codes is not None
    assert chunk.codes.dtype is torch.uint8
    assert chunk.codes.numel() == value.numel()
    assert torch.equal(stored.materialize().view(torch.int32), expected.view(torch.int32))
    assert chunk.dictionary.numel() <= 4


def test_compressed_history_accepts_noncontiguous_input() -> None:
    value = torch.arange(20, dtype=torch.float32)[::2]
    assert not value.is_contiguous()

    assert torch.equal(
        CompressedHistoryVector(value).materialize(), float_prefix(value.contiguous())
    )


@pytest.mark.parametrize(("distinct", "repeats", "layout", "dtype", "code_bits"), [
    (256, 4, "direct", torch.uint8, 8),
    (257, 16, "packed", torch.uint8, 9),
    (65_536, 4, "direct", torch.uint16, 16),
    (65_537, 8, "packed", torch.uint8, 17),
])
def test_compressed_history_selects_the_smallest_worthwhile_layout(
    distinct: int, repeats: int, layout: str, dtype: torch.dtype, code_bits: int,
) -> None:
    bit_patterns = (torch.arange(distinct, dtype=torch.int64) * (1 << (32 - _PREFIX_BITS)) + 0x10000000).to(torch.int32)
    value = bit_patterns.repeat(repeats).view(torch.float32)
    chunk = next(CompressedHistoryVector(value)._iter_encoded_chunks())
    assert chunk.layout == layout
    assert chunk.codes is not None
    assert chunk.codes.dtype is dtype
    assert chunk.code_bits == code_bits
    assert (chunk.packed_high is not None) is (layout == "packed")


def test_compressed_history_uses_raw_prefixes_when_a_dictionary_would_expand_data() -> None:
    patterns = (
        torch.arange(257, dtype=torch.int64) * (1 << (32 - _PREFIX_BITS)) + 0x10000000
    ).to(torch.int32)
    stored = CompressedHistoryVector(patterns.view(torch.float32))
    chunk = next(stored._iter_encoded_chunks())

    assert chunk.layout == "raw"
    assert chunk.dictionary is None and chunk.codes is None and chunk.code_bits is None
    assert chunk.nbytes == patterns.numel() * 4
    assert torch.equal(stored.materialize(), float_prefix(patterns.view(torch.float32)))


@pytest.mark.parametrize(("distinct", "repeats"), [(257, 16), (65_536, 4), (65_537, 8)])
def test_compressed_history_torch_serialization_preserves_layout(
    distinct: int, repeats: int,
) -> None:
    patterns = (
        torch.arange(distinct, dtype=torch.int64) * (1 << (32 - _PREFIX_BITS)) + 0x10000000
    ).to(torch.int32)
    original = CompressedHistoryVector(patterns.repeat(repeats).view(torch.float32))
    buffer = io.BytesIO()
    torch.save(original, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)

    original_chunk = next(original._iter_encoded_chunks())
    restored_chunk = next(restored._iter_encoded_chunks())
    assert (restored_chunk.layout, restored_chunk.code_bits) == (
        original_chunk.layout, original_chunk.code_bits)
    assert restored_chunk.codes is not None and original_chunk.codes is not None
    assert restored_chunk.codes.dtype is original_chunk.codes.dtype
    assert torch.equal(
        original.materialize(), float_prefix(patterns.repeat(repeats).view(torch.float32)))
    assert torch.equal(restored.materialize(), original.materialize())


def test_compressed_history_ledger_uses_shared_entries_and_reports_constants() -> None:
    history = CompressedReconstructionHistory(2)
    history.commit(torch.ones(3), _record(0, 2), Access.TEMPORAL_TOO)
    server = history.view(Access.SERVER_ONLY, 9)[2][0]
    temporal = history.view(Access.TEMPORAL_TOO, 2)[2][0]
    assert server is temporal
    assert history.config_snapshot() == {
        "implementation": "FL_code.cancer_protocol.history.CompressedReconstructionHistory",
        "max_per_client": 2,
        "prefix_bits": _PREFIX_BITS,
        "chunk_elements": _CHUNK_ELEMENTS,
        "storage_format": "adaptive_split_packed_v1",
    }
