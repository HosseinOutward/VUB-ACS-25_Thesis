from __future__ import annotations

import gc
from pathlib import Path
import weakref

import numpy as np
import torch
from torch import nn

from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.utils import StateDictManager
from FL_code.cancer_protocol.optimized_history import (
    LazyGradientTensor,
    OptimizedGradientHistory,
    _codes_at,
    _pack_codes,
    _unpack_codes,
    deduplicate_gradient_values,
)


def _manager() -> StateDictManager:
    return StateDictManager(nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2)))


def _delta(manager: StateDictManager) -> torch.Tensor:
    values = torch.linspace(-4, 4, manager.param_count, dtype=torch.float32)
    values[::7] *= 5
    return values


def _expected(
    manager: StateDictManager,
    delta: torch.Tensor,
    threshold: float,
    match_bits: int = 14,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    normalized, metadata = manager.normalize_delta(delta, threshold)
    values, inverse = deduplicate_gradient_values(normalized, match_bits, storage_dtype)
    return manager.denormalize_delta(values[inverse], metadata, threshold)


def test_lazy_history_reconstructs_only_requested_coordinates() -> None:
    manager = _manager()
    delta = _delta(manager)
    expected = _expected(manager, delta, 1.5)
    history = OptimizedGradientHistory(2, manager, outlier_threshold=1.5)
    history.commit(delta, CompressionRecord(3, 7, "T"), Access.TEMPORAL_TOO)

    entry = history.view(Access.TEMPORAL_TOO, 7)[7][0]
    assert isinstance(entry.tensor, LazyGradientTensor)
    assert entry.tensor.dtype == torch.float32 and entry.tensor.numel() == manager.param_count
    assert entry.tensor.size() == (manager.param_count,) and entry.tensor.size(0) == manager.param_count
    torch.testing.assert_close(entry.tensor[3:13:2], expected[3:13:2])
    torch.testing.assert_close(entry.tensor[torch.tensor([0, 9, 9, -1])], expected[[0, 9, 9, -1]])
    torch.testing.assert_close(entry.tensor.materialize(), expected)
    torch.testing.assert_close(torch.stack((entry.tensor, entry.tensor))[0], expected)


def test_state_manager_normalization_and_outliers_are_reversible_before_float16() -> None:
    manager = _manager()
    delta = _delta(manager)
    normalized, metadata = manager.normalize_delta(delta, outlier_threshold=1.5)

    assert metadata.outliers.positions.size > 0
    torch.testing.assert_close(
        manager.denormalize_delta(normalized, metadata, outlier_threshold=1.5), delta)


def test_harsher_prefix_dedup_uses_fewer_representatives_with_more_error() -> None:
    manager = _manager()
    delta = _delta(manager)
    normalized, metadata = manager.normalize_delta(delta, outlier_threshold=1.5)
    exact_values, exact_inverse = deduplicate_gradient_values(normalized, match_bits=19)
    harsh_values, harsh_inverse = deduplicate_gradient_values(normalized, match_bits=11)
    exact = manager.denormalize_delta(exact_values[exact_inverse], metadata, 1.5)
    harsh = manager.denormalize_delta(harsh_values[harsh_inverse], metadata, 1.5)

    assert harsh_values.numel() < exact_values.numel()
    assert (harsh - delta).abs().mean() > (exact - delta).abs().mean()


def test_packed_codes_round_trip_non_byte_aligned_widths() -> None:
    rng = np.random.default_rng(4)
    for code_bits in (6, 9, 14):
        codes = rng.integers(0, 2**code_bits, 1_003, dtype=np.uint16)
        packed = _pack_codes(codes, code_bits)
        positions = np.array([0, 1, 2, 137, 137, 1_002])

        np.testing.assert_array_equal(_unpack_codes(packed, codes.size, code_bits), codes)
        np.testing.assert_array_equal(_codes_at(packed, positions, code_bits), codes[positions])


def test_history_does_not_retain_the_committed_raw_tensor() -> None:
    manager = _manager()
    delta = _delta(manager)
    raw_reference = weakref.ref(delta)
    history = OptimizedGradientHistory(2, manager)
    history.commit(delta, CompressionRecord(0, 0, "R"), Access.SERVER_ONLY)

    del delta
    gc.collect()
    assert raw_reference() is None
    assert history.resident_nbytes < manager.param_count * torch.tensor([], dtype=torch.float32).element_size()


def test_disk_backing_keeps_arrays_out_of_resident_payload(tmp_path: Path) -> None:
    manager = _manager()
    delta = _delta(manager)
    expected = _expected(manager, delta, 1.5)
    history = OptimizedGradientHistory(2, manager, outlier_threshold=1.5, storage_dir=tmp_path)
    history.commit(delta, CompressionRecord(1, 2, "O"), Access.SERVER_ONLY)

    files = sorted(tmp_path.glob("*.npy"))
    assert len(files) == 4
    assert history.disk_nbytes == sum(path.stat().st_size for path in files)
    assert history.resident_nbytes < history.disk_nbytes
    tensor = history.view(Access.SERVER_ONLY, 999)[2][0].tensor
    torch.testing.assert_close(tensor[np.array([1, 5, 11])], expected[[1, 5, 11]])


def test_two_level_index_restores_frequent_and_escape_codes() -> None:
    manager = StateDictManager(nn.Linear(512, 1, bias=False))
    delta = torch.randn(manager.param_count, generator=torch.Generator().manual_seed(7))
    expected = _expected(manager, delta, 8.0, match_bits=14)
    history = OptimizedGradientHistory(1, manager, dedup_match_bits=14, base_code_bits=6)
    history.commit(delta, CompressionRecord(0, 0, "T"), Access.SERVER_ONLY)
    tensor = history.view(Access.SERVER_ONLY, 0)[0][0].tensor

    assert tensor._packed._arrays().exceptions.size > 0
    indices = torch.tensor([0, 17, 255, 17, 511])
    torch.testing.assert_close(tensor[indices], expected[indices])
    torch.testing.assert_close(tensor.materialize(), expected)


def test_history_eviction_and_copy_keep_shared_lazy_entries() -> None:
    manager = _manager()
    history = OptimizedGradientHistory(1, manager, outlier_threshold=None)
    history.commit(_delta(manager), CompressionRecord(0, 0, "T"), Access.TEMPORAL_TOO)
    copied = history.copy()
    first = copied.view(Access.SERVER_ONLY, 0)[0][0]

    history.commit(_delta(manager) + 1, CompressionRecord(1, 0, "T"), Access.TEMPORAL_TOO)
    assert history.view(Access.SERVER_ONLY, 0)[0][0].round_id == 1
    assert copied.view(Access.SERVER_ONLY, 0)[0][0] is first
    assert isinstance(first.tensor, LazyGradientTensor)
