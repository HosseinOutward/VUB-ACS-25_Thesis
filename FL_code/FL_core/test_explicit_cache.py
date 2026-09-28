from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import re
import time

import pytest
import torch

from .explicit_cache import CacheEntry, atomic_output, content_hash


def test_content_hash_is_order_independent_for_mapping_and_tensor_values() -> None:
    first = {"tensor": torch.arange(4), "settings": {"lr": 0.1, "epochs": 2}}
    second = {"settings": {"epochs": 2, "lr": 0.1}, "tensor": torch.arange(4)}

    assert content_hash(first) == content_hash(second)
    assert content_hash(first) != content_hash({**first, "tensor": torch.arange(5)})


def test_cache_entry_exposes_identity_and_computes_once(tmp_path: Path) -> None:
    entry = CacheEntry(tmp_path, {"model": "small", "seed": 3})
    calls = 0

    def compute() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": 7}

    assert entry.path == tmp_path / f"{entry.identity}.pkl"
    assert entry.resolve(compute) == {"value": 7}
    assert entry.resolve(compute) == {"value": 7}
    assert calls == 1


def test_concurrent_resolves_compute_once(tmp_path: Path) -> None:
    calls = 0

    def compute() -> int:
        nonlocal calls
        calls += 1
        time.sleep(0.2)
        return 5

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: CacheEntry(tmp_path, "shared").resolve(compute), range(2)))

    assert results == [5, 5]
    assert calls == 1

def test_cache_entry_refresh_recomputes(tmp_path: Path) -> None:
    entry = CacheEntry(tmp_path, "identity")
    assert entry.resolve(lambda: 1) == 1
    assert entry.resolve(lambda: 2, refresh=True) == 2
    assert entry.resolve(lambda: 3) == 2


def test_atomic_output_removes_partial_file_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "result.bin"

    with pytest.raises(RuntimeError):
        with atomic_output(destination) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("failed computation")

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_corrupt_entry_fails_with_its_path(tmp_path: Path) -> None:
    entry = CacheEntry(tmp_path, "corrupt")
    entry.path.parent.mkdir(parents=True, exist_ok=True)
    entry.path.write_bytes(b"not a pickle")

    with pytest.raises(pickle.UnpicklingError, match=re.escape(str(entry.path))):
        entry.resolve(lambda: "unused")
