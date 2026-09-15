from __future__ import annotations

import pytest
import torch

from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.cancer_protocol.cancer_protocol import ReconstructionHistory


def _record(round_id: int, client_id: int, round_name: str = "T") -> CompressionRecord:
    return CompressionRecord(round_id, client_id, round_name)


def test_commit_stores_a_copy_not_an_alias_of_the_caller_tensor() -> None:
    history = ReconstructionHistory(2)
    original = torch.zeros(4)
    history.commit(original, _record(0, 0), Access.TEMPORAL_TOO)

    original[:] = 99.0
    stored = history.view(Access.TEMPORAL_TOO, 0)[0][0].tensor
    assert torch.equal(stored, torch.zeros(4))
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
    assert torch.equal(server_entry.tensor, torch.ones(4))
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
    assert torch.equal(server_entry.tensor, torch.ones(4))
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
