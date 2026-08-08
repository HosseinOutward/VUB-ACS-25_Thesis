from __future__ import annotations

from typing import Any

import pytest
import torch

from FL_code.FL_core.codec import (
    Access, CompressionRecord, ReconstructionHistory)
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol

from .round_codecs import OMI_RoundCodec, TMI_RoundCodec


def _codec_options() -> dict[str, Any]:
    return {
        "bins_per_plane": 2,
        "num_planes": 2,
        "sd_slices": [slice(None)],
        "hash_table_count": 1,
        "hash_dim": 2,
        "route_count": 3,
        "embedding_dim": 2,
        "hidden_dim": 4,
        "layers": 1,
        "fused_optimizer": False,
    }


def _history() -> ReconstructionHistory:
    history = ReconstructionHistory(
        max_per_client=5,
        routine_accesses=(Access.SERVER_ONLY, Access.TEMPORAL_TOO),
    )
    for round_id in range(5):
        for client_id in (0, 1):
            history.commit(
                torch.tensor([100 * client_id + round_id], dtype=torch.float32),
                CompressionRecord(round_id, client_id, "seed"),
                Access.TEMPORAL_TOO,
            )
    return history


@pytest.mark.parametrize(
    ("round_name", "codec_class", "access"),
    [
        ("OMI", OMI_RoundCodec, Access.SERVER_ONLY),
        ("TMI", TMI_RoundCodec, Access.TEMPORAL_TOO),
    ],
)
def test_protocol_discovers_multi_round_index_codecs_and_retains_lagged_history(
    round_name: str,
    codec_class: type[OMI_RoundCodec | TMI_RoundCodec],
    access: Access,
) -> None:
    protocol = WZCancerProtocol(
        options={"replace_O_with": round_name, "train_offsets": 3},
        sd_slices=[slice(None)],
    )
    discovered = {
        round_codec
        for round_codec in protocol._round_codec_classes.values()
        if getattr(round_codec, "round_name", None) == round_name
    }
    assert discovered == {codec_class}
    assert codec_class.can_decode_where is access
    assert codec_class.history_depth((1, 2, 3)) == 3
    assert protocol.codec_protocol_inputs(codec_class, None) == {
        "history_window": 5,
        "train_offsets": (1, 2, 3),
    }
    assert protocol._lagged_history is not None
    assert protocol._lagged_history.max_per_client == 8


def test_omi_uses_all_server_visible_ledgers_for_current_and_lagged_si() -> None:
    codec = OMI_RoundCodec(
        _codec_options(), "OMI", train_offsets=(1, 3), history_window=2)
    target, current_si = codec.get_training_data(_history(), client_id=0)

    assert target is None
    assert [value.item() for value in current_si] == [3, 4, 103, 104]
    assert [
        (lagged_target.item(), [value.item() for value in lagged_si])
        for lagged_target, lagged_si in codec.extra_training_pairs
    ] == [
        (4, [2, 3, 102, 103]),
        (2, [0, 1, 100, 101]),
    ]


def test_omi_refreshes_current_si_without_changing_snapshotted_lagged_pairs() -> None:
    codec = OMI_RoundCodec(
        _codec_options(), "OMI", train_offsets=(1, 3), history_window=2)
    live_history = _history()
    snapshot = live_history.copy()
    snapshot.freeze()
    _, snapshotted_si = codec.get_training_data(snapshot, client_id=1)
    lagged_pairs = codec.extra_training_pairs
    live_history.commit(
        torch.tensor([5], dtype=torch.float32),
        CompressionRecord(5, 0, "current"),
        Access.SERVER_ONLY,
    )

    current_si = codec.refresh_current_training_side_information(
        live_history, client_id=1, side_information=snapshotted_si
    )

    assert [value.item() for value in current_si] == [4, 5, 103, 104]
    assert codec.extra_training_pairs == lagged_pairs
    assert [
        (target.item(), [value.item() for value in side_information])
        for target, side_information in codec.extra_training_pairs
    ] == [
        (104, [2, 3, 102, 103]),
        (102, [0, 1, 100, 101]),
    ]


def test_tmi_uses_only_requesting_workers_temporal_ledger() -> None:
    codec = TMI_RoundCodec(
        _codec_options(), "TMI", train_offsets=(1, 3), history_window=2)
    history = _history()
    target, current_si = codec.get_training_data(history, client_id=1)

    assert target is None
    assert [value.item() for value in current_si] == [103, 104]
    assert [
        (lagged_target.item(), [value.item() for value in lagged_si])
        for lagged_target, lagged_si in codec.extra_training_pairs
    ] == [
        (104, [102, 103]),
        (102, [100, 101]),
    ]
    assert codec.refresh_current_training_side_information(
        history, client_id=1, side_information=current_si
    ) == current_si


def test_multi_round_codec_rejects_incomplete_visible_history() -> None:
    codec = OMI_RoundCodec(
        _codec_options(), "OMI", train_offsets=(1, 3), history_window=2)
    history = _history()
    history._server[1].pop()
    with pytest.raises(AssertionError, match="needs 5 entries"):
        codec.get_training_data(history, client_id=0)
