"""Check that round-boundary memory release preserves the next round's side information."""

import subprocess
import sys

import pytest
import torch

from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol
from FL_code.cancer_protocol.cancer_protocol import _WZProtocol
from FL_code.cancer_protocol.history import CompressedReconstructionHistory
from FL_code.cancer_protocol.round_codecs import ServerSIPriorMixin, _WZRoundCodec


@pytest.mark.parametrize("token", ["OR", "TR"])
def test_frozen_round_refreshes_released_side_information(token: str) -> None:
    """Releasing old snapshots keeps trained weights and restores current native/server SI."""
    protocol = create_protocol("wz_cancer", (slice(0, 4),))
    assert isinstance(protocol, _WZProtocol)
    protocol.set_schedule(5, (f"{token}|bins_per_plane=2|num_planes=3|layers=1|hidden_dim=4", "F"))
    for round_id in range(5):
        for client_id in range(2):
            protocol._recons_history.commit(torch.arange(4).float() + round_id + client_id,
                CompressionRecord(round_id, client_id, "I"), Access.TEMPORAL_TOO)
    codec = protocol.create_round_codec(5, 0)
    assert isinstance(codec, _WZRoundCodec)
    codec.quantizer.set_side_information(codec.si)
    codec.quantizer.vector_size = 4
    codec.quantizer.quantizer_prior_stage_rates = (.1, .2, .3)
    weights = {name: value.clone() for name, value in codec.quantizer.coding_model.state_dict().items()}
    codec.release_round_resources()
    assert not codec.si and codec.quantizer.side_info_list_used == []
    if isinstance(codec, ServerSIPriorMixin):
        assert not codec._server_si_prior_side_information
    for client_id in range(2):
        protocol._recons_history.commit(torch.arange(4).float() + 20 + client_id,
            CompressionRecord(5, client_id, token), codec.history_access)
    refreshed = protocol.create_round_codec(6, 0)
    expected = [entry.value for entries in protocol._recons_history.view(codec.history_access, 0).values() for entry in entries]
    assert refreshed is codec and codec.quantizer.side_info_list_used == expected
    if isinstance(codec, ServerSIPriorMixin):
        expected_server = [entry.value for entries in protocol._recons_history.view(Access.SERVER_ONLY, 0).values() for entry in entries]
        assert codec._server_si_prior_side_information == expected_server
    for name, value in codec.quantizer.coding_model.state_dict().items():
        assert torch.equal(value, weights[name])


def test_monitors_do_not_import_training_stack() -> None:
    """Detached monitoring must not retain torch, pandas, or matplotlib while waiting."""
    subprocess.run((sys.executable, "-c", "import sys; import Experiments.Protocol_test.detached_replay; "
        "assert not {'torch', 'pandas', 'matplotlib'} & sys.modules.keys()"), check=True)


def test_server_only_seeding_preserves_decoder_values() -> None:
    """Omitting an unused worker ledger preserves every server-visible value and its ordering."""
    histories = [CompressedReconstructionHistory(5, (slice(0, 4),)) for _ in range(2)]
    for round_id in range(5):
        for client_id in range(2):
            for history, access in zip(histories, (Access.TEMPORAL_TOO, Access.SERVER_ONLY), strict=True):
                history.commit(torch.arange(4).float() + round_id + client_id,
                    CompressionRecord(round_id, client_id, "I"), access)
    views = [history.view(Access.SERVER_ONLY, 0) for history in histories]
    assert list(views[0]) == list(views[1])
    for client_id in views[0]:
        for old, new in zip(views[0][client_id], views[1][client_id], strict=True):
            assert torch.equal(old.value.materialize(), new.value.materialize())
    assert histories[0]._temporal and not histories[1]._temporal
