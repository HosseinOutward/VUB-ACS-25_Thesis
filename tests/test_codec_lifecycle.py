from __future__ import annotations

from typing import Any

import pytest
import torch

from FL_code.FL_core.codec import Access, CompressionRecord, IdentityRoundCodec
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol
from FL_code.cancer_protocol.history import CompressedReconstructionHistory
from FL_code.cancer_protocol.round_codecs import F_RoundCodec, T_RoundCodec, _WZRoundCodec

TRAIN_TOKEN = "T|bins_per_plane=2|num_planes=1"


def test_protocol_always_uses_compressed_reconstruction_history() -> None:
    protocol = WZCancerProtocol()

    assert isinstance(protocol._recons_history, CompressedReconstructionHistory)
    assert protocol.config_snapshot()["history"]["implementation"].endswith(
        "CompressedReconstructionHistory")


def test_set_schedule_places_training_token_then_freezes_forever() -> None:
    protocol = WZCancerProtocol()
    protocol.set_schedule(3, (TRAIN_TOKEN, "F"))

    assert all(protocol._get_curr_round_codec_name(round_id)[0] is IdentityRoundCodec
               for round_id in range(3))
    assert protocol._get_curr_round_codec_name(3)[0] is T_RoundCodec
    # Every round after the training round must stay frozen; the routine tuple collapses to a
    # single "F" marker precisely so this lookup never cycles back to T, unlike the old draft
    # which kept the caller's whole tuple as the routine and retrained every other round.
    for round_id in (4, 5, 6, 20):
        assert protocol._get_curr_round_codec_name(round_id)[0] is F_RoundCodec

    snapshot = protocol.config_snapshot()
    assert snapshot["warmup_round_codecs"] == ("I", "I", "I", TRAIN_TOKEN)
    assert snapshot["routine_round_codecs"] == ("F",)


def test_set_schedule_may_be_installed_only_once() -> None:
    protocol = WZCancerProtocol()
    protocol.set_schedule(0, (TRAIN_TOKEN, "F"))
    with pytest.raises(AssertionError, match="only once"):
        protocol.set_schedule(1, (TRAIN_TOKEN, "F"))


def test_set_schedule_rejects_a_non_frozen_token_after_training() -> None:
    protocol = WZCancerProtocol()
    with pytest.raises(AssertionError, match="literal frozen marker"):
        protocol.set_schedule(0, (TRAIN_TOKEN, TRAIN_TOKEN))


def test_set_schedule_rejects_a_schedule_with_no_frozen_round() -> None:
    protocol = WZCancerProtocol()
    with pytest.raises(AssertionError, match="training token followed by at least one"):
        protocol.set_schedule(0, (TRAIN_TOKEN,))


def test_set_schedule_rejects_f_as_the_first_token() -> None:
    protocol = WZCancerProtocol()
    with pytest.raises(AssertionError, match="not 'F'"):
        protocol.set_schedule(0, ("F", "F"))


def test_training_then_frozen_round_reuses_the_exact_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = WZCancerProtocol()
    start_round = protocol.max_per_client_recons_history  # chronological warmup needs this much history
    protocol.set_schedule(start_round, (TRAIN_TOKEN, "F"))

    # Legal warmup seeds only rounds strictly before start_round; seeding the start round itself
    # (or later) would hand the codec side information from its own future, which the draft did.
    for round_id in range(start_round):
        protocol._recons_history.commit(
            torch.full((4,), float(round_id)),
            CompressionRecord(round_id, 0, "I"),
            Access.TEMPORAL_TOO,
        )

    codec = protocol.create_round_codec(start_round, 0)
    assert isinstance(codec, T_RoundCodec)
    delta = torch.arange(4, dtype=torch.float32)
    record = codec.create_r_record(start_round, 0)
    training_calls = 0

    def fake_train_model(
        x: torch.Tensor,
        side_information: list[torch.Tensor],
        extra_pairs: tuple[Any, ...],
        problem_weights: tuple[float, ...] | None,
    ) -> None:
        nonlocal training_calls
        training_calls += 1
        assert torch.equal(x, delta)
        assert not extra_pairs and problem_weights is None
        codec.quantizer.side_info_list_used = list(side_information)
        codec.quantizer.vector_size = x.numel()
        codec.quantizer.training_prior_stage_rates = (0.5,)
        codec.quantizer.training_prior_rate = 0.5

    monkeypatch.setattr(codec.quantizer, "train_model", fake_train_model)
    monkeypatch.setattr(
        _WZRoundCodec,
        "get_encod_payload",
        lambda self, delta_vec, compression_record: {"payload_content": delta_vec},
    )

    with pytest.raises(AssertionError, match="pre_encode_training_steps"):
        codec.get_encod_payload(delta, record)
    codec.pre_encode_training_steps(delta)
    assert training_calls == 1
    assert "quantizer_state" in codec.get_encod_payload(delta, record)
    assert "quantizer_state" not in codec.get_encod_payload(delta, record)

    latest = torch.full((4,), 10.0)
    protocol.after_decode(latest, record, codec)
    reused = protocol.create_round_codec(start_round + 1, 0)
    assert reused is codec
    assert reused.round_name_full == "F"
    assert torch.equal(reused.quantizer.side_info_list_used[-1].materialize(), latest)

    reused.pre_encode_training_steps(delta)
    assert training_calls == 1, "a frozen round must not retrain the quantizer"
    assert "quantizer_state" not in reused.get_encod_payload(
        delta, reused.create_r_record(start_round + 1, 0))


def test_frozen_round_requires_a_completed_training_round() -> None:
    protocol = WZCancerProtocol()
    protocol.set_schedule(0, (TRAIN_TOKEN, "F"))
    with pytest.raises(AssertionError, match="No frozen quantizer"):
        protocol.create_round_codec(1, 3)


def test_two_selected_workers_keep_distinct_frozen_codecs_and_si(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two measured workers must not share a trained codec instance or cross-contaminate SI."""
    monkeypatch.setattr(
        _WZRoundCodec,
        "get_encod_payload",
        lambda self, delta_vec, compression_record: {"payload_content": delta_vec},
    )
    protocol = WZCancerProtocol()
    start_round = protocol.max_per_client_recons_history
    protocol.set_schedule(start_round, (TRAIN_TOKEN, "F"))

    for client_id in (0, 1):
        for round_id in range(start_round):
            protocol._recons_history.commit(
                torch.full((4,), float(round_id + client_id * 100)),
                CompressionRecord(round_id, client_id, "I"),
                Access.TEMPORAL_TOO,
            )

    codecs: dict[int, T_RoundCodec] = {}
    training_calls = {0: 0, 1: 0}
    for client_id in (0, 1):
        codec = protocol.create_round_codec(start_round, client_id)
        assert isinstance(codec, T_RoundCodec)

        def fake_train_model(
            x: torch.Tensor,
            si: list[torch.Tensor],
            extra_pairs: tuple[Any, ...],
            problem_weights: tuple[float, ...] | None,
            _client_id: int = client_id,
            _codec: T_RoundCodec = codec,
        ) -> None:
            training_calls[_client_id] += 1
            _codec.quantizer.side_info_list_used = list(si)
            _codec.quantizer.vector_size = x.numel()
            _codec.quantizer.training_prior_stage_rates = (0.5,)
            _codec.quantizer.training_prior_rate = 0.5

        codec.quantizer.train_model = fake_train_model
        delta = torch.full((4,), float(client_id))
        codec.pre_encode_training_steps(delta)
        codec.get_encod_payload(delta, codec.create_r_record(start_round, client_id))
        codecs[client_id] = codec

    assert codecs[0] is not codecs[1]
    assert codecs[0].quantizer is not codecs[1].quantizer

    for client_id in (0, 1):
        record = codecs[client_id].create_r_record(start_round, client_id)
        protocol.after_decode(torch.full((4,), 10.0 + client_id), record, codecs[client_id])

    reused = {
        client_id: protocol.create_round_codec(start_round + 1, client_id)
        for client_id in (0, 1)
    }
    assert reused[0] is codecs[0] and reused[1] is codecs[1]
    assert training_calls == {0: 1, 1: 1}, "reuse on frozen rounds must not retrain either client"
    assert torch.equal(
        reused[0].quantizer.side_info_list_used[-1].materialize(), torch.full((4,), 10.0))
    assert torch.equal(
        reused[1].quantizer.side_info_list_used[-1].materialize(), torch.full((4,), 11.0))
