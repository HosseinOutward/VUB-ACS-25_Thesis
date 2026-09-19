from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from FL_code.FL_core.codec import Access, CompressionRecord, IdentityRoundCodec
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol
from FL_code.cancer_protocol.history import CompressedReconstructionHistory
from FL_code.cancer_protocol.round_codecs import F_RoundCodec, T_RoundCodec, _WZRoundCodec
from FL_code.cancer_protocol.prior_code import PriorCalculator

TRAIN_TOKEN = "T|bins_per_plane=2|num_planes=1"


def test_protocol_always_uses_compressed_reconstruction_history() -> None:
    protocol = WZCancerProtocol()

    assert isinstance(protocol._recons_history, CompressedReconstructionHistory)
    assert protocol.config_snapshot()["history"]["implementation"].endswith(
        "CompressedReconstructionHistory")


def test_set_schedule_places_the_full_window_then_freezes_forever() -> None:
    protocol = WZCancerProtocol()
    protocol.set_schedule(3, (TRAIN_TOKEN, "F", TRAIN_TOKEN, "F"))

    assert all(protocol._get_curr_round_codec_name(round_id)[0] is IdentityRoundCodec
               for round_id in range(3))
    assert protocol._get_curr_round_codec_name(3)[0] is T_RoundCodec
    assert protocol._get_curr_round_codec_name(4)[0] is F_RoundCodec
    assert protocol._get_curr_round_codec_name(5)[0] is T_RoundCodec
    for round_id in (6, 7, 20):
        assert protocol._get_curr_round_codec_name(round_id)[0] is F_RoundCodec

    snapshot = protocol.config_snapshot()
    assert snapshot["warmup_round_codecs"] == (
        "I", "I", "I", TRAIN_TOKEN, "F", TRAIN_TOKEN, "F")
    assert snapshot["routine_round_codecs"] == ("F",)


def test_set_schedule_may_be_installed_only_once() -> None:
    protocol = WZCancerProtocol()
    protocol.set_schedule(0, (TRAIN_TOKEN, "F"))
    with pytest.raises(AssertionError, match="only once"):
        protocol.set_schedule(1, (TRAIN_TOKEN, "F"))


def test_set_schedule_accepts_retraining_after_a_frozen_round() -> None:
    protocol = WZCancerProtocol()
    protocol.set_schedule(0, (TRAIN_TOKEN, "F", TRAIN_TOKEN, "F"))
    assert protocol._get_curr_round_codec_name(2)[0] is T_RoundCodec


def test_set_schedule_rejects_a_schedule_with_no_frozen_round() -> None:
    protocol = WZCancerProtocol()
    with pytest.raises(AssertionError, match="at least two"):
        protocol.set_schedule(0, (TRAIN_TOKEN,))
    with pytest.raises(AssertionError, match="frozen round"):
        WZCancerProtocol().set_schedule(0, (TRAIN_TOKEN, TRAIN_TOKEN))


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


def test_frozen_round_reuses_retrained_prior_with_new_si(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = T_RoundCodec({"bins_per_plane": 2, "num_planes": 1, "sd_slices": None}, "T")
    prior_model = object()
    training_calls = 0

    def train_prior(*_args: Any, **_kwargs: Any) -> object:
        nonlocal training_calls
        training_calls += 1
        return prior_model

    observed_si: list[torch.Tensor] = []

    def evaluate_prior(
        _model: object, bins: torch.Tensor, side_info: torch.Tensor,
    ) -> torch.Tensor:
        observed_si.append(side_info.clone())
        return torch.full((1, bins.shape[1], 2), 0.5)

    monkeypatch.setattr(PriorCalculator, "make_trained_prior_model", train_prior)
    monkeypatch.setattr(PriorCalculator, "compute_prior_from_network", evaluate_prior)
    quantizer = SimpleNamespace(prior_calculator=PriorCalculator)
    bins = torch.tensor([[0, 1]], dtype=torch.uint8)
    soft_codes = torch.full((1, 2, 2), 0.5)

    _, trained_model = codec.conditional_code_lengths(
        quantizer, bins, soft_codes, torch.tensor([[1.0], [2.0]]), codec.c_cfg)
    codec.round_name_full = "F"
    _, frozen_model = codec.conditional_code_lengths(
        quantizer, bins, soft_codes, torch.tensor([[3.0], [4.0]]), codec.c_cfg)

    assert training_calls == 1
    assert trained_model is frozen_model is prior_model
    assert [side_info[:, 0].tolist() for side_info in observed_si] == [[1.0, 2.0], [3.0, 4.0]]


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
