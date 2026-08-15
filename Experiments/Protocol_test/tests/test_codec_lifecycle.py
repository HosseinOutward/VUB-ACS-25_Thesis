from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch

from FL_code.FL_core.codec import (
    Access,
    CompressionRecord,
    IdentityProtocol,
    simulate_compression,
)
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol
from FL_code.cancer_protocol.round_codecs import F_RoundCodec, T_RoundCodec, _WZRoundCodec


def test_simulation_records_pre_encode_training_separately(tmp_path: Path) -> None:
    """Record the pre-encode hook independently from setup and symbol encoding."""
    delta = torch.arange(1, 5, dtype=torch.float32)
    simulate_compression(
        IdentityProtocol(),
        delta,
        client_id=0,
        round_id=0,
        sd_manager=SimpleNamespace(param_count=delta.numel()),
        save_dir=tmp_path,
        server_eval_metrics={},
        worker_eval_metrics=(),
        metric_keys=(),
    )

    record = pd.read_csv(tmp_path / "compression_records.csv").iloc[0]
    assert record["pre_encode_training_seconds"] >= 0
    assert record["encode_seconds"] >= 0
    assert record["total_seconds"] >= (
        record["setup_seconds"]
        + record["pre_encode_training_seconds"]
        + record["encode_seconds"]
        + record["decode_seconds"]
    )


def test_t_training_is_required_once_and_f_reuses_the_exact_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train T explicitly, transmit its model once, then reuse it through F without training."""
    protocol = WZCancerProtocol()
    protocol.warmup_round_codecs = ()
    protocol.routine_round_codecs = (
        "T|bins_per_plane=2|num_planes=1",
        "F",
    )
    protocol._round_codec_classes = protocol._validate_round_codecs()
    for round_id in range(5):
        protocol._recons_history.commit(
            torch.full((4,), float(round_id)),
            CompressionRecord(round_id, 0, "seed"),
            Access.TEMPORAL_TOO,
        )

    codec = protocol.create_round_codec(0, 0)
    assert isinstance(codec, T_RoundCodec)
    delta = torch.arange(4, dtype=torch.float32)
    record = codec.create_r_record(0, 0)
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
    reused = protocol.create_round_codec(1, 0)
    assert reused is codec
    assert reused.round_name_full == "F"
    assert torch.equal(reused.quantizer.side_info_list_used[-1], latest)

    frozen_record = reused.create_r_record(1, 0)
    reused.pre_encode_training_steps(delta)
    assert training_calls == 1
    assert "quantizer_state" not in reused.get_encod_payload(delta, frozen_record)


def test_f_requires_a_completed_training_round() -> None:
    """Reject a frozen schedule entry when the client has no reusable codec."""
    protocol = WZCancerProtocol()
    protocol.warmup_round_codecs = ()
    protocol.routine_round_codecs = ("F",)
    protocol._round_codec_classes = protocol._validate_round_codecs()
    round_codec_class, parsed, round_name_full = protocol._get_curr_round_codec_name(0)
    assert round_codec_class is F_RoundCodec
    assert parsed.options is None and round_name_full == "F"
    with pytest.raises(AssertionError, match="No frozen quantizer"):
        protocol.create_round_codec(0, 3)


def test_f_protocol_behavior_rejects_options() -> None:
    """Keep F as an option-free protocol marker rather than a configurable codec."""
    protocol = WZCancerProtocol()
    protocol.warmup_round_codecs = ()
    protocol.routine_round_codecs = ("F|refresh=True",)
    with pytest.raises(AssertionError, match="does not accept options"):
        protocol._validate_round_codecs()
