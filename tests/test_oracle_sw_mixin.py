"""Focused tests for the cheating successive-stage SW round-codec mixin."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest
import torch

from FL_code.FL_core.utils import compress_data_list, decompress_data_list
from FL_code.cancer_protocol.round_codecs import CancerRecord
from FL_code.cancer_protocol.bin_coder.oracle_sw_mixin import (
    OracleSWPayload,
    OracleSWRoundMixin,
    _wilson_upper,
)
from FL_code.cancer_protocol.bin_coder.sw_coder import (
    SWCoderConfig,
    SWPayload,
    SWRatePlan,
    SlepianWolfCodec,
)


class _CausalPrior:
    def __init__(self, crossover: float) -> None:
        self.crossover = crossover
        self.decoded_inputs: list[torch.Tensor] = []

    def compute_prior_from_network(
        self,
        model: object,
        decoded_stages: torch.Tensor,
        side_info: torch.Tensor,
    ) -> torch.Tensor:
        self.decoded_inputs.append(decoded_stages.clone())
        likely_one = (side_info[:, 0].long(), decoded_stages[0].long())
        priors = []
        for likely in likely_one:
            probability_one = torch.where(
                likely.bool(), 1 - self.crossover, self.crossover
            ).float()
            priors.append(torch.stack((1 - probability_one, probability_one), dim=1))
        return torch.stack(priors)


class _FakeQuantizer:
    """``bins=None`` builds decoder-legal state only: no source bins, num_planes given directly."""

    def __init__(
        self,
        bins: torch.Tensor | None,
        side_info: torch.Tensor,
        alphabet: int = 2,
        crossover: float = 0.11,
        num_planes: int | None = None,
    ) -> None:
        self.source_bins = bins
        self.side_info = side_info
        self.bins_per_plane = alphabet
        self.num_planes = num_planes if bins is None else bins.shape[0]
        self.coding_model = object()
        self.prior_calculator = _CausalPrior(crossover)

    def side_info_tensor(self, metadata: object) -> torch.Tensor:
        return self.side_info

    def decoding_stages_process(
        self, payload_content: tuple[torch.Tensor, object]
    ) -> torch.Tensor:
        bins, _ = payload_content
        return bins.float()


class _FakeIndexRoundCodec:
    def __init__(self, quantizer: _FakeQuantizer) -> None:
        self.quantizer = quantizer
        self.last_stage_reconstructions: torch.Tensor | None = None

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: object
    ) -> dict[str, Any]:
        return {
            "payload_content": (self.quantizer.source_bins, {"metadata": True}),
            "quantizer_state": "preserved",
        }

    def add_stage_distortion_to_record(
        self, stage_reconstructions: torch.Tensor, record: object
    ) -> None:
        self.last_stage_reconstructions = stage_reconstructions


class _OracleHost(OracleSWRoundMixin, _FakeIndexRoundCodec):
    pass


def _problem(positions: int = 4_800) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    side = torch.randint(0, 2, (positions, 1), generator=generator)
    stage_one = side[:, 0] ^ (
        torch.rand(positions, generator=generator) < 0.11
    ).long()
    stage_two = stage_one ^ (
        torch.rand(positions, generator=generator) < 0.11
    ).long()
    return torch.stack((stage_one, stage_two)), side


def test_oracle_round_payload_preserves_outer_fields_and_recovers_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bins, side_info = _problem()
    host = _OracleHost(
        _FakeQuantizer(bins, side_info),
        sw_config=SWCoderConfig(block_length=32, max_bp_iterations=30),
    )

    decoder_calls = 0
    original_decode = host.decode_staged_indices

    def counted_decode(payload: OracleSWPayload, metadata: object) -> torch.Tensor:
        nonlocal decoder_calls
        decoder_calls += 1
        return original_decode(payload, metadata)

    monkeypatch.setattr(host, "decode_staged_indices", counted_decode)
    record = CancerRecord(round_id=0, client_id=0, round_name_full="OSW")
    outer = host.get_encod_payload(torch.zeros(bins.shape[1]), record)
    assert decoder_calls == 0
    assert outer["quantizer_state"] == "preserved"
    transmitted, metadata = outer["payload_content"]
    assert isinstance(transmitted, OracleSWPayload)
    restored = decompress_data_list(compress_data_list(outer))
    torch.testing.assert_close(
        host.decode_staged_indices(restored["payload_content"][0], metadata),
        bins,
    )
    assert record.symbol_error_rate == 0.0

    # A genuinely separate decoder: no source bins, only the side information a real decoder
    # would legally hold plus whatever the wire payload carries.
    decoder = _OracleHost(
        _FakeQuantizer(None, side_info, num_planes=bins.shape[0]),
        sw_config=SWCoderConfig(block_length=32, max_bp_iterations=30),
    )
    for _ in range(2):
        reconstruction = decoder.decode(compress_data_list(outer), record)
        torch.testing.assert_close(reconstruction, bins[-1].float())


def test_oracle_plan_is_wilson_safe_and_uses_successive_decoded_context() -> None:
    bins, side_info = _problem()
    host = _OracleHost(
        _FakeQuantizer(bins, side_info),
        sw_config=SWCoderConfig(block_length=32, max_bp_iterations=30),
    )
    transmitted = host.encode_staged_indices(bins, {"metadata": True})
    decoded = host.decode_staged_indices(transmitted, {"metadata": True})
    errors = host._bit_errors(bins, decoded, transmitted.plan.bit_width)
    assert _wilson_upper(errors, bins.numel()) < 0.01

    calls = host.quantizer.prior_calculator.decoded_inputs
    assert any(torch.equal(call[0], bins[0]) for call in calls)
    assert torch.equal(decoded, bins)

    side, prior = host._sw_context({"metadata": True})
    codec = SlepianWolfCodec(host.sw_config)
    bootstrap = codec.initial_plan(bins.shape[0], 2)
    base = codec.decode_syndromes(
        codec.encode_syndromes(bins, bootstrap), bootstrap, prior, side
    ).next_plan
    offsets = {
        chosen - initial
        for chosen_stage, initial_stage in zip(
            transmitted.plan.syndrome_rows, base.syndrome_rows, strict=True
        )
        for chosen, initial in zip(chosen_stage, initial_stage, strict=True)
        if initial < host.sw_config.block_length
    }
    assert len(offsets) == 1
    selected_offset = offsets.pop()
    for row_backoff in range(selected_offset):
        lower = base.model_copy(update={
            "syndrome_rows": tuple(
                tuple(
                    min(host.sw_config.block_length, rows + row_backoff)
                    for rows in stage_rows
                )
                for stage_rows in base.syndrome_rows
            )
        })
        lower_decoded = codec.decode_syndromes(
            codec.encode_syndromes(bins, lower), lower, prior, side
        ).symbols
        lower_errors = host._bit_errors(bins, lower_decoded, lower.bit_width)
        assert _wilson_upper(lower_errors, bins.numel()) >= 0.01


def test_later_stage_prior_uses_recovered_not_true_earlier_stage() -> None:
    bins, side_info = _problem()
    quantizer = _FakeQuantizer(bins, side_info)
    host = _OracleHost(
        quantizer,
        sw_config=SWCoderConfig(block_length=32, max_bp_iterations=30),
    )
    zero_plan = SWRatePlan.initial(2, 2, host.sw_config).model_copy(
        update={"syndrome_rows": ((0,), (0,))}
    )
    zero_payload = OracleSWPayload(
        plan=zero_plan,
        syndrome=SlepianWolfCodec(host.sw_config).encode_syndromes(bins, zero_plan),
    )
    quantizer.prior_calculator.decoded_inputs.clear()
    decoded = host.decode_staged_indices(zero_payload, {"metadata": True})

    stage_two_input = quantizer.prior_calculator.decoded_inputs[-1][0]
    torch.testing.assert_close(stage_two_input, decoded[0])
    assert not torch.equal(stage_two_input, bins[0])


def test_payload_rejects_a_mismatched_plan() -> None:
    cfg = SWCoderConfig(block_length=32)
    first = SWRatePlan.initial(2, 2, cfg)
    second = first.model_copy(update={"code_seed": 1})
    syndrome = SWPayload(positions=32, plan_id=first.identifier, syndromes=b"\x00" * 8)
    with pytest.raises(ValidationError, match="does not match"):
        OracleSWPayload(plan=second, syndrome=syndrome)


def test_non_power_of_two_alphabet_fails_loudly() -> None:
    bins, side_info = _problem(32)
    host = _OracleHost(
        _FakeQuantizer(bins, side_info, alphabet=3),
        sw_config=SWCoderConfig(block_length=32),
    )
    with pytest.raises(AssertionError, match="power-of-two"):
        host.encode_staged_indices(bins, {"metadata": True})
