"""Focused tests for the two feedback-driven Slepian--Wolf round-codec mixins."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
import torch

from FL_code.FL_core.utils import compress_data_list, decompress_data_list
from FL_code.cancer_protocol.bin_coder.practical_sw_mixin import (
    LDPCAConfig,
    LDPCASWPayload,
    LDPCASWRecord,
    LDPCASWRoundMixin,
    PracticalSWConfig,
    PracticalSWPayload,
    PracticalSWRecord,
    PracticalSWRoundMixin,
    SWRepair,
    _RepairReplay,
)
from FL_code.cancer_protocol.bin_coder.sw_coder import (
    LDPCA_BACKEND,
    HashVerification,
    PlaneHashes,
    SlepianWolfCodec,
    SWCoderConfig,
    SWLadderPayload,
    SWLadderReplay,
    _binary_entropy,
    _boundary_index,
    _ladder_syndrome,
    _lsb_plane_probabilities,
    expected_plane_rates,
    plane_rows,
    split_boundaries,
)


class _CausalPrior:
    """Binary symmetric chain: each stage flips its predecessor with a fixed crossover."""

    def __init__(self, crossover: float) -> None:
        self.crossover = crossover

    def compute_prior_from_network(
        self, model: object, decoded_stages: torch.Tensor, side_info: torch.Tensor
    ) -> torch.Tensor:
        contexts = (side_info[:, 0].long(), *(stage.long() for stage in decoded_stages[:-1]))
        priors = []
        for likely in contexts:
            probability_one = torch.where(likely.bool(), 1 - self.crossover, self.crossover).float()
            priors.append(torch.stack((1 - probability_one, probability_one), dim=1))
        return torch.stack(priors)


class _FakeQuantizer:
    def __init__(self, bins: torch.Tensor, side_info: torch.Tensor, crossover: float = 0.11) -> None:
        self.source_bins = bins
        self.side_info = side_info
        self.bins_per_plane = 2
        self.num_planes = bins.shape[0]
        self.coding_model = object()
        self.prior_calculator = _CausalPrior(crossover)

    def side_info_tensor(self, metadata: object) -> torch.Tensor:
        return self.side_info

    def decoding_stages_process(self, payload_content: tuple[torch.Tensor, object]) -> torch.Tensor:
        return payload_content[0].float()


class _FakeIndexRoundCodec:
    def __init__(self, quantizer: _FakeQuantizer) -> None:
        self.quantizer = quantizer
        self.last_stage_reconstructions: torch.Tensor | None = None

    def get_encod_payload(self, delta_vec: torch.Tensor, record: object) -> dict[str, Any]:
        return {"payload_content": (self.quantizer.source_bins, {"metadata": True}),
                "quantizer_state": "preserved"}

    def add_stage_distortion_to_record(self, stage_reconstructions: torch.Tensor, record: object) -> None:
        self.last_stage_reconstructions = stage_reconstructions


class _PracticalHost(PracticalSWRoundMixin, _FakeIndexRoundCodec):
    pass


class _LadderHost(LDPCASWRoundMixin, _FakeIndexRoundCodec):
    pass


def _problem(positions: int = 4_800, stages: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a staged binary source correlated with one bit of side information."""
    generator = torch.Generator().manual_seed(7)
    side = torch.randint(0, 2, (positions, 1), generator=generator)
    planes = [side[:, 0]]
    for _ in range(stages):
        planes.append(planes[-1] ^ (torch.rand(positions, generator=generator) < 0.11).long())
    return torch.stack(planes[1:]), side


def _host(bins: torch.Tensor, side: torch.Tensor, **options: Any) -> _PracticalHost:
    return _PracticalHost(
        _FakeQuantizer(bins, side),
        sw_config=SWCoderConfig(block_length=32, max_bp_iterations=30),
        practical_config=PracticalSWConfig(**options),
    )


def _ladder_host(bins: torch.Tensor, side: torch.Tensor, **options: Any) -> _LadderHost:
    return _LadderHost(
        _FakeQuantizer(bins, side),
        sw_config=SWCoderConfig(
            block_length=64, max_bp_iterations=30, matrix_backend=LDPCA_BACKEND, ladder_step=4),
        ldpca_config=LDPCAConfig(**options),
    )


def _exact_stage_entropies(
    prior_provider: Any, side: torch.Tensor, bins: torch.Tensor
) -> tuple[float, ...]:
    """Enumerate the model's own conditional entropy per stage over every earlier-stage context."""
    positions, stages = bins.shape[1], bins.shape[0]
    weights = torch.ones(1, positions)
    contexts = [torch.zeros((stages, positions), dtype=torch.long)]
    entropies: list[float] = []
    for stage in range(stages):
        priors = [prior_provider(context, side, stage) for context in contexts]
        entropies.append(
            float(sum(
                (weight * _binary_entropy(prior[:, 0], prior[:, 1])).sum()
                for weight, prior in zip(weights, priors, strict=True)
            ) / positions)
        )
        weights = torch.cat([
            torch.stack([weight * prior[:, symbol] for symbol in range(2)])
            for weight, prior in zip(weights, priors, strict=True)
        ])
        contexts = [
            context.clone().index_copy_(
                0, torch.tensor([stage]), torch.full((1, positions), symbol, dtype=torch.long))
            for context in contexts
            for symbol in range(2)
        ]
    return tuple(entropies)


def test_expected_plane_rates_follow_the_chain_rule() -> None:
    """Match the enumerated model entropy of every stage without observing one symbol."""
    bins, side = _problem()
    host = _host(bins, side, pilot_blocks=2)
    _, prior_provider = host._sw_context({"metadata": True})
    rates = expected_plane_rates(prior_provider, side.float(), bins.shape[0], 2, samples=32, seed=3)
    exact = _exact_stage_entropies(prior_provider, side.float(), bins)

    # Stage one conditions on nothing drawn, so its rate is exact rather than sampled.
    assert rates[0][0] == pytest.approx(exact[0], abs=1e-6)
    assert rates[1][0] == pytest.approx(exact[1], abs=5e-3)


def test_pilot_split_is_one_permutation_of_every_position() -> None:
    """Cover every position exactly once, with the pilot a whole number of blocks."""
    bins, side = _problem()
    host = _host(bins, side, pilot_blocks=3)
    pilot, bulk = host._split(bins.shape[1])
    assert pilot.numel() == 3 * host.sw_config.block_length
    torch.testing.assert_close(
        torch.cat((pilot, bulk)).sort().values, torch.arange(bins.shape[1]))


def test_hashes_reject_a_block_that_is_not_the_source() -> None:
    """Catch a decode that satisfies its syndrome but differs from the transmitted block."""
    hashes = PlaneHashes(block_length=32, hash_bits=16, seed=0)
    generator = torch.Generator().manual_seed(11)
    plane = torch.randint(0, 2, (4, 32), generator=generator).float()
    digests = torch.stack([hashes.digest(plane)])
    verify = HashVerification(hashes, digests, bit_width=1)
    blocks = torch.arange(4)
    assert bool(verify(0, 0, plane, blocks).all())

    corrupted = plane.clone()
    corrupted[2, 5] = 1 - corrupted[2, 5]
    assert verify(0, 0, corrupted, blocks).tolist() == [True, True, False, True]
    # A ladder only ever verifies the blocks still climbing, so a subset must verify the same way.
    assert verify(0, 0, corrupted[2:], blocks[2:]).tolist() == [False, True]


def test_repairs_make_a_deliberately_tight_plan_exact() -> None:
    """Recover every symbol at a rate chosen to fail, and charge the repairs it needed."""
    bins, side = _problem()
    host = _host(bins, side, pilot_blocks=2, bootstrap_margin=0.0, safety_rows=0)
    payload = host.encode_staged_indices(bins, {"metadata": True})
    assert payload.repaired_blocks > 0
    assert payload.uplink_bits > sum(
        part.wire_bits for _, part, _ in payload.parts
    ), "repairs and hashes must be charged on top of the syndromes."
    torch.testing.assert_close(host.decode_staged_indices(payload, {"metadata": True}), bins)


def test_one_shot_ships_the_a_priori_plan_and_nothing_else() -> None:
    """Send the plan the prior alone justifies when no pilot is bought."""
    bins, side = _problem()
    host = _host(bins, side, pilot_blocks=0, bootstrap_margin=0.1)
    payload = host.encode_staged_indices(bins, {"metadata": True})
    assert payload.pilot is None and payload.pilot_plan is None

    _, prior_provider = host._sw_context({"metadata": True})
    pilot, bulk = host._split(bins.shape[1])
    assert pilot.numel() == 0
    expected = plane_rows(
        expected_plane_rates(prior_provider, side.float()[bulk], bins.shape[0], 2, 2, 0),
        0.1,
        host.sw_config.block_length,
    )
    assert payload.bulk_plan.syndrome_rows == expected
    torch.testing.assert_close(host.decode_staged_indices(payload, {"metadata": True}), bins)


def test_round_payload_preserves_outer_fields_and_records_its_rates() -> None:
    """Keep the wrapped codec's payload intact and account for both directions of the link."""
    bins, side = _problem()
    host = _host(bins, side, pilot_blocks=2)
    record = PracticalSWRecord(round_id=0, client_id=0, round_name_full="PSW")
    outer = host.get_encod_payload(torch.zeros(bins.shape[1]), record)
    assert outer["quantizer_state"] == "preserved"
    transmitted, metadata = outer["payload_content"]
    assert isinstance(transmitted, PracticalSWPayload)
    assert record.sw_uplink_rate == pytest.approx(transmitted.uplink_bits / bins.shape[1])
    assert record.sw_downlink_rate is not None and record.sw_downlink_rate > 0
    assert record.sw_pilot_cross_entropy_rate is not None

    restored = decompress_data_list(compress_data_list(outer))["payload_content"][0]
    torch.testing.assert_close(host.decode_staged_indices(restored, metadata), bins)
    reconstruction = host.decode(compress_data_list(outer), record)
    torch.testing.assert_close(reconstruction, bins[-1].float())


def test_replay_refuses_a_repair_the_encoder_never_recorded() -> None:
    """Fail loudly when a decode rerun asks for a block the payload does not carry."""
    replay = _RepairReplay(SWRepair(block_length=32, hash_bits=0, hashed_blocks=0))
    with pytest.raises(AssertionError, match="did not reproduce"):
        replay(0, 0, torch.tensor([1]))


def test_a_pilot_without_a_way_to_be_exact_is_rejected() -> None:
    """Refuse to calibrate on a pilot the decoder cannot verify or repair."""
    with pytest.raises(ValidationError):
        PracticalSWConfig(pilot_blocks=4, verification_bits=0)


def test_zero_bit_block_verification_is_rejected_without_a_pilot() -> None:
    """Keep the one-shot decoder from accepting every candidate through an empty hash."""
    with pytest.raises(ValidationError):
        PracticalSWConfig(pilot_blocks=0, verification_bits=0)


def test_ladder_levels_are_nested_and_accumulate() -> None:
    """Make every rung a refinement of the one below it, and differencing recover its syndrome."""
    for level in range(1, 64):
        assert set(split_boundaries(64, level)) <= set(split_boundaries(64, level + 1))
    accumulated = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    torch.testing.assert_close(_ladder_syndrome(accumulated), torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
    assert _boundary_index(64, 4).tolist() == [15, 31, 47, 63]


def test_ladder_decodes_every_block_and_charges_the_rung_it_stopped_at() -> None:
    """Recover the source exactly and account for the ladder one block at a time."""
    bins, side = _problem(positions=1_280)
    host = _ladder_host(bins, side)
    payload = host.encode_staged_indices(bins, {"metadata": True})
    torch.testing.assert_close(host.decode_staged_indices(payload, {"metadata": True}), bins)

    levels = payload.ladder.levels
    assert len(levels) == bins.shape[0] and len(levels[0]) == bins.shape[1] // 64
    assert len({level for plane in levels for level in plane}) > 1, (
        "a rate-adaptive transport whose blocks all stop at the same rung is not adapting."
    )
    assert payload.ladder.uplink_syndrome_bits == sum(map(sum, levels))
    assert payload.uplink_bits == (
        payload.ladder.uplink_bits + payload.hashed_blocks * payload.hash_bits)
    assert payload.downlink_bits > payload.ladder.requests


def test_ladder_starts_where_the_model_entropy_says_and_never_below() -> None:
    """Take the first rung from the prior alone, which is the only rate a decoder can name."""
    bins, side = _problem(positions=1_280)
    host = _ladder_host(bins, side)
    _, prior_provider = host._sw_context({"metadata": True})
    plan, apriori = host._start_plan(bins.shape[0], 2, prior_provider, side.float())
    assert plan.syndrome_rows == plane_rows(apriori, 0.0, 64)
    payload = host.encode_staged_indices(bins, {"metadata": True})
    assert all(
        level >= max(1, plan.syndrome_rows[plane][0])
        for plane, levels in enumerate(payload.ladder.levels)
        for level in levels
    )


def test_ladder_replay_refuses_a_rung_the_worker_never_sent() -> None:
    """Fail loudly when a decode rerun climbs past what the payload carries."""
    bins, side = _problem(positions=1_280)
    host = _ladder_host(bins, side)
    payload = host.encode_staged_indices(bins, {"metadata": True})
    replay = SWLadderReplay(payload.ladder)
    highest = max(payload.ladder.levels[0])
    with pytest.raises(AssertionError, match="did not reproduce"):
        replay.request(0, torch.arange(len(payload.ladder.levels[0])), highest + 1)


def test_ladder_round_payload_survives_the_wire_and_records_its_rungs() -> None:
    """Keep the wrapped codec's payload intact and account for both directions of the link."""
    bins, side = _problem(positions=1_280)
    host = _ladder_host(bins, side)
    record = LDPCASWRecord(round_id=0, client_id=0, round_name_full="LSW")
    outer = host.get_encod_payload(torch.zeros(bins.shape[1]), record)
    assert outer["quantizer_state"] == "preserved"
    transmitted, metadata = outer["payload_content"]
    assert isinstance(transmitted, LDPCASWPayload)
    assert record.sw_uplink_rate == pytest.approx(transmitted.uplink_bits / bins.shape[1])
    assert record.sw_raw_blocks == len(transmitted.ladder.raw)
    assert sum(transmitted.stage_uplink_bits) == transmitted.uplink_bits

    restored = decompress_data_list(compress_data_list(outer))["payload_content"][0]
    torch.testing.assert_close(host.decode_staged_indices(restored, metadata), bins)
    torch.testing.assert_close(host.decode(compress_data_list(outer), record), bins[-1].float())


def test_a_ladder_payload_must_name_its_own_plan() -> None:
    """Refuse a ladder whose plan identifier is not the plan that travelled with it."""
    bins, side = _problem(positions=1_280)
    host = _ladder_host(bins, side)
    payload = host.encode_staged_indices(bins, {"metadata": True})
    wrong = payload.ladder.model_copy(update={"plan_id": "0" * 16})
    with pytest.raises(Exception, match="does not match"):
        LDPCASWPayload(plan=payload.plan, ladder=wrong,
                       hash_bits=payload.hash_bits, hashes=payload.hashes)


def test_accumulated_prefix_differences_into_the_merged_syndrome() -> None:
    """Recover each level's parity-check product from the accumulator the encoder actually sends."""
    bins, side = _problem(positions=256, stages=1)
    codec = SlepianWolfCodec(SWCoderConfig(block_length=64, matrix_backend=LDPCA_BACKEND))
    plan = codec.initial_plan(1, 2).model_copy(update={"syndrome_rows": ((32,),)})
    ladder = codec.accumulate_ladder(bins, plan)
    for level in (1, 7, 32, 64):
        served = ladder.accumulated[0][:, _boundary_index(64, level)].float()
        direct = codec._plane_syndrome(ladder.planes[0], plan, 0, 0, level)
        torch.testing.assert_close(_ladder_syndrome(served), direct)


def test_the_ldpca_backend_also_serves_the_fixed_rate_path() -> None:
    """Round-trip the plain syndrome codec on merged matrices, so both families share one core."""
    bins, side = _problem(positions=256, stages=1)
    codec = SlepianWolfCodec(SWCoderConfig(block_length=64, matrix_backend=LDPCA_BACKEND))
    plan = codec.initial_plan(1, 2).model_copy(update={"syndrome_rows": ((64,),)})
    payload = codec.encode_syndromes(bins, plan)
    prior_provider = _CausalPrior(0.11).compute_prior_from_network
    decoded = codec.decode_syndromes(
        payload, plan, lambda stages, info, stage: prior_provider(None, stages, info)[stage],
        side.float())
    torch.testing.assert_close(decoded.symbols, bins)
