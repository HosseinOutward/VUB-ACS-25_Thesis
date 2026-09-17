"""Focused tests for the shared staged-index transport, rANS wrapper, and oracle-prior grant."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from FL_code.FL_core.utils import compress_data_list, decompress_data_list
from FL_code.cancer_protocol.bin_coder.coded_rounds import TRansOraclePriorRoundCodec
from FL_code.cancer_protocol.bin_coder.rans_coder import HEADER, RansConfig
from FL_code.cancer_protocol.bin_coder.rans_mixin import RansRoundMixin
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.round_codecs import CancerRecord, T_RoundCodec


class _PriorCalculator:
    @staticmethod
    def compute_prior_from_network(
        model: object, decoded_stages: torch.Tensor, side_info: torch.Tensor
    ) -> torch.Tensor:
        likely = torch.stack((side_info[:, 0].long(), decoded_stages[0].long()))
        probability_one = torch.where(likely.bool(), 0.9, 0.1).float()
        return torch.stack((1 - probability_one, probability_one), dim=2)


class _EncoderQuantizer:
    """Legal encoder-side state: the source bins plus everything a decoder also sees."""

    bins_per_plane = 2
    num_planes = 2
    coding_model = object()
    prior_calculator = _PriorCalculator()

    def __init__(self, bins: torch.Tensor, side_info: torch.Tensor) -> None:
        self.bins = bins
        self.side_info = side_info

    def side_info_tensor(self, metadata: object) -> torch.Tensor:
        return self.side_info

    @staticmethod
    def decoding_stages_process(payload_content: tuple[torch.Tensor, object]) -> torch.Tensor:
        return payload_content[0].float()


class _DecoderQuantizer:
    """Legal decoder-side state only: side information and the shared model, never source bins."""

    bins_per_plane = 2
    num_planes = 2
    coding_model = object()
    prior_calculator = _PriorCalculator()

    def __init__(self, side_info: torch.Tensor) -> None:
        self.side_info = side_info

    def side_info_tensor(self, metadata: object) -> torch.Tensor:
        return self.side_info

    @staticmethod
    def decoding_stages_process(payload_content: tuple[torch.Tensor, object]) -> torch.Tensor:
        return payload_content[0].float()


class _IndexRoundCodec:
    def __init__(self, quantizer: Any) -> None:
        self.quantizer = quantizer

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        return {
            "payload_content": (self.quantizer.bins, {"metadata": True}),
            "quantizer_state": "preserved",
        }

    def add_stage_distortion_to_record(
        self, stage_reconstructions: torch.Tensor, record: CancerRecord
    ) -> None:
        pass


class _RansHost(RansRoundMixin, _IndexRoundCodec):
    pass


def _rans_problem(positions: int = 257) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    side_info = torch.randint(0, 2, (positions, 1), generator=generator)
    first = side_info[:, 0] ^ (torch.rand(positions, generator=generator) < 0.1).long()
    second = first ^ (torch.rand(positions, generator=generator) < 0.1).long()
    return torch.stack((first, second)), side_info


def test_rans_transport_accounts_for_header_and_flush_and_decodes_separately() -> None:
    """Charge fixed wire overhead and decode with a distinct object holding no source bins."""
    bins, side_info = _rans_problem()
    config = RansConfig(lanes=8)
    encoder = _RansHost(_EncoderQuantizer(bins, side_info), rans_config=config)
    record = CancerRecord(round_id=0, client_id=0, round_name_full="TR")

    outer = encoder.get_encod_payload(torch.zeros(bins.shape[1]), record)

    assert record.coder_overhead_rate == pytest.approx(
        8 * (HEADER.size + 4 * config.lanes) / bins.shape[1]
    )
    assert record.coder_rate == pytest.approx(sum(record.coder_stage_rates or ()))
    assert record.symbol_error_rate == 0.0
    assert outer["quantizer_state"] == "preserved"

    # A genuinely separate decoder: no source bins, no reference to the encoder object, only the
    # side information a real worker/server would legally hold plus the wire payload.
    decoder = _RansHost(_DecoderQuantizer(side_info), rans_config=config)
    compressed = compress_data_list(outer)
    reconstructed = decoder.decode(compressed, record)
    torch.testing.assert_close(reconstructed, bins[-1].float())
    # Every other outer payload field, including the initial quantizer state, survives the wrap.
    assert decompress_data_list(compressed)["quantizer_state"] == "preserved"


def _tiny_wz_options(positions: int, **overrides: Any) -> dict[str, Any]:
    """Return construction options for a fast, deterministic CUDA quantizer round-trip."""
    options: dict[str, Any] = dict(
        bins_per_plane=2,
        num_planes=1,
        sd_slices=[slice(0, positions)],
        cache_dir=None,
        cache_overwrite=False,
        hidden_dim=4,
        layers=1,
        train_epochs=2,
        train_sample_size=positions,
        train_batch_size=positions,
        quantizer_train_repeats=1,
        prior_train_repeats=1,
        deduplicate_training=False,
    )
    options.update(overrides)
    return options


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantizer training requires CUDA")
def test_trr_oracle_prior_grant_travels_on_the_wire_and_decodes_independently() -> None:
    """TRR's source-fitted prior must be explicit transmitted state, not a same-object shortcut."""
    torch.manual_seed(0)
    positions = 96
    delta = torch.randn(positions)
    si = [HistoryVector(torch.randn(positions)), HistoryVector(torch.randn(positions))]

    encoder = TRansOraclePriorRoundCodec(_tiny_wz_options(positions, rans_lanes=4), "TRR")
    encoder.build_quantizer(None, si)
    encoder.pre_encode_training_steps(delta)
    record = CancerRecord(round_id=0, client_id=0, round_name_full="TRR")
    payload = encoder.encode(delta, record)

    decoded_outer = decompress_data_list(payload)
    assert "oracle_prior_state" not in decoded_outer, (
        "the fitted prior is an out-of-band grant; it must never be serialized into the wire "
        "payload or it would silently inflate final_bytes for every TRR/OSWR round.")
    assert "quantizer_state" in decoded_outer, "the first trained round must also send decoder parameters."
    assert encoder.oracle_prior_grant is not None, "encoding must set the explicit grant attribute."

    # A distinct decoder codec: fresh instance, architecture rebuilt from the same declared
    # options, decoder parameters loaded from the payload, only the worker's own legal SI, and the
    # grant handed off explicitly (the one line that stands in for an out-of-band oracle channel)
    # -- never the encoder's quantizer object, its _oracle_prior, or the original delta.
    decoder = TRansOraclePriorRoundCodec(_tiny_wz_options(positions, rans_lanes=4), "TRR")
    decoder.quantizer = decoder.quantizer_class(c_cfg=decoder.c_cfg, si_size=len(si))
    decoder.quantizer.vector_size = positions
    decoder.quantizer.coding_model.load_state_dict(decoded_outer["quantizer_state"], strict=False)
    decoder.quantizer.set_side_information(si)
    decoder.oracle_prior_grant = encoder.oracle_prior_grant
    assert not hasattr(decoder, "_oracle_prior"), "the decoder must not start with an encoder-side prior."

    reconstruction = decoder.decode(payload, record)
    transmitted, metadata = decoded_outer["payload_content"]
    encoder_bins = encoder.decode_staged_indices(transmitted, metadata)
    encoder_reconstruction = encoder.quantizer.decoding_stages_process((encoder_bins, metadata))
    torch.testing.assert_close(reconstruction, encoder_reconstruction[-1])
    assert record.coder_rate == pytest.approx(sum(record.coder_stage_rates or ()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantizer training requires CUDA")
def test_seeded_training_leaves_caller_rng_identical_on_hit_and_miss(tmp_path: Any) -> None:
    """Cache hits skip training; a miss must fork+restore RNG so both leave it unperturbed."""
    positions = 64
    delta = torch.randn(positions)
    si = [HistoryVector(torch.randn(positions)), HistoryVector(torch.randn(positions))]
    cache_dir = tmp_path / "cache"
    device = torch.cuda.current_device()

    def rng_state() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()

    def run_and_capture() -> tuple[torch.Tensor, torch.Tensor]:
        # Reset and build the (uncached) model identically each call, then isolate the RNG effect
        # of exactly the cached call: pre_encode_training_steps -> quantizer.train_model.
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        codec = T_RoundCodec(
            _tiny_wz_options(positions, cache_dir=cache_dir, training_seed=7), "T")
        codec.build_quantizer(None, si)
        before = rng_state()
        codec.pre_encode_training_steps(delta)
        after = rng_state()
        torch.testing.assert_close(after[0], before[0])
        torch.testing.assert_close(after[1], before[1])
        return after

    after_miss = run_and_capture()  # first call: cache miss, actually trains
    after_hit = run_and_capture()  # second call: cache hit, no training at all
    torch.testing.assert_close(after_miss[0], after_hit[0])
    torch.testing.assert_close(after_miss[1], after_hit[1])


def test_trr_decode_without_an_explicit_grant_fails_loudly() -> None:
    """A decoder must never fall back to reading the grant off the wire or the encoder object."""
    encoder = TRansOraclePriorRoundCodec.__new__(TRansOraclePriorRoundCodec)
    encoder.round_name_full = "TRR"
    with pytest.raises(AssertionError, match="oracle_prior_grant"):
        encoder.decode(compress_data_list({"payload_content": (b"", {})}), CancerRecord(
            round_id=0, client_id=0, round_name_full="TRR"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantizer training requires CUDA")
def test_matched_transports_share_the_cached_quantizer_and_a_changed_seed_misses(
    tmp_path: Any,
) -> None:
    """T and TR train the same quantizer for identical inputs; a different seed must not collide."""
    positions = 96
    torch.manual_seed(3)
    delta = torch.randn(positions)
    si = [HistoryVector(torch.randn(positions)), HistoryVector(torch.randn(positions))]
    cache_dir = tmp_path / "cache"

    plain = T_RoundCodec(_tiny_wz_options(positions, cache_dir=cache_dir), "T")
    plain.build_quantizer(None, si)
    plain.pre_encode_training_steps(delta)

    transported = TRansOraclePriorRoundCodec(
        _tiny_wz_options(positions, cache_dir=cache_dir, rans_lanes=4), "TRR")
    transported.build_quantizer(None, si)
    transported.pre_encode_training_steps(delta)

    for key in plain.quantizer.coding_model.state_dict():
        torch.testing.assert_close(
            plain.quantizer.coding_model.state_dict()[key],
            transported.quantizer.coding_model.state_dict()[key],
        )

    seeded_once = T_RoundCodec(
        _tiny_wz_options(positions, cache_dir=cache_dir, training_seed=1), "T")
    seeded_once.build_quantizer(None, si)
    seeded_once.pre_encode_training_steps(delta)

    seeded_twice = T_RoundCodec(
        _tiny_wz_options(positions, cache_dir=cache_dir, training_seed=2), "T")
    seeded_twice.build_quantizer(None, si)
    seeded_twice.pre_encode_training_steps(delta)

    mismatches = sum(
        not torch.equal(
            seeded_once.quantizer.coding_model.state_dict()[key],
            seeded_twice.quantizer.coding_model.state_dict()[key],
        )
        for key in seeded_once.quantizer.coding_model.state_dict()
    )
    assert mismatches > 0, "a different training_seed must miss the cache and retrain differently."
