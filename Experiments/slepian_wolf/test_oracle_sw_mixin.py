"""Focused tests for the study-only value-ordered oracle index mapping."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from FL_code.FL_core.utils import compress_data_list
from FL_code.cancer_protocol.round_codecs import CancerRecord
from FL_code.cancer_protocol.bin_coder.oracle_sw_mixin import OracleSWRoundMixin
from FL_code.cancer_protocol.bin_coder.sw_coder import SWCoderConfig
from Experiments.slepian_wolf.oracle_index_mapping_mixin import (
    IndexMappingPlan,
    MappedOracleSWPayload,
    OracleValueOrderedIndexMixin,
)


def test_index_mapping_round_trip_and_prior_permutation() -> None:
    """Preserve symbols exactly while expressing priors in transmitted-label order."""
    mapping = IndexMappingPlan(original_to_label=((0, 1, 3, 2), (2, 3, 1, 0)))
    bins = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    torch.testing.assert_close(mapping.unmap_indices(mapping.map_indices(bins)), bins)

    prior = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    mapped = mapping.map_prior(prior, stage=0)
    torch.testing.assert_close(mapped, torch.tensor([[0.1, 0.2, 0.4, 0.3]]))


class _FakeQuantizer:
    def __init__(self, bins: torch.Tensor, side_info: torch.Tensor) -> None:
        self.source_bins = bins
        self.side_info = side_info
        self.bins_per_plane = 2
        self.num_planes = bins.shape[0]
        self.coding_model = object()
        self.prior_calculator = _UniformPrior()

    def side_info_tensor(self, metadata: object) -> torch.Tensor:
        return self.side_info

    def decoding_stages_process(self, payload_content: tuple[torch.Tensor, object]) -> torch.Tensor:
        return payload_content[0].float()


class _FakeIndexRoundCodec:
    def __init__(self, quantizer: _FakeQuantizer) -> None:
        self.quantizer = quantizer

    def get_encod_payload(self, delta_vec: torch.Tensor, record: object) -> dict[str, Any]:
        return {"payload_content": (self.quantizer.source_bins, {"metadata": True})}


class _UniformPrior:
    def compute_prior_from_network(
        self, model: object, decoded_stages: torch.Tensor, side_info: torch.Tensor
    ) -> torch.Tensor:
        return torch.full((decoded_stages.shape[0], side_info.shape[0], 2), 0.5)


class _MappedOracleHost(OracleValueOrderedIndexMixin, OracleSWRoundMixin, _FakeIndexRoundCodec):
    pass


def test_mapped_oracle_rate_includes_the_transmitted_mapping() -> None:
    """Charge the value-derived label permutation as part of the mapped coder payload."""
    bins = torch.tensor([[0, 1] * 2_400, [1, 0] * 2_400])
    side_info = torch.zeros((bins.shape[1], 1), dtype=torch.long)
    host = _MappedOracleHost(
        _FakeQuantizer(bins, side_info),
        sw_config=SWCoderConfig(block_length=32, max_bp_iterations=30),
    )
    record = CancerRecord(round_id=0, client_id=0, round_name_full="MOSW")
    outer = host.get_encod_payload(torch.zeros(bins.shape[1]), record)
    transmitted = outer["payload_content"][0]

    assert isinstance(transmitted, MappedOracleSWPayload)
    assert transmitted.wire_bits == (
        transmitted.syndrome.wire_bits
        + transmitted.plan.description_bits
        + transmitted.mapping.description_bits
    )
    assert record.coder_rate == pytest.approx(transmitted.wire_bits / bins.shape[1])
