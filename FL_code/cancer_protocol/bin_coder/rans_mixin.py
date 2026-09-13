"""Lossless rANS transport for round codecs that emit staged quantizer indices."""

from __future__ import annotations

from typing import Any

import torch

from FL_code.FL_core.utils import decompress_data_list

from ..round_codecs import CancerRecord
from ..wz_quantizer import PreprocessMetadata
from .rans_coder import RansConfig, decode, encode


class RansRoundMixin:
    """Replace staged indices with rANS coded against encoder-and-decoder-visible SI."""

    rans_config: RansConfig

    def __init__(self, *args: Any, rans_config: RansConfig | None = None, **kwargs: Any) -> None:
        self.rans_config = rans_config or RansConfig()
        super().__init__(*args, **kwargs)

    def coder_prior_model(self) -> Any:
        """Return the network the coder conditions on, and so the rate it is measured against."""
        return self.quantizer.coding_model

    def _rans_prior(self, metadata: PreprocessMetadata) -> Any:
        side_info = self.quantizer.side_info_tensor(metadata)

        def prior(decoded_stages: torch.Tensor, stage: int) -> torch.Tensor:
            context = torch.zeros(
                (self.quantizer.num_planes, decoded_stages.shape[1]),
                dtype=decoded_stages.dtype,
                device=decoded_stages.device,
            )
            context[:decoded_stages.shape[0]] = decoded_stages
            return self.quantizer.prior_calculator.compute_prior_from_network(
                self.coder_prior_model(), context, side_info
            )[stage]

        return prior

    def add_prior_to_record(self, *args: Any, **kwargs: Any) -> None:
        """Skip the separate diagnostic prior because this payload measures its own coder rate."""

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        """Replace ordinary staged indices with their lossless conditional rANS stream."""
        outer_payload = super().get_encod_payload(delta_vec, record)
        bins, metadata = outer_payload["payload_content"]
        prior = self._rans_prior(metadata)
        prefix_rates = [
            len(payload := encode(bins[:stage], prior, self.rans_config)) * 8 / bins.shape[1]
            for stage in range(1, bins.shape[0] + 1)
        ]
        stage_rates = [
            rate - previous for rate, previous in zip(prefix_rates, (0.0, *prefix_rates[:-1]), strict=True)
        ]
        wire_rate = prefix_rates[-1]
        record.update_record_fields(
            coder_rate=wire_rate,
            coder_stage_rates=stage_rates,
            coder_overhead_rate=0.0,
            symbol_error_rate=0.0,
        )
        outer_payload["payload_content"] = (payload, metadata)
        return outer_payload

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Recover the staged indices and return the quantizer's normal reconstruction."""
        transmitted, metadata = decompress_data_list(payload)["payload_content"]
        assert isinstance(transmitted, bytes)
        bins = decode(transmitted, self._rans_prior(metadata))
        stage_reconstructions = self.quantizer.decoding_stages_process((bins, metadata))
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        return stage_reconstructions[-1]


__all__ = ["RansRoundMixin"]
