"""Shared round-codec plumbing for transports that replace staged WZ indices."""

from __future__ import annotations

from math import isclose
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access
from FL_code.FL_core.utils import decompress_data_list

from ..round_codecs import CancerRecord
from ..wz_quantizer import PreprocessMetadata


class StagedIndexTransportMixin:
    """Encode, account for, and decode the staged indices emitted by a WZ round codec."""

    transport_prior_access: ClassVar[Access]

    def encode_staged_indices(self, bins: torch.Tensor, metadata: PreprocessMetadata) -> Any:
        """Return the transport payload for a staged symbol tensor."""
        raise NotImplementedError

    def decode_staged_indices(self, payload: Any, metadata: PreprocessMetadata) -> torch.Tensor:
        """Recover staged symbols using decoder-available state."""
        raise NotImplementedError

    def payload_bits(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: Any
    ) -> tuple[list[int], int]:
        """Return the coded bits attributed to each stage and fixed payload overhead bits."""
        raise NotImplementedError

    def symbol_error_rate(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: Any
    ) -> float:
        """Return the encoder-known symbol error rate for this transport payload."""
        raise NotImplementedError

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        """Replace plain staged indices with a transport payload and record wire metrics."""
        outer_payload = super().get_encod_payload(delta_vec, record)
        bins, metadata = outer_payload["payload_content"]
        payload = self.encode_staged_indices(bins, metadata)
        stage_bits, overhead_bits = self.payload_bits(bins, metadata, payload)
        assert len(stage_bits) == bins.shape[0]
        positions = bins.shape[1]
        # Fixed per-payload overhead (header, lane flush, or transmitted rate plan) is charged
        # entirely to stage 0 rather than spread across stages, so coder_stage_rates always sums
        # exactly to coder_rate and coder_overhead_rate names an already-included component of it,
        # not an extra charge.
        coder_rate = (sum(stage_bits) + overhead_bits) / positions
        coder_stage_rates = [
            (bits + (overhead_bits if stage == 0 else 0)) / positions
            for stage, bits in enumerate(stage_bits)
        ]
        assert isclose(sum(coder_stage_rates), coder_rate, rel_tol=1e-9), (
            f"{self.round_name_full}: coder_stage_rates must sum to coder_rate exactly.")
        record.update_record_fields(
            coder_rate=coder_rate,
            coder_stage_rates=coder_stage_rates,
            coder_overhead_rate=overhead_bits / positions,
            symbol_error_rate=self.symbol_error_rate(bins, metadata, payload),
        )
        outer_payload["payload_content"] = (payload, metadata)
        return outer_payload

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Recover staged indices and return the quantizer's final reconstruction."""
        decoded_payload = decompress_data_list(payload)
        transmitted, metadata = decoded_payload["payload_content"]
        bins = self.decode_staged_indices(transmitted, metadata)
        stage_reconstructions = self.quantizer.decoding_stages_process((bins, metadata))
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        return stage_reconstructions[-1]

    @classmethod
    def codec_provenance(cls, **options: Any) -> dict[str, Any]:
        """Layer coder-prior facts and rate-field availability onto the base quantizer facts."""
        facts = super().codec_provenance(**options)
        facts["transport_prior_si"] = cls.transport_prior_access.value
        facts["rate_fields_available"] = facts["rate_fields_available"] | {
            "coder_rate": True,
            "coder_stage_rates": True,
            "coder_overhead_rate": True,
            "symbol_error_rate": True,
        }
        return facts


__all__ = ["StagedIndexTransportMixin"]
