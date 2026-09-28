"""Lossless rANS transport for round codecs that emit staged quantizer indices."""

from __future__ import annotations

from typing import Any

import torch

from FL_code.FL_core.codec import Access

from ..round_codecs import CancerRecord
from ..wz_quantizer import PreprocessMetadata
from .rans_coder import HEADER, RansConfig, decode, encode
from .staged_transport import StagedIndexTransportMixin


class RansRoundMixin(StagedIndexTransportMixin):
    """Replace staged indices with rANS coded against encoder-and-decoder-visible SI.

    ``rans_config=`` is the direct-construction path used by tests that build this mixin without
    a round-token protocol (bypassing ``options_to_config`` entirely). Whenever a round token is
    parsed instead, ``options_to_config`` is the single source of truth for ``lanes``: an explicit
    ``rans_lanes`` option always wins, and omitting it keeps whatever ``rans_config`` already held
    (``RansConfig()``'s default of 2048 unless the constructor was given a different one) rather
    than silently resetting it.
    """

    rans_config: RansConfig

    def __init__(self, *args: Any, rans_config: RansConfig | None = None, **kwargs: Any) -> None:
        self.rans_config = rans_config or RansConfig()
        super().__init__(*args, **kwargs)

    def options_to_config(self, rans_lanes: int | None = None, **options: Any) -> None:
        """Apply an explicit lane-count override; forward the rest to the quantizer options."""
        if rans_lanes is not None:
            self.rans_config = self.rans_config.model_copy(update={"lanes": rans_lanes})
        super().options_to_config(**options)

    @classmethod
    def validate_cfg(cls, rans_lanes: int | None = None, **options: Any) -> None:
        """Validate an explicit rANS lane count alongside the remaining quantizer options."""
        if rans_lanes is not None:
            RansConfig(lanes=rans_lanes)
        super().validate_cfg(**options)

    @classmethod
    def codec_provenance(cls, *, rans_lanes: int | None = None, **options: Any) -> dict[str, Any]:
        """Report the resolved rANS transport settings alongside the base quantizer facts."""
        facts = super().codec_provenance(**options)
        facts["transport"] = "rans"
        facts["resolved_transport_settings"] = {
            "rans_lanes": RansConfig().lanes if rans_lanes is None else rans_lanes,
            "rans_precision": RansConfig().precision,
        }
        facts["rate_fields_available"] = facts["rate_fields_available"] | {
            "temporal_si_rans_rate": cls.history_access is Access.TEMPORAL_TOO,
            "temporal_si_rans_stage_rates": cls.history_access is Access.TEMPORAL_TOO,
            "temporal_si_rans_overhead_rate": cls.history_access is Access.TEMPORAL_TOO,
            "server_si_rans_rate": True,
            "server_si_rans_stage_rates": True,
            "server_si_rans_overhead_rate": True,
        }
        return facts

    def _rans_prior(self, metadata: PreprocessMetadata, access: Access | None = None) -> Any:
        evaluation = self.prior_evaluations(metadata)[access or self.transport_prior_access]
        side_info = evaluation.side_information

        def prior(decoded_stages: torch.Tensor, stage: int) -> torch.Tensor:
            context = torch.zeros(
                (self.quantizer.num_planes, decoded_stages.shape[1]),
                dtype=decoded_stages.dtype,
                device=decoded_stages.device,
            )
            context[:decoded_stages.shape[0]] = decoded_stages
            return self.quantizer.prior_calculator.compute_prior_from_network(
                evaluation.model, context, side_info
            )[stage]

        return prior

    def encode_staged_indices(self, bins: torch.Tensor, metadata: PreprocessMetadata) -> bytes:
        """Encode every available SI-specific report and return the declared transport payload."""
        self._rans_payloads = {
            access: encode(bins, self._rans_prior(metadata, access), self.rans_config)
            for access in self.prior_evaluations(metadata)
        }
        self._rans_metrics = {
            access: self._rans_rate_fields(bins, metadata, payload, access)
            for access, payload in self._rans_payloads.items()
        }
        return self._rans_payloads[self.transport_prior_access]

    def decode_staged_indices(self, payload: bytes, metadata: PreprocessMetadata) -> torch.Tensor:
        """Losslessly recover all stages against the decoder's matching prior."""
        assert isinstance(payload, bytes)
        return decode(payload, self._rans_prior(metadata, self.transport_prior_access))

    def _rans_rate_fields(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: bytes, access: Access,
    ) -> tuple[float, list[float], float]:
        """Return total, incremental-stage, and included-overhead rates for one SI source."""
        overhead_bits = 8 * (HEADER.size + 4 * self.rans_config.lanes)
        prior = self._rans_prior(metadata, access)
        prefix_bits = [
            8 * len(encode(bins[:stage], prior, self.rans_config)) - overhead_bits
            for stage in range(1, bins.shape[0])
        ] + [8 * len(payload) - overhead_bits]
        stage_bits = [
            current - previous
            for current, previous in zip(prefix_bits, (0, *prefix_bits[:-1]), strict=True)
        ]
        positions = bins.shape[1]
        stage_rates = [
            (bits + (overhead_bits if stage == 0 else 0)) / positions
            for stage, bits in enumerate(stage_bits)
        ]
        return sum(stage_rates), stage_rates, overhead_bits / positions

    def payload_bits(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: bytes
    ) -> tuple[list[int], int]:
        """Separate each stage's stream growth from the fixed header and lane flush."""
        total, stage_rates, overhead_rate = self._rans_metrics[self.transport_prior_access]
        positions = bins.shape[1]
        overhead_bits = round(overhead_rate * positions)
        stage_bits = [round(rate * positions) for rate in stage_rates]
        stage_bits[0] -= overhead_bits
        assert abs(total - sum(stage_rates)) < 1e-9
        return stage_bits, overhead_bits

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        """Add separate temporal- and server-SI rANS measurements to the selected payload."""
        payload = super().get_encod_payload(delta_vec, record)
        updates: dict[str, Any] = {}
        for access, (rate, stage_rates, overhead_rate) in self._rans_metrics.items():
            prefix = "temporal_si" if access is Access.TEMPORAL_TOO else "server_si"
            updates |= {
                f"{prefix}_rans_rate": rate,
                f"{prefix}_rans_stage_rates": stage_rates,
                f"{prefix}_rans_overhead_rate": overhead_rate,
            }
        # Server-SI rANS on T/TR is an oracle comparison: the temporal encoder lacks this SI.
        record.update_record_fields(**updates)
        return payload

    def symbol_error_rate(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: bytes
    ) -> float:
        """Return zero because rANS decoding verifies the complete symbol digest."""
        return 0.0


__all__ = ["RansRoundMixin"]
