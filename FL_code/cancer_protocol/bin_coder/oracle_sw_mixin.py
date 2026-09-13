"""Study-only Slepian--Wolf transport for codecs that produce staged indices."""

from __future__ import annotations

from math import log2, sqrt
from statistics import NormalDist
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator
import torch

from FL_code.FL_core.utils import decompress_data_list

from ..round_codecs import CancerRecord
from .sw_coder import ConditionalPrior, SWCoderConfig, SWPayload, SWRatePlan, SlepianWolfCodec
from ..wz_quantizer import PreprocessMetadata


class OracleSWPayload(BaseModel):
    """Self-contained SW data produced by the oracle mixin and consumed by its decoder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: SWRatePlan
    syndrome: SWPayload

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        """Require the transmitted syndrome to identify the accompanying rate plan."""
        assert self.syndrome.plan_id == self.plan.identifier, (
            f"syndrome plan {self.syndrome.plan_id} does not match "
            f"the transmitted plan {self.plan.identifier}."
        )
        return self

    @property
    def wire_bits(self) -> int:
        """Return the transmitted syndrome and complete rate-plan description size."""
        return self.syndrome.wire_bits + self.plan.description_bits


def _wilson_upper(errors: int, trials: int, confidence: float = 0.95) -> float:
    """Return the one-sided Wilson upper confidence bound for a bit-error rate."""
    assert 0 <= errors <= trials and trials > 0
    z = NormalDist().inv_cdf(confidence)
    estimate = errors / trials
    denominator = 1 + z**2 / trials
    center = estimate + z**2 / (2 * trials)
    radius = z * sqrt(estimate * (1 - estimate) / trials + z**2 / (4 * trials**2))
    return (center + radius) / denominator


class OracleSWRoundMixin:
    """Cheating staged-index SW transport mixed in before an index-producing round codec."""

    sw_target_ber = 0.01
    sw_confidence = 0.95
    sw_config: SWCoderConfig
    sw_search_step: int
    sw_search_bisect: bool
    sw_search_log: list[tuple[int, tuple[tuple[int, ...], ...], float, float]]

    def __init__(self, *args: Any, sw_config: SWCoderConfig | None = None,
                 sw_search_step: int = 1, sw_search_bisect: bool = False, **kwargs: Any) -> None:
        assert sw_search_step >= 1
        self.sw_config = SWCoderConfig() if sw_config is None else sw_config
        self.sw_search_step = sw_search_step
        self.sw_search_bisect = sw_search_bisect
        # One codec for the search, the payload, and the decoder: its parity matrices are the
        # expensive part, and the study runs both ends in this process anyway.
        self.sw_codec = SlepianWolfCodec(self.sw_config)
        super().__init__(*args, **kwargs)

    def coder_prior_model(self) -> Any:
        """Return the network the coder conditions on, and so the rate it is measured against."""
        return self.quantizer.coding_model

    def _sw_context(self, metadata: PreprocessMetadata) -> tuple[torch.Tensor, ConditionalPrior]:
        """Return decoder side information and the coder's causal stage prior."""
        side_info = self.quantizer.side_info_tensor(metadata)

        def prior_provider(
            decoded_stages: torch.Tensor, decoder_side_info: torch.Tensor, stage: int
        ) -> torch.Tensor:
            return self.quantizer.prior_calculator.compute_prior_from_network(
                self.coder_prior_model(), decoded_stages, decoder_side_info
            )[stage].to(decoder_side_info.device)

        return side_info, prior_provider

    def add_prior_to_record(self, *args: Any, **kwargs: Any) -> None:
        """Skip the separate diagnostic prior because this payload measures its own coder rate."""

    def _validate_staged_indices(self, bins: torch.Tensor) -> tuple[int, int]:
        """Return the staged alphabet and width after checking the current SW contract."""
        assert bins.ndim == 2 and bins.shape[0] == self.quantizer.num_planes, (
            f"expected staged indices with shape ({self.quantizer.num_planes}, positions), "
            f"got {tuple(bins.shape)}."
        )
        alphabet = self.quantizer.bins_per_plane
        assert alphabet >= 2 and alphabet & (alphabet - 1) == 0, (
            f"oracle SW requires a power-of-two stage alphabet; got {alphabet}."
        )
        assert bool(((bins >= 0) & (bins < alphabet)).all()), f"staged indices must lie in [0, {alphabet})."
        return alphabet, int(log2(alphabet))

    @staticmethod
    def _bit_errors(source: torch.Tensor, decoded: torch.Tensor, bit_width: int) -> int:
        """Count differing bits in equally shaped staged symbol tensors."""
        differences = source.long().cpu() ^ decoded.long().cpu()
        return sum(int(((differences >> bit) & 1).sum()) for bit in range(bit_width))

    def encode_staged_indices(self, bins: torch.Tensor, metadata: PreprocessMetadata) -> OracleSWPayload:
        """Cheat by trial-decoding the source and return the smallest reliable common-backoff plan."""
        alphabet, bit_width = self._validate_staged_indices(bins)
        side_info, prior_provider = self._sw_context(metadata)
        codec = self.sw_codec
        block_length = codec.cfg.block_length

        bootstrap = codec.initial_plan(bins.shape[0], alphabet)
        bootstrap_payload = codec.encode_syndromes(bins, bootstrap)
        base_plan = codec.decode_syndromes(bootstrap_payload, bootstrap, prior_provider, side_info).next_plan

        source_bits = bins.numel() * bit_width
        largest_backoff = max(
            block_length - rows for stage_rows in base_plan.syndrome_rows for rows in stage_rows
        )
        grid = [*range(0, largest_backoff + 1, self.sw_search_step)]
        if grid[-1] != largest_backoff:
            grid.append(largest_backoff)

        self.sw_search_log = []

        def trial(row_backoff: int) -> OracleSWPayload | None:
            plan = base_plan.model_copy(update={
                "syndrome_rows": tuple(
                    tuple(min(block_length, rows + row_backoff) for rows in stage_rows)
                    for stage_rows in base_plan.syndrome_rows
                )
            })
            syndrome = codec.encode_syndromes(bins, plan)
            decoded = codec.decode_syndromes(syndrome, plan, prior_provider, side_info).symbols
            errors = self._bit_errors(bins, decoded, bit_width)
            # What the scan tried and why it rejected it; the only way to tell a plan that is
            # genuinely unreachable from one the grid simply stepped over.
            self.sw_search_log.append(
                (row_backoff, plan.syndrome_rows, errors / source_bits,
                 _wilson_upper(errors, source_bits, self.sw_confidence)))
            if _wilson_upper(errors, source_bits, self.sw_confidence) >= self.sw_target_ber:
                return None
            return OracleSWPayload(plan=plan, syndrome=syndrome)

        # The scan returns the smallest reliable grid point, at one trial per point.  Bisection
        # costs log-many trials but is NOT equivalent: measured at step 1, the reliable backoffs
        # do not form a suffix -- the threshold is a ragged band ~10-30 rows wide in which
        # pass and fail alternate -- so bisection returns a point inside that band rather than
        # its lower edge, overshooting the scan by 0-31 rows (0 to 0.039 bit/position here).
        # That is a different, more conservative estimator, not a faster one; opt in knowingly.
        selected: OracleSWPayload | None = None
        if self.sw_search_bisect:
            low, high = 0, len(grid) - 1
            while low <= high:
                middle = (low + high) // 2
                candidate = trial(grid[middle])
                if candidate is None:
                    low = middle + 1
                else:
                    selected, high = candidate, middle - 1
        else:
            selected = next((found for found in map(trial, grid) if found is not None), None)

        assert selected is not None, (
            f"no common-backoff plan established BER < {self.sw_target_ber:.1%} at "
            f"{self.sw_confidence:.0%} confidence over {source_bits} source bits; "
            "the vector may be too short to certify even an error-free full-rate result."
        )
        return selected

    def decode_staged_indices(self, payload: OracleSWPayload, metadata: PreprocessMetadata) -> torch.Tensor:
        """Recover successive staged indices using the payload's complete rate plan."""
        assert payload.plan.block_length == self.sw_config.block_length, (
            f"payload block length {payload.plan.block_length} does not match "
            f"decoder block length {self.sw_config.block_length}."
        )
        assert payload.plan.num_stages == self.quantizer.num_planes
        assert payload.plan.alphabet == self.quantizer.bins_per_plane
        side_info, prior_provider = self._sw_context(metadata)
        return self.sw_codec.decode_syndromes(
            payload.syndrome, payload.plan, prior_provider, side_info
        ).symbols

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        """Replace the ordinary staged indices with their self-contained oracle SW payload."""
        outer_payload = super().get_encod_payload(delta_vec, record)
        bins, metadata = outer_payload["payload_content"]
        payload = self.encode_staged_indices(bins, metadata)
        blocks = (bins.shape[1] + payload.plan.block_length - 1) // payload.plan.block_length
        stage_rates = [
            blocks * sum(rows) / bins.shape[1] for rows in payload.plan.syndrome_rows
        ]
        wire_rate = payload.wire_bits / bins.shape[1]
        overhead_rate = wire_rate - sum(stage_rates)
        stage_rates[0] += overhead_rate
        decoded = self.decode_staged_indices(payload, metadata)
        record.update_record_fields(
            coder_rate=wire_rate,
            coder_stage_rates=stage_rates,
            coder_overhead_rate=overhead_rate,
            symbol_error_rate=float((decoded != bins.long().cpu()).sum() / bins.numel()),
        )
        outer_payload["payload_content"] = (payload, metadata)
        return outer_payload

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Recover staged indices and return the quantizer's normal final reconstruction."""
        transmitted, metadata = decompress_data_list(payload)["payload_content"]
        assert isinstance(transmitted, OracleSWPayload)
        bins = self.decode_staged_indices(transmitted, metadata)
        stage_reconstructions = self.quantizer.decoding_stages_process((bins, metadata))
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        return stage_reconstructions[-1]


__all__ = ["OracleSWPayload", "OracleSWRoundMixin"]
