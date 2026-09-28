"""Study-only Slepian--Wolf transport for codecs that produce staged indices."""

from __future__ import annotations

from math import log2, sqrt
from statistics import NormalDist
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator
import torch

from .staged_transport import StagedIndexTransportMixin
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


class OracleSWRoundMixin(StagedIndexTransportMixin):
    """Cheating staged-index SW transport mixed in before an index-producing round codec.

    ``sw_config=``/``sw_search_step=`` constructor kwargs are the direct-construction path used
    by tests that build this mixin without a round-token protocol (bypassing ``options_to_config``
    entirely). Whenever a round token is parsed instead, ``options_to_config`` is the single
    source of truth: an explicit ``sw_search_step`` option always wins, and omitting it keeps
    whatever the constructor already set rather than silently resetting it to 1.
    """

    sw_target_ber = 0.01
    sw_confidence = 0.95
    sw_config: SWCoderConfig
    sw_search_step: int = 1
    sw_search_bisect: bool = False
    _encoded_symbol_error_rate: float

    def __init__(self, *args: Any, sw_config: SWCoderConfig | None = None,
                 sw_search_step: int = 1, sw_search_bisect: bool = False, **kwargs: Any) -> None:
        assert sw_search_step >= 1
        self.sw_config = SWCoderConfig() if sw_config is None else sw_config
        self.sw_search_step = sw_search_step
        self.sw_search_bisect = sw_search_bisect
        super().__init__(*args, **kwargs)
        # One codec owns the deterministic matrices used by search, payload production, and decode.
        self.sw_codec = SlepianWolfCodec(self.sw_config)

    def options_to_config(self, sw_search_step: int | None = None, **options: Any) -> None:
        """Apply an explicit row-search-step override before forwarding quantizer/cache options."""
        if sw_search_step is not None:
            self.sw_search_step = sw_search_step
        if (cache_dir := options.get("cache_dir")) is not None:
            self.sw_config = self.sw_config.model_copy(
                update={"matrix_cache_dir": cache_dir / "sw_matrices"}
            )
        super().options_to_config(**options)

    @classmethod
    def validate_cfg(cls, sw_search_step: int | None = None, **options: Any) -> None:
        """Validate an explicit row-search step alongside the remaining quantizer options."""
        assert sw_search_step is None or (isinstance(sw_search_step, int) and sw_search_step >= 1), (
            f"sw_search_step must be a positive integer; got {sw_search_step!r}."
        )
        super().validate_cfg(**options)

    @classmethod
    def codec_provenance(cls, *, sw_search_step: int | None = None, **options: Any) -> dict[str, Any]:
        """Report the oracle SW search and its resolved settings alongside the base coder facts."""
        facts = super().codec_provenance(**options)
        facts["sw_oracle_search"] = True
        facts["transport"] = "oracle_sw"
        facts["resolved_transport_settings"] = {
            "sw_search_step": cls.sw_search_step if sw_search_step is None else sw_search_step,
            "sw_search_bisect": cls.sw_search_bisect,
            "sw_target_ber": cls.sw_target_ber,
            "sw_confidence": cls.sw_confidence,
            "sw_block_length": SWCoderConfig().block_length,
        }
        return facts

    def _sw_context(self, metadata: PreprocessMetadata) -> tuple[torch.Tensor, ConditionalPrior]:
        """Return decoder side information and the coder's causal stage prior."""
        evaluation = self.prior_evaluations(metadata)[self.transport_prior_access]
        side_info = evaluation.side_information

        def prior_provider(
            decoded_stages: torch.Tensor, decoder_side_info: torch.Tensor, stage: int
        ) -> torch.Tensor:
            return self.quantizer.prior_calculator.compute_prior_from_network(
                evaluation.model, decoded_stages, decoder_side_info
            )[stage].to(decoder_side_info.device)

        return side_info, prior_provider

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

        def trial(row_backoff: int) -> tuple[OracleSWPayload, float] | None:
            plan = base_plan.model_copy(update={
                "syndrome_rows": tuple(
                    tuple(min(block_length, rows + row_backoff) for rows in stage_rows)
                    for stage_rows in base_plan.syndrome_rows
                )
            })
            syndrome = codec.encode_syndromes(bins, plan)
            decoded = codec.decode_syndromes(syndrome, plan, prior_provider, side_info).symbols
            errors = self._bit_errors(bins, decoded, bit_width)
            if _wilson_upper(errors, source_bits, self.sw_confidence) >= self.sw_target_ber:
                return None
            return OracleSWPayload(plan=plan, syndrome=syndrome), float((decoded != bins).float().mean())

        # The scan returns the smallest reliable grid point, at one trial per point.  Bisection
        # costs log-many trials but is NOT equivalent: measured at step 1, the reliable backoffs
        # do not form a suffix -- the threshold is a ragged band ~10-30 rows wide in which
        # pass and fail alternate -- so bisection returns a point inside that band rather than
        # its lower edge, overshooting the scan by 0-31 rows (0 to 0.039 bit/position here).
        # That is a different, more conservative estimator, not a faster one; opt in knowingly.
        selected: tuple[OracleSWPayload, float] | None = None
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
        payload, self._encoded_symbol_error_rate = selected
        return payload

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

    def payload_bits(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: OracleSWPayload
    ) -> tuple[list[int], int]:
        """Separate stage syndromes from the transmitted rate-plan and framing overhead."""
        blocks = (bins.shape[1] + payload.plan.block_length - 1) // payload.plan.block_length
        stage_bits = [blocks * sum(rows) for rows in payload.plan.syndrome_rows]
        return stage_bits, payload.wire_bits - sum(stage_bits)

    def symbol_error_rate(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: OracleSWPayload
    ) -> float:
        """Return the error rate already observed in the selected oracle-search trial."""
        return self._encoded_symbol_error_rate


__all__ = ["OracleSWPayload", "OracleSWRoundMixin"]
