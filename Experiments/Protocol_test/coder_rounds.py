"""Sparse-test round codecs for matched rANS and oracle Slepian--Wolf comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Experiments.Protocol_test.sparse_replay import ReplayCase
from FL_code.FL_core.utils import StateDictManager, parse_configurable_name
from FL_code.cancer_protocol.bin_coder.oracle_sw_mixin import OracleSWRoundMixin
from FL_code.cancer_protocol.bin_coder.rans_coder import RansConfig
from FL_code.cancer_protocol.bin_coder.rans_mixin import RansRoundMixin
from FL_code.cancer_protocol.bin_coder.sw_coder import SWCoderConfig
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol, _WZProtocol
from FL_code.cancer_protocol.optimized_history import OptimizedGradientHistory
from FL_code.cancer_protocol.round_codecs import Oracle_RoundCodec, T_RoundCodec, _WZRoundCodec


class TRansRoundCodec(RansRoundMixin, T_RoundCodec):
    """Temporal WZ quantization followed by lossless conditional rANS."""

    round_name = "TR"


class OracleSWRoundCodec(OracleSWRoundMixin, Oracle_RoundCodec):
    """Oracle WZ quantization followed by the study-only oracle SW transport."""

    round_name = "OSW"


class _RetrainedPriorCoder:
    """Code against the retrained diagnostic prior rather than the quantizer's own network.

    `quantizer.coding_model` is trained alongside the quantizer and so is handicapped by that
    objective, while the retrained prior is the rate the RD curves are read against.  Handing it
    to the decoder is a further cheat -- it is fitted on the very bins it scores -- so these
    variants live beside the shipped ones rather than replacing them.
    """

    _retrained_prior: EncoderDecoderLayeredRNN

    def conditional_code_lengths(self, *args: Any, **kwargs: Any) -> tuple[Any, EncoderDecoderLayeredRNN]:
        """Keep the prior the diagnostic just fitted, for the coder to condition on."""
        code_lengths, self._retrained_prior = super().conditional_code_lengths(*args, **kwargs)
        return code_lengths, self._retrained_prior

    def coder_prior_model(self) -> EncoderDecoderLayeredRNN:
        return self._retrained_prior


class OracleSWRetrainedPriorRoundCodec(_RetrainedPriorCoder, OracleSWRoundMixin, Oracle_RoundCodec):
    """Oracle SW transport conditioned on the retrained prior."""

    round_name = "OSWR"
    # The SW mixin suppresses the diagnostic prior; here it is the thing being coded against.
    add_prior_to_record = Oracle_RoundCodec.add_prior_to_record


class TRansRetrainedPriorRoundCodec(_RetrainedPriorCoder, RansRoundMixin, T_RoundCodec):
    """Conditional rANS coded against the retrained prior."""

    round_name = "TRR"
    add_prior_to_record = T_RoundCodec.add_prior_to_record


class _OptimizedCoderProtocol(WZCancerProtocol):
    """Experiment protocol that supplies shared SW matrices to its composed codec."""

    sw_search_step = 120
    # The rANS flush costs 4 bytes per lane whatever the vector length, so the default 2048 is
    # 0.077 bit/coordinate on a 0.86M-parameter model against 0.006 on an 11.2M one.  Sized here
    # so the coder is not charged for the model being small.
    rans_lanes = 2048

    def codec_protocol_inputs(
        self, round_codec_class: type[_WZRoundCodec], previous: _WZRoundCodec | None
    ) -> dict[str, Any]:
        inputs = super().codec_protocol_inputs(round_codec_class, previous)
        if issubclass(round_codec_class, RansRoundMixin):
            inputs["rans_config"] = RansConfig(lanes=self.rans_lanes)
        if issubclass(round_codec_class, OracleSWRoundMixin):
            assert self.cache_dir is not None
            inputs["sw_config"] = SWCoderConfig(
                matrix_cache_dir=self.cache_dir.parent / "sw_matrices"
            )
            inputs["sw_search_step"] = self.sw_search_step
        return inputs

    def config_snapshot(self) -> dict[str, Any]:
        """Record transport settings supplied outside the round-config string."""
        return super().config_snapshot() | {
            "rans": {"lanes": self.rans_lanes, "precision": 16},
            "oracle_sw": {
                "block_length": 2400,
                "search_step": self.sw_search_step,
                "target_ber": 0.01,
                "confidence": 0.95,
            },
        }


def optimized_coder_protocol_factory(
    case: ReplayCase,
    state_manager: StateDictManager,
    cache_dir: Path,
    cache_overwrite: bool,
) -> _WZProtocol:
    """Build one matched codec case with shared quantizer caching and packed RAM history."""
    family = {"T_rans": "T", "T_rans_retrained": "T",
              "O_oracle_sw": "O", "O_oracle_sw_retrained": "O"}.get(case.codec_id, case.codec_id)
    parsed = parse_configurable_name(case.protocol, "protocol")
    protocol = _OptimizedCoderProtocol(
        parsed.options,
        case.protocol,
        state_manager.get_slices(),
        cache_dir.with_name(family),
        cache_overwrite,
    )
    # A case may straddle the warmup/routine boundary: a trained round sits at the end of warmup
    # and the frozen rounds that reuse it come from routine, so split rather than assert.
    warmup = protocol.warmup_round_codecs
    split = max(0, min(len(case.round_codecs), len(warmup) - case.start_round))
    head, tail = case.round_codecs[:split], case.round_codecs[split:]
    if head:
        protocol.warmup_round_codecs = (
            warmup[:case.start_round] + head + warmup[case.start_round + split:]
        )
    if tail:
        phase = (case.start_round + split - len(warmup)) % len(tail)
        protocol.routine_round_codecs = tail[-phase:] + tail[:-phase] if phase else tail
    protocol._round_codec_classes = protocol._validate_round_codecs()
    protocol._recons_history = OptimizedGradientHistory(
        protocol.max_per_client_recons_history, state_manager
    )
    protocol.histories = (protocol._recons_history,)
    return protocol


def fine_search_coder_protocol_factory(
    case: ReplayCase,
    state_manager: StateDictManager,
    cache_dir: Path,
    cache_overwrite: bool,
) -> _WZProtocol:
    """Small-model factory with a fine SW row search.

    At step 120 the scan can step straight over the last viable rate and clip the plane to
    `block_length`, which ships it uncompressed; step 15 costs 8x the trials to find out whether
    any sub-ceiling rate exists at all.
    """
    protocol = small_model_coder_protocol_factory(case, state_manager, cache_dir, cache_overwrite)
    protocol.sw_search_step = 15
    return protocol


def small_model_coder_protocol_factory(
    case: ReplayCase,
    state_manager: StateDictManager,
    cache_dir: Path,
    cache_overwrite: bool,
) -> _WZProtocol:
    """As above, with the rANS lane count sized for a sub-megabyte parameter vector.

    At 0.86M coordinates the default 2048 lanes flush 0.077 bit/coordinate, which is 8% of a
    one-bit rate; 128 lanes bring it to 0.005, the same share the default costs at 11.2M.
    """
    protocol = optimized_coder_protocol_factory(case, state_manager, cache_dir, cache_overwrite)
    protocol.rans_lanes = 128
    return protocol


__all__ = ["OracleSWRetrainedPriorRoundCodec", "OracleSWRoundCodec", "TRansRetrainedPriorRoundCodec",
           "TRansRoundCodec", "fine_search_coder_protocol_factory", "optimized_coder_protocol_factory",
           "small_model_coder_protocol_factory"]
