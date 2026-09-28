"""Compare staged quantizer indices with and without the oracle SW transport.

The quantizer is trained once. Both codec paths then receive the same inference
target, side information, quantizer instance, and deterministic staged indices.
The benchmark reports the prior estimate, LDPC syndrome rate, complete serialized
wire rates, residual index errors, and reconstruction distortion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import ceil, log2
from pathlib import Path
from typing import Any

import torch

from FL_code.FL_core.codec import Access, record_reconstruction_metrics
from FL_code.FL_core.utils import decompress_data_list, set_global_seed
from FL_code.cancer_protocol.bin_coder.oracle_sw_mixin import (
    OracleSWPayload,
    OracleSWRoundMixin,
    _wilson_upper,
)
from Experiments.slepian_wolf.oracle_index_mapping_mixin import (
    MappedOracleSWPayload,
    OracleValueOrderedIndexMixin,
)
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.round_codecs import Oracle_RoundCodec
from FL_code.cancer_protocol.bin_coder.sw_coder import SWCoderConfig, SlepianWolfCodec
from FL_code.cancer_protocol.wz_quantizer import (
    DedupedWZQuantizerCancer,
    PreprocessMetadata,
    WZcfgQuant,
)


@dataclass(frozen=True, slots=True)
class QuantizerSWResult:
    """One real-quantizer A/B result produced for inspection and JSON persistence."""

    bins_per_stage: int
    stages: int
    positions: int
    training_epochs: int
    block_length: int
    row_search_step: int
    bisect_search: bool
    search_rate_resolution: float
    target_ber: float
    value_ordered_mapping: bool
    original_to_label: tuple[tuple[int, ...], ...] | None
    source_index_rate: float
    inference_prior_rate: float
    inference_prior_stage_rates: tuple[float, ...]
    syndrome_rate: float
    syndrome_stage_rates: tuple[float, ...]
    syndrome_excess_over_prior: float
    direct_core_wire_rate: float
    plan_description_rate: float
    mapping_description_rate: float
    baseline_wire_rate: float
    sw_wire_rate: float
    integration_wire_delta: float
    sw_wire_saving: float
    bit_errors: int
    bit_error_rate: float
    bit_error_upper_95: float
    original_label_bit_error_rate: float
    direct_core_bit_errors: int
    direct_core_bit_error_rate: float
    direct_core_decode_disagreements: int
    symbol_errors: int
    symbol_error_rate: float
    baseline_mse: float
    sw_mse: float
    sw_mse_increase: float
    direct_core_payload_equal: bool
    direct_core_decode_equal: bool


class _SkipReferencePrior:
    """Keep the shared trained quantizer fixed across all benchmark transport paths."""

    _fine_tune_native_prior = False
    transport_prior_access = Access.SERVER_ONLY


class _BaselineCheckCodec(_SkipReferencePrior, Oracle_RoundCodec):
    """Ordinary staged-index payload used as the A/B wire-rate baseline."""


class _CaptureIndexBoundary:
    """Capture original indices crossing the experiment's codec boundary."""

    encoded_bins: torch.Tensor
    decoded_bins: torch.Tensor

    def encode_staged_indices(
        self,
        bins: torch.Tensor,
        metadata: PreprocessMetadata,
    ) -> OracleSWPayload:
        """Capture the exact integration-boundary symbols before compressing them."""
        self.encoded_bins = bins.detach().clone()
        return super().encode_staged_indices(bins, metadata)

    def decode_staged_indices(
        self,
        payload: OracleSWPayload,
        metadata: PreprocessMetadata,
    ) -> torch.Tensor:
        """Capture the symbols used by the integrated reconstruction path."""
        self.decoded_bins = super().decode_staged_indices(payload, metadata)
        return self.decoded_bins


class _OracleSWCheckCodec(
    _CaptureIndexBoundary,
    OracleSWRoundMixin,
    _SkipReferencePrior,
    Oracle_RoundCodec,
):
    """The same round codec with only its staged indices replaced by oracle SW data."""


class _MappedOracleSWCheckCodec(
    _CaptureIndexBoundary,
    OracleValueOrderedIndexMixin,
    OracleSWRoundMixin,
    _SkipReferencePrior,
    Oracle_RoundCodec,
):
    """The oracle SW codec with reconstruction-ordered Gray labeling added."""


def _codec(
    codec_class: type[Oracle_RoundCodec],
    quantizer: DedupedWZQuantizerCancer,
    config: WZcfgQuant,
    side_information: torch.Tensor,
    sw_config: SWCoderConfig | None = None,
) -> Oracle_RoundCodec:
    """Bind one already-trained quantizer to a baseline or SW experiment codec."""
    kwargs = {} if sw_config is None else {"sw_config": sw_config}
    codec = codec_class(
        {
            "bins_per_plane": config.bins_per_plane,
            "num_planes": config.num_planes,
            "sd_slices": config.norm_slices,
        },
        codec_class.__name__,
        **kwargs,
    )
    codec.c_cfg = config
    codec.quantizer = quantizer
    codec._send_quantizer_state = False
    codec.refresh_side_information([HistoryVector(side_information)])
    return codec


def run_quantizer_sw_comparison(
    *,
    bins_per_stage: int,
    stages: int,
    reconst_ld: float,
    positions: int,
    training_epochs: int,
    block_length: int,
    row_search_step: int,
    bisect_search: bool,
    value_ordered_mapping: bool,
    target_ber: float,
    seed: int,
) -> QuantizerSWResult:
    """Train once and compare baseline and oracle-SW payloads on identical inference data."""
    assert bins_per_stage >= 2 and bins_per_stage & (bins_per_stage - 1) == 0
    assert positions >= block_length
    assert row_search_step >= 1
    assert 0.0 < target_ber < 1.0
    set_global_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    noise_std = 0.1
    training_side_information = torch.randn(positions, generator=generator)
    training_target = training_side_information + torch.randn(
        positions, generator=generator
    ) * noise_std
    inference_side_information = torch.randn(positions, generator=generator)
    inference_target = inference_side_information + torch.randn(
        positions, generator=generator
    ) * noise_std

    quantizer_config = WZcfgQuant(
        bins_per_plane=bins_per_stage,
        num_planes=stages,
        norm_slices=(slice(None),),
        reconst_ld=reconst_ld,
        train_epochs=training_epochs,
        train_sample_size=min(200_000, positions),
        train_batch_size=min(1_000, positions),
        lr=3e-3,
        lr_step=40,
        tau=1.0,
        quantizer_train_repeats=1,
        prior_train_repeats=1,
        shared_heads=True,
        prior_context="hard",
        prior_probabilities="categorical",
        fused_optimizer=True,
        hidden_dim=100,
        layers=3,
    )
    quantizer = DedupedWZQuantizerCancer(quantizer_config, si_size=1)
    quantizer.train_model(training_target, [HistoryVector(training_side_information)])

    sw_config = SWCoderConfig(block_length=block_length, code_seed=seed)
    baseline = _codec(
        _BaselineCheckCodec,
        quantizer,
        quantizer_config,
        inference_side_information,
    )
    sw_codec_class = _MappedOracleSWCheckCodec if value_ordered_mapping else _OracleSWCheckCodec
    sw_codec = _codec(
        sw_codec_class,
        quantizer,
        quantizer_config,
        inference_side_information,
        sw_config,
    )
    assert isinstance(sw_codec, (_OracleSWCheckCodec, _MappedOracleSWCheckCodec))
    sw_codec.sw_search_step = row_search_step
    sw_codec.sw_search_bisect = bisect_search
    sw_codec.sw_target_ber = target_ber

    source_bins, _, source_metadata = quantizer.encoding_process(inference_target)
    baseline_record = baseline.create_r_record(0, 0)
    baseline_payload = baseline.encode(inference_target, baseline_record)
    baseline_content = decompress_data_list(baseline_payload)["payload_content"]
    baseline_bins, baseline_metadata = baseline_content
    torch.testing.assert_close(baseline_bins, source_bins)
    baseline_reconstruction = baseline.decode(baseline_payload, baseline_record)

    formatted_side_information = quantizer.side_info_tensor(source_metadata)
    inference_prior = PriorCalculator.compute_prior_from_network(
        quantizer.coding_model,
        source_bins,
        formatted_side_information,
    )
    prior_stage_rates = PriorCalculator.compute_plane_rates(
        inference_prior, source_bins, stages
    )

    sw_record = sw_codec.create_r_record(0, 0)
    sw_payload = sw_codec.encode(inference_target, sw_record)
    sw_content = decompress_data_list(sw_payload)["payload_content"]
    transmitted, sw_metadata = sw_content
    assert isinstance(transmitted, OracleSWPayload)
    assert isinstance(sw_metadata, PreprocessMetadata)
    if value_ordered_mapping:
        assert isinstance(transmitted, MappedOracleSWPayload)
        mapping = transmitted.mapping
        mapped_source_bins = mapping.map_indices(source_bins)
    else:
        assert not isinstance(transmitted, MappedOracleSWPayload)
        mapping = None
        mapped_source_bins = source_bins
    torch.testing.assert_close(sw_codec.encoded_bins, source_bins)
    sw_reconstruction = sw_codec.decode(sw_payload, sw_record)
    decoded_bins = sw_codec.decoded_bins

    direct_codec = SlepianWolfCodec(sw_config)
    direct_payload = direct_codec.encode_syndromes(mapped_source_bins, transmitted.plan)
    side_info, prior_provider = sw_codec._sw_context(sw_metadata)
    direct_decoded = direct_codec.decode_syndromes(
        direct_payload,
        transmitted.plan,
        prior_provider,
        side_info,
    ).symbols
    direct_payload_equal = direct_payload == transmitted.syndrome
    direct_original_bins = (
        direct_decoded if mapping is None else mapping.unmap_indices(direct_decoded)
    )
    direct_decode_equal = torch.equal(direct_original_bins, decoded_bins)
    assert direct_payload_equal, "integrated and direct-core syndrome bytes differ"

    bit_width = int(log2(bins_per_stage))
    mapped_decoded_bins = (
        decoded_bins if mapping is None else mapping.map_indices(decoded_bins)
    )
    bit_errors = sw_codec._bit_errors(mapped_source_bins, mapped_decoded_bins, bit_width)
    original_bit_errors = sw_codec._bit_errors(source_bins, decoded_bins, bit_width)
    source_bits = source_bins.numel() * bit_width
    direct_bit_errors = sw_codec._bit_errors(mapped_source_bins, direct_decoded, bit_width)
    direct_decode_disagreements = int((direct_original_bins != decoded_bins).sum())
    symbol_errors = int((source_bins.long() != decoded_bins.long()).sum())
    blocks = ceil(positions / block_length)
    syndrome_stage_rates = tuple(
        blocks * sum(rows) / positions for rows in transmitted.plan.syndrome_rows
    )
    syndrome_rate = sum(syndrome_stage_rates)
    direct_core_wire_rate = direct_payload.wire_bits / positions
    plan_description_rate = transmitted.plan.description_bits / positions
    baseline_metrics = record_reconstruction_metrics(
        inference_target, baseline_reconstruction
    )
    sw_metrics = record_reconstruction_metrics(inference_target, sw_reconstruction)
    baseline_wire_rate = len(baseline_payload) * 8 / positions
    sw_wire_rate = len(sw_payload) * 8 / positions

    return QuantizerSWResult(
        bins_per_stage=bins_per_stage,
        stages=stages,
        positions=positions,
        training_epochs=training_epochs,
        block_length=block_length,
        row_search_step=row_search_step,
        bisect_search=bisect_search,
        search_rate_resolution=stages * bit_width * row_search_step / block_length,
        target_ber=target_ber,
        value_ordered_mapping=value_ordered_mapping,
        original_to_label=None if mapping is None else mapping.original_to_label,
        source_index_rate=stages * bit_width,
        inference_prior_rate=sum(prior_stage_rates),
        inference_prior_stage_rates=prior_stage_rates,
        syndrome_rate=syndrome_rate,
        syndrome_stage_rates=syndrome_stage_rates,
        syndrome_excess_over_prior=syndrome_rate - sum(prior_stage_rates),
        direct_core_wire_rate=direct_core_wire_rate,
        plan_description_rate=plan_description_rate,
        mapping_description_rate=(
            0.0 if mapping is None else mapping.description_bits / positions
        ),
        baseline_wire_rate=baseline_wire_rate,
        sw_wire_rate=sw_wire_rate,
        integration_wire_delta=sw_wire_rate - direct_core_wire_rate,
        sw_wire_saving=1 - sw_wire_rate / baseline_wire_rate,
        bit_errors=bit_errors,
        bit_error_rate=bit_errors / source_bits,
        bit_error_upper_95=_wilson_upper(bit_errors, source_bits),
        original_label_bit_error_rate=original_bit_errors / source_bits,
        direct_core_bit_errors=direct_bit_errors,
        direct_core_bit_error_rate=direct_bit_errors / source_bits,
        direct_core_decode_disagreements=direct_decode_disagreements,
        symbol_errors=symbol_errors,
        symbol_error_rate=symbol_errors / source_bins.numel(),
        baseline_mse=baseline_metrics["mse"],
        sw_mse=sw_metrics["mse"],
        sw_mse_increase=sw_metrics["mse"] / baseline_metrics["mse"] - 1,
        direct_core_payload_equal=direct_payload_equal,
        direct_core_decode_equal=direct_decode_equal,
    )


def _print_result(result: QuantizerSWResult) -> None:
    """Print the compact rate, reliability, and distortion comparison."""
    print(
        f"{result.bins_per_stage} bins x {result.stages} stages, "
        f"positions={result.positions}, epochs={result.training_epochs}, "
        f"target_BER={result.target_ber:.4%}, "
        f"value_mapping={result.value_ordered_mapping} {result.original_to_label}"
    )
    print(
        f"prior={result.inference_prior_rate:.4f} syndrome={result.syndrome_rate:.4f} "
        f"excess={result.syndrome_excess_over_prior:+.4f} "
        f"search_resolution<={result.search_rate_resolution:.4f}"
    )
    print(
        f"direct_core_wire={result.direct_core_wire_rate:.4f} "
        f"integrated_wire={result.sw_wire_rate:.4f} "
        f"delta={result.integration_wire_delta:+.4f} plan_description={result.plan_description_rate:.4f}"
    )
    print(
        f"baseline_wire={result.baseline_wire_rate:.4f} sw_wire={result.sw_wire_rate:.4f} "
        f"saving={result.sw_wire_saving:.2%}"
    )
    print(
        f"BER={result.bit_error_rate:.4%} upper95={result.bit_error_upper_95:.4%} "
        f"original_label_BER={result.original_label_bit_error_rate:.4%} "
        f"direct_BER={result.direct_core_bit_error_rate:.4%} "
        f"decode_disagreements={result.direct_core_decode_disagreements} "
        f"symbol_error={result.symbol_error_rate:.4%}"
    )
    print(
        f"baseline_MSE={result.baseline_mse:.8f} sw_MSE={result.sw_mse:.8f} "
        f"increase={result.sw_mse_increase:+.2%} "
        f"direct_core_equal={result.direct_core_payload_equal and result.direct_core_decode_equal}"
    )


def _parse_args() -> argparse.Namespace:
    """Parse one reproducible quantizer/SW integration run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=2)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--reconst-ld", type=float, default=120.0)
    parser.add_argument("--positions", type=int, default=48_000)
    parser.add_argument("--training-epochs", type=int, default=60)
    parser.add_argument("--block-length", type=int, default=2_400)
    parser.add_argument("--row-search-step", type=int, default=100)
    parser.add_argument(
        "--bisect-search",
        action="store_true",
        help="probe the backoff grid by bisection: log-many trials, but it returns a point "
             "inside the ragged decoding threshold rather than its lower edge",
    )
    parser.add_argument("--value-ordered-mapping", action="store_true")
    parser.add_argument("--target-ber", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Experiments/slepian_wolf/results/quantizer_sw_integration.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run, print, and save one quantizer/SW A/B comparison."""
    args = _parse_args()
    result = run_quantizer_sw_comparison(
        bins_per_stage=args.bins,
        stages=args.stages,
        reconst_ld=args.reconst_ld,
        positions=args.positions,
        training_epochs=args.training_epochs,
        block_length=args.block_length,
        row_search_step=args.row_search_step,
        bisect_search=args.bisect_search,
        value_ordered_mapping=args.value_ordered_mapping,
        target_ber=args.target_ber,
        seed=args.seed,
    )
    _print_result(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2) + "\n")


if __name__ == "__main__":
    main()
