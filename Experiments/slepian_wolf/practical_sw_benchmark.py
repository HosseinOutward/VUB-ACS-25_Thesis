"""Measure what practical Slepian-Wolf coding spends against the prior rate it aims at.

One quantizer is trained on a synthetic Gaussian source whose conditional prior the quantizer
itself supplies, so the ideal rate of the entropy coder that follows it is known.  The same
inference vector is then coded five ways -- plain indices, the cheating oracle search, the
one-shot feedback controller, the pilot-calibrated one and the rate-adaptive LDPCA ladder -- and
every refinement stage is reported as its own rate/distortion node.

The point of the run is the decomposition, not any single rate:

* the prior cross-entropy is the floor a conditional entropy coder would reach;
* the oracle's excess over it is the finite-length LDPC gap, which no rate controller can remove;
* what the practical variants spend on top of the oracle is the price of not being allowed to
  look at the vector being sent;
* and the ladder's excess over the fixed-rate variants is what the *code family* costs, since it
  is the one method that pays per block instead of per plane.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from dataclasses import asdict, dataclass
import json
from math import ceil, log2
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.codec import record_reconstruction_metrics
from FL_code.FL_core.utils import decompress_data_list, set_global_seed
from FL_code.cancer_protocol.bin_coder.oracle_sw_mixin import OracleSWPayload, OracleSWRoundMixin
from Experiments.slepian_wolf.practical_sw_mixin import (
    LDPCAConfig,
    LDPCASWPayload,
    LDPCASWRoundMixin,
    PracticalSWConfig,
    PracticalSWPayload,
    PracticalSWRoundMixin,
    plan_row_bits,
)
from FL_code.cancer_protocol.bin_coder.sw_coder import LDPCA_BACKEND, SWCoderConfig
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.round_codecs import Oracle_RoundCodec
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer, WZcfgQuant
from FL_code.cancer_protocol.history import HistoryVector

NOISE_POWER = 0.01


@dataclass(frozen=True, slots=True)
class MethodResult:
    """One coding method's rate/distortion nodes and reliability on the shared inference vector."""

    name: str
    stage_rates: tuple[float, ...]
    overhead_rate: float
    downlink_rate: float
    stage_mse: tuple[float, ...]
    stage_wmape: tuple[float, ...]
    symbol_errors: int
    repaired_blocks: int
    seconds: float
    controller: dict[str, Any] | None

    @property
    def total_rate(self) -> float:
        return sum(self.stage_rates) + self.overhead_rate


class _SkipReferencePrior:
    """Avoid retraining the diagnostic prior four times; the shared quantizer prior is measured once."""

    def add_prior_to_record(self, *args: Any, **kwargs: Any) -> None:
        """Leave reference-prior fields empty."""


class _CaptureBins:
    """Keep the staged indices that crossed the codec boundary so the caller need not decode twice."""

    decoded_bins: torch.Tensor

    def decode_staged_indices(self, payload: Any, metadata: Any) -> torch.Tensor:
        """Capture the symbols the integrated reconstruction path is about to use."""
        self.decoded_bins = super().decode_staged_indices(payload, metadata)
        return self.decoded_bins


class _BaselineCodec(_SkipReferencePrior, Oracle_RoundCodec):
    """Ordinary staged-index payload, the distortion every SW variant is compared against."""


class _OracleSWCodec(_CaptureBins, OracleSWRoundMixin, _SkipReferencePrior, Oracle_RoundCodec):
    """The cheating search: rate chosen by trial-decoding the vector it is about to send."""


class _PracticalSWCodec(_CaptureBins, PracticalSWRoundMixin, _SkipReferencePrior, Oracle_RoundCodec):
    """The feedback controller: rate chosen at the decoder, from data or from the prior alone."""


class _LDPCASWCodec(_CaptureBins, LDPCASWRoundMixin, _SkipReferencePrior, Oracle_RoundCodec):
    """The rate-adaptive ladder: every block climbs until its own hash verifies."""


def _codec(codec_class: type[Oracle_RoundCodec], quantizer: DedupedWZQuantizerCancer,
           config: WZcfgQuant, side_information: torch.Tensor, **kwargs: Any) -> Oracle_RoundCodec:
    """Bind one already trained quantizer to an experiment codec."""
    codec = codec_class(
        {"bins_per_plane": config.bins_per_plane, "num_planes": config.num_planes,
         "sd_slices": config.norm_slices},
        codec_class.__name__,
        **kwargs,
    )
    codec.c_cfg = config
    codec.quantizer = quantizer
    codec._send_quantizer_state = False
    codec.refresh_side_information([HistoryVector(side_information)])
    return codec


def _stage_metrics(target: torch.Tensor, reconstructions: torch.Tensor) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return each refinement stage's MSE and weighted mean absolute percentage error."""
    metrics = [record_reconstruction_metrics(target, stage) for stage in reconstructions]
    return tuple(m["mse"] for m in metrics), tuple(m["wmape"] for m in metrics)


# One accounting function per transport: staged uplink bits, total uplink, downlink, repairs.
Accounting = Callable[[Any], tuple[tuple[int, ...], int, int, int]]


def _oracle_accounting(payload: OracleSWPayload) -> tuple[tuple[int, ...], int, int, int]:
    """Account for the oracle transport, whose plan is chosen by the encoder and travels up."""
    blocks = ceil(payload.syndrome.positions / payload.plan.block_length)
    stage_bits = tuple(blocks * sum(rows) for rows in payload.plan.syndrome_rows)
    return stage_bits, payload.syndrome.wire_bits + plan_row_bits(payload.plan), 0, 0


def _practical_accounting(payload: PracticalSWPayload) -> tuple[tuple[int, ...], int, int, int]:
    """Account for the feedback transport: syndromes, hashes and repairs up, plans and requests down."""
    bits = [0] * payload.bulk_plan.num_stages
    for plan, part, repair in payload.parts:
        blocks = ceil(part.positions / plan.block_length)
        for stage, rows in enumerate(plan.syndrome_rows):
            bits[stage] += blocks * (sum(rows) + plan.bit_width * repair.hash_bits)
        for stage, _, repaired in repair.requests:
            bits[stage] += len(repaired) * plan.block_length
    return tuple(bits), payload.uplink_bits, payload.downlink_bits, payload.repaired_blocks


def _ldpca_accounting(payload: LDPCASWPayload) -> tuple[tuple[int, ...], int, int, int]:
    """Account for the ladder: rungs and hashes up, the starting plan and one request bit down."""
    return (
        payload.stage_uplink_bits,
        payload.uplink_bits,
        payload.downlink_bits,
        len(payload.ladder.raw),
    )


def _run_method(
    name: str,
    codec: Oracle_RoundCodec,
    target: torch.Tensor,
    source_bins: torch.Tensor,
    alphabet: int,
    accounting: Accounting | None,
) -> MethodResult:
    """Encode and decode one vector with one codec and collect its rate, distortion and errors.

    ``accounting`` is ``None`` for the plain-index codec, whose staged symbols are the payload and
    therefore cost the alphabet's width per stage with nothing to attribute.
    """
    start = perf_counter()
    record = codec.create_r_record(0, 0)
    payload = codec.encode(target, record)
    reconstruction = codec.decode(payload, record)
    seconds = perf_counter() - start

    positions = target.numel()
    content, metadata = decompress_data_list(payload)["payload_content"]
    if accounting is None:
        decoded_bins = content
        stage_bits = (int(positions * log2(alphabet)),) * source_bins.shape[0]
        uplink, downlink, repaired = sum(stage_bits), 0, 0
    else:
        decoded_bins = codec.decoded_bins
        stage_bits, uplink, downlink, repaired = accounting(content)
    stages = codec.quantizer.decoding_stages_process((decoded_bins, metadata))
    torch.testing.assert_close(reconstruction, stages[-1], rtol=0, atol=0)
    mse, wmape = _stage_metrics(target, stages)

    controller = {field: value for field, value in vars(record).items() if field.startswith("sw_")}
    return MethodResult(
        name=name,
        stage_rates=tuple(bits / positions for bits in stage_bits),
        overhead_rate=(uplink - sum(stage_bits)) / positions,
        downlink_rate=downlink / positions,
        stage_mse=mse,
        stage_wmape=wmape,
        symbol_errors=int((decoded_bins.long() != source_bins.long()).sum()),
        repaired_blocks=repaired,
        seconds=seconds,
        controller=controller or None,
    )


def run_comparison(
    *,
    bins_per_stage: int,
    stages: int,
    reconst_ld: float,
    positions: int,
    training_epochs: int,
    block_length: int,
    pilot_blocks: int,
    bootstrap_margin: float,
    safety_rows: int,
    verification_bits: int,
    ladder_step: int,
    ladder_start_margin: float,
    ladder_verification_bits: int,
    oracle_target_ber: float,
    oracle_row_step: int,
    seed: int,
    cache_dir: Path | None,
) -> dict[str, Any]:
    """Train one quantizer and code one inference vector with every method under study."""
    assert bins_per_stage >= 2 and bins_per_stage & (bins_per_stage - 1) == 0
    set_global_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    noise_std = NOISE_POWER**0.5

    def source() -> tuple[torch.Tensor, torch.Tensor]:
        side_information = torch.randn(positions, generator=generator)
        return side_information + torch.randn(positions, generator=generator) * noise_std, side_information

    training_target, training_side_information = source()
    inference_target, inference_side_information = source()

    quantizer_config = WZcfgQuant(
        bins_per_plane=bins_per_stage,
        num_planes=stages,
        norm_slices=(slice(None),),
        reconst_ld=reconst_ld,
        train_epochs=training_epochs,
        train_sample_size=min(200_000, positions),
        train_batch_size=1_000,
        lr=3e-3,
        lr_step=40,
        tau=1.0,
        training_progress_bar=False,
        training_log_every_epochs=20,
        quantizer_train_repeats=1,
        prior_train_repeats=1,
        shared_heads=True,
        prior_context="hard",
        prior_probabilities="categorical",
        fused_optimizer=True,
        hidden_dim=100,
        layers=3,
        cache_dir=cache_dir,
    )
    quantizer = DedupedWZQuantizerCancer(quantizer_config, si_size=1)
    quantizer.train_model(training_target, [HistoryVector(training_side_information)])

    sw_config = SWCoderConfig(
        block_length=block_length,
        code_seed=seed,
        matrix_cache_dir=None if cache_dir is None else cache_dir / "sw_matrices",
    )
    # Binding the codec is what replaces the quantizer's training side information with the
    # inference vector's, so nothing that reads the prior may run before it.
    baseline = _codec(_BaselineCodec, quantizer, quantizer_config, inference_side_information)
    source_bins, _, source_metadata = quantizer.encoding_process(inference_target)
    side_info = quantizer.side_info_tensor(source_metadata)
    prior = PriorCalculator.compute_prior_from_network(quantizer.coding_model, source_bins, side_info)
    prior_stage_rates = PriorCalculator.compute_plane_rates(prior, source_bins, stages)
    marginal_stage_rates = PriorCalculator.compute_plane_rates(
        PriorCalculator.compute_marginal_prior(source_bins, bins_per_stage, stages), source_bins, stages)

    results = [_run_method("indices", baseline, inference_target, source_bins, bins_per_stage, None)]

    oracle = _codec(_OracleSWCodec, quantizer, quantizer_config, inference_side_information,
                    sw_config=sw_config, sw_search_step=oracle_row_step)
    oracle.sw_target_ber = oracle_target_ber
    results.append(
        _run_method("oracle", oracle, inference_target, source_bins, bins_per_stage, _oracle_accounting))

    for name, practical_config in (
        ("one-shot", PracticalSWConfig(
            pilot_blocks=0, permutation_seed=seed, bootstrap_margin=bootstrap_margin,
            verification_bits=verification_bits)),
        ("pilot", PracticalSWConfig(
            pilot_blocks=pilot_blocks, permutation_seed=seed, bootstrap_margin=bootstrap_margin,
            safety_rows=safety_rows, verification_bits=verification_bits)),
    ):
        codec = _codec(_PracticalSWCodec, quantizer, quantizer_config, inference_side_information,
                       sw_config=sw_config, practical_config=practical_config)
        results.append(_run_method(
            name, codec, inference_target, source_bins, bins_per_stage, _practical_accounting))

    ladder = _codec(
        _LDPCASWCodec, quantizer, quantizer_config, inference_side_information,
        sw_config=sw_config.model_copy(update={
            "matrix_backend": LDPCA_BACKEND, "ladder_step": ladder_step}),
        ldpca_config=LDPCAConfig(rate_seed=seed, start_margin=ladder_start_margin,
                                 verification_bits=ladder_verification_bits),
    )
    results.append(_run_method(
        "ldpca", ladder, inference_target, source_bins, bins_per_stage, _ldpca_accounting))

    return {
        "bins_per_stage": bins_per_stage,
        "stages": stages,
        "positions": positions,
        "block_length": block_length,
        "pilot_blocks": pilot_blocks,
        "pilot_fraction": pilot_blocks * block_length / positions,
        "bootstrap_margin": bootstrap_margin,
        "safety_rows": safety_rows,
        "verification_bits": verification_bits,
        "ladder_step": ladder_step,
        "ladder_start_margin": ladder_start_margin,
        "ladder_verification_bits": ladder_verification_bits,
        "training_epochs": training_epochs,
        "oracle_target_ber": oracle_target_ber,
        "oracle_row_step": oracle_row_step,
        "seed": seed,
        "prior_stage_rates": list(prior_stage_rates),
        "prior_rate": sum(prior_stage_rates),
        "marginal_stage_rates": list(marginal_stage_rates),
        "marginal_rate": sum(marginal_stage_rates),
        "source_index_rate": stages * float(np.log2(bins_per_stage)),
        "methods": [asdict(result) for result in results],
    }


def _print_report(summary: dict[str, Any]) -> None:
    """Print the per-stage rate/distortion nodes and the gap decomposition."""
    prior = summary["prior_stage_rates"]
    stages = summary["stages"]
    print(
        f"\n{summary['bins_per_stage']} bins x {stages} stages, {summary['positions']} positions, "
        f"block {summary['block_length']}, pilot {summary['pilot_blocks']} blocks "
        f"({summary['pilot_fraction']:.1%})"
    )
    print(f"{'method':>10} {'stage':>5} {'rate':>8} {'MSE dB':>8} {'wMAPE dB':>9} {'vs prior':>9}")
    for method in summary["methods"]:
        for stage in range(stages):
            cumulative = sum(method["stage_rates"][: stage + 1])
            print(
                f"{method['name']:>10} {stage + 1:>5} {cumulative:>8.4f} "
                f"{10 * np.log10(method['stage_mse'][stage]):>8.2f} "
                f"{20 * np.log10(method['stage_wmape'][stage] / 100):>9.2f} "
                f"{cumulative / sum(prior[: stage + 1]) - 1:>8.1%}"
            )
    print(
        f"\nprior (ideal conditional entropy coder) {summary['prior_rate']:.4f} | "
        f"marginal {summary['marginal_rate']:.4f} | raw indices {summary['source_index_rate']:.4f}"
    )
    for method in summary["methods"]:
        total = sum(method["stage_rates"]) + method["overhead_rate"]
        print(
            f"{method['name']:>10} total uplink {total:>7.4f} "
            f"(stages {sum(method['stage_rates']):.4f} + framing {method['overhead_rate']:.4f}) | "
            f"downlink {method['downlink_rate']:.5f} | "
            f"symbol errors {method['symbol_errors']} | repaired blocks {method['repaired_blocks']} | "
            f"{method['seconds']:.0f}s"
        )
    for method in summary["methods"]:
        controller = method["controller"]
        if controller is None:
            continue
        pilot = controller.get("sw_pilot_cross_entropy_rate")
        line = f"{method['name']:>10} controller: model entropy {controller['sw_apriori_rate']:.4f}"
        if pilot is not None:
            line += f" | pilot cross-entropy {pilot:.4f}"
        if controller.get("sw_bulk_rate") is not None:
            line += (
                f" | syndromes {controller['sw_bulk_rate'] + (controller.get('sw_pilot_rate') or 0.0):.4f}"
                f" | hashes and repairs {controller['sw_repair_rate']:.4f}"
                f" ({controller['sw_repaired_blocks']} blocks, "
                f"{controller['sw_pilot_repaired_blocks']} in the pilot)"
            )
        else:
            levels = controller["sw_plane_mean_levels"]
            line += (
                f" | decoded cross-entropy {controller['sw_decoded_cross_entropy_rate']:.4f}"
                f" | rungs {controller['sw_syndrome_rate']:.4f}"
                f" | hashes {controller['sw_hash_rate']:.4f}"
                f" | raw {controller['sw_raw_rate']:.4f} ({controller['sw_raw_blocks']} blocks)"
                f" | mean/min/max level "
                f"{sum(levels) / len(levels):.0f}/{min(controller['sw_plane_min_levels'])}/"
                f"{max(controller['sw_plane_max_levels'])}"
                f" | {controller['sw_ladder_requests']} requests"
            )
        print(line)


def _write_outputs(summary: dict[str, Any], output: Path) -> None:
    """Persist the full summary as JSON and its rate/distortion nodes as a CSV curve."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    curve = output.with_suffix(".csv")
    rows = [
        {
            "config": f"{summary['bins_per_stage']}b_{summary['stages']}s",
            "method": method["name"],
            "stage": stage + 1,
            "rate": sum(method["stage_rates"][: stage + 1]),
            "prior_rate": sum(summary["prior_stage_rates"][: stage + 1]),
            "mse": method["stage_mse"][stage],
            "wmape": method["stage_wmape"][stage],
            "symbol_errors": method["symbol_errors"],
        }
        for method in summary["methods"]
        for stage in range(summary["stages"])
    ]
    with curve.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    """Parse one reproducible practical/oracle comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=2)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--reconst-ld", type=float, default=120.0)
    parser.add_argument("--positions", type=int, default=1_000_000)
    parser.add_argument("--training-epochs", type=int, default=60)
    parser.add_argument("--block-length", type=int, default=2_400)
    parser.add_argument("--pilot-blocks", type=int, default=20)
    parser.add_argument("--bootstrap-margin", type=float, default=PracticalSWConfig().bootstrap_margin)
    parser.add_argument("--safety-rows", type=int, default=PracticalSWConfig().safety_rows)
    parser.add_argument("--verification-bits", type=int, default=PracticalSWConfig().verification_bits)
    parser.add_argument("--ladder-step", type=int, default=SWCoderConfig().ladder_step)
    parser.add_argument("--ladder-start-margin", type=float, default=LDPCAConfig().start_margin)
    parser.add_argument("--ladder-verification-bits", type=int, default=LDPCAConfig().verification_bits)
    parser.add_argument("--oracle-target-ber", type=float, default=1e-5)
    parser.add_argument("--oracle-row-step", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("_/data/protocol_cache/sw_practical"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run, print, and save one practical Slepian-Wolf comparison."""
    args = _parse_args()
    summary = run_comparison(
        bins_per_stage=args.bins,
        stages=args.stages,
        reconst_ld=args.reconst_ld,
        positions=args.positions,
        training_epochs=args.training_epochs,
        block_length=args.block_length,
        pilot_blocks=args.pilot_blocks,
        bootstrap_margin=args.bootstrap_margin,
        safety_rows=args.safety_rows,
        verification_bits=args.verification_bits,
        ladder_step=args.ladder_step,
        ladder_start_margin=args.ladder_start_margin,
        ladder_verification_bits=args.ladder_verification_bits,
        oracle_target_ber=args.oracle_target_ber,
        oracle_row_step=args.oracle_row_step,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    _print_report(summary)
    output = args.output or Path(
        f"Experiments/slepian_wolf/results/practical_sw_{args.bins}b{args.stages}s.json")
    _write_outputs(summary, output)
    print(f"\nsaved {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
