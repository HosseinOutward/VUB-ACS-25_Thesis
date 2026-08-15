"""Compare Zhang--Tao-style LDPC and SW-ECCT on identical BSC bitstreams.

The 64-bit neural checkpoint and the 2400-bit fixed-rate codec have inherently
different block contracts. This benchmark therefore shares the complete source
and side-information stream, measures fixed-rate BER with one reliability rule,
and reports the neural paper's oracle-adaptive rate separately.

Run the full comparison with::

    python -m Experiments.slepian_wolf.cross_paper_sw_benchmark \
        --positions 240000 --output Experiments/slepian_wolf/results/cross_paper.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from math import ceil, log2
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Final, Sequence

import numpy as np
import pytest
import torch

from Experiments.slepian_wolf.neural_sw_comparison_benchmark import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MOTHER_CODE,
    PAPER_BSC_LEVELS,
    _binary_entropy,
    _load_neural_decoder,
    _matrix_identifier,
    _require_artifacts,
    _resolve_device,
    _sample_bsc,
    _synchronize,
    _wilson_upper,
)
from FL_code.cancer_protocol.sw_coder import (
    ConditionalPrior,
    SWCoderConfig,
    SWRatePlan,
    evaluate_slepian_wolf,
)


TAO_BLOCK_LENGTH: Final = 2_400
NEURAL_BLOCK_LENGTH: Final = 64
DEFAULT_MARGINS: Final = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20)


@dataclass(frozen=True, slots=True)
class CrossPaperPoint:
    """One fixed-rate observation on the common BSC bitstream."""

    decoder: str
    block_length: int
    crossover: float
    entropy_rate: float
    syndrome_rate: float
    bit_errors: int
    block_errors: int
    syndrome_failures: int
    source_bits: int
    source_blocks: int
    seconds: float

    @property
    def bit_error_rate(self) -> float:
        return self.bit_errors / self.source_bits

    @property
    def bit_error_upper_95(self) -> float:
        return _wilson_upper(self.bit_errors, self.source_bits)

    @property
    def block_error_rate(self) -> float:
        return self.block_errors / self.source_blocks

    def row(self) -> dict[str, int | float | str]:
        """Return a stable CSV row including derived reliability metrics."""
        return asdict(self) | {
            "bit_error_rate": self.bit_error_rate,
            "bit_error_upper_95": self.bit_error_upper_95,
            "block_error_rate": self.block_error_rate,
        }


@dataclass(frozen=True, slots=True)
class CrossPaperSummary:
    """Comparable reliable fixed rates and the separately labelled neural oracle rate."""

    crossover: float
    entropy_rate: float
    tao_reliable_fixed_rate: float | None
    neural_reliable_fixed_rate: float | None
    neural_oracle_rate_with_uncoded_fallback: float
    neural_oracle_uncoded_fraction: float

    @property
    def fixed_rate_difference(self) -> float | None:
        if self.tao_reliable_fixed_rate is None or self.neural_reliable_fixed_rate is None:
            return None
        return self.tao_reliable_fixed_rate - self.neural_reliable_fixed_rate

    def row(self) -> dict[str, float | None]:
        """Return the summary and its signed Tao-minus-neural fixed-rate difference."""
        return asdict(self) | {"fixed_rate_difference": self.fixed_rate_difference}


def _bsc_prior(crossover: float) -> ConditionalPrior:
    """Return the exact decoder-owned posterior for a uniform source through a BSC."""
    assert 0 < crossover < 0.5

    def provide(
        decoded_stages: torch.Tensor,
        side_info: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        assert stage == 0 and decoded_stages.shape[0] == 1
        side = side_info[:, 0].long()
        probability_one = torch.where(side.bool(), 1 - crossover, crossover)
        return torch.stack((1 - probability_one, probability_one), dim=1)

    return provide


def _tao_plan(rate: float, cfg: SWCoderConfig) -> SWRatePlan:
    """Create a one-plane fixed-rate plan while preserving the complete matrix contract."""
    rows = min(cfg.block_length, ceil(cfg.block_length * rate))
    return SWRatePlan(
        block_length=cfg.block_length,
        alphabet=2,
        syndrome_rows=((rows,),),
        code_seed=cfg.code_seed,
        degree_distribution=cfg.degree_distribution,
        girth_candidates=cfg.girth_candidates,
        matrix_attempts=cfg.matrix_attempts,
        matrix_backend=cfg.matrix_backend,
    )


def _evaluate_tao_rates(
    source: torch.Tensor,
    side_info: torch.Tensor,
    crossover: float,
    margins: Sequence[float],
    seed: int,
) -> tuple[CrossPaperPoint, ...]:
    """Evaluate independent 2400-bit fixed-rate matrices on one common stream."""
    entropy = _binary_entropy(crossover)
    cfg = SWCoderConfig(block_length=TAO_BLOCK_LENGTH, code_seed=seed)
    prior = _bsc_prior(crossover)
    points: list[CrossPaperPoint] = []
    seen_rows: set[int] = set()
    for margin in margins:
        plan = _tao_plan(entropy + margin, cfg)
        rows = plan.syndrome_rows[0][0]
        if rows in seen_rows:
            continue
        seen_rows.add(rows)
        result = evaluate_slepian_wolf(
            source[None, :],
            plan,
            prior,
            side_info[:, None],
            cfg,
        )
        points.append(
            CrossPaperPoint(
                decoder="tao_ldpc",
                block_length=TAO_BLOCK_LENGTH,
                crossover=crossover,
                entropy_rate=entropy,
                syndrome_rate=result.syndrome_rate,
                bit_errors=result.bit_errors,
                block_errors=result.block_errors,
                syndrome_failures=result.nonconverged_blocks,
                source_bits=result.source_bits,
                source_blocks=result.plane_blocks,
                seconds=result.seconds,
            )
        )
    return tuple(points)


@torch.inference_mode()
def _evaluate_neural_rates(
    model: torch.nn.Module,
    codes: dict[int, SimpleNamespace],
    source: torch.Tensor,
    side_info: torch.Tensor,
    crossover: float,
    batch_size: int,
    device: torch.device,
) -> tuple[tuple[CrossPaperPoint, ...], np.ndarray]:
    """Evaluate every neural LDPCA rate and return per-block exact-decoding flags."""
    sent_blocks = source.view(-1, NEURAL_BLOCK_LENGTH)
    side_blocks = side_info.view(-1, NEURAL_BLOCK_LENGTH)
    magnitude = log2((1 - crossover) / crossover)
    points: list[CrossPaperPoint] = []
    correctness: list[np.ndarray] = []

    for code_number, code in sorted(codes.items()):
        model.set_code(code_number)
        matrix = code.pc_matrix.to(device=device, dtype=torch.float32)
        bit_errors = block_errors = syndrome_failures = 0
        exact_blocks: list[np.ndarray] = []
        seconds = 0.0
        warmed_up = False
        for start in range(0, sent_blocks.shape[0], batch_size):
            sent = sent_blocks[start : start + batch_size].to(device)
            side = side_blocks[start : start + batch_size].to(device)
            syndrome = (sent.float() @ matrix.t()).remainder(2)
            llr = torch.where(side == 0, magnitude, -magnitude).float()
            if not warmed_up:
                model.decode(llrs=llr, syndromes=syndrome)
                _synchronize(device)
                warmed_up = True
            _synchronize(device)
            began = perf_counter()
            decoded = model.decode(llrs=llr, syndromes=syndrome)
            _synchronize(device)
            seconds += perf_counter() - began

            differences = decoded != sent
            failed = differences.any(dim=1)
            decoded_syndrome = (decoded.float() @ matrix.t()).remainder(2)
            bit_errors += int(differences.sum())
            block_errors += int(failed.sum())
            syndrome_failures += int((decoded_syndrome != syndrome).any(dim=1).sum())
            exact_blocks.append((~failed).cpu().numpy())

        correctness.append(np.concatenate(exact_blocks))
        points.append(
            CrossPaperPoint(
                decoder="sw_ecct",
                block_length=NEURAL_BLOCK_LENGTH,
                crossover=crossover,
                entropy_rate=_binary_entropy(crossover),
                syndrome_rate=matrix.shape[0] / matrix.shape[1],
                bit_errors=bit_errors,
                block_errors=block_errors,
                syndrome_failures=syndrome_failures,
                source_bits=source.numel(),
                source_blocks=sent_blocks.shape[0],
                seconds=seconds,
            )
        )
    return tuple(points), np.stack(correctness)


def compare_on_common_bsc_streams(
    checkpoint: Path,
    mother_code: Path,
    crossovers: Sequence[float],
    positions: int,
    margins: Sequence[float],
    batch_size: int,
    target_ber: float,
    seed: int,
    device: torch.device,
) -> tuple[tuple[CrossPaperPoint, ...], tuple[CrossPaperSummary, ...]]:
    """Run both codec families on identical bits and summarize like-for-like fixed rates."""
    assert positions >= TAO_BLOCK_LENGTH and positions % 4_800 == 0
    assert margins and all(margin >= 0 for margin in margins)
    assert 0 < target_ber < 0.5
    model, codes, args = _load_neural_decoder(checkpoint, mother_code, device)
    print(
        f"checkpoint_layers={args.iterations} matrix_family={_matrix_identifier(codes)} "
        f"positions={positions} device={device}"
    )
    points: list[CrossPaperPoint] = []
    summaries: list[CrossPaperSummary] = []

    for noise_index, crossover in enumerate(crossovers):
        source, side_info = _sample_bsc(1, positions, crossover, seed + noise_index)
        source, side_info = source.flatten(), side_info.flatten()
        tao_points = _evaluate_tao_rates(
            source, side_info, crossover, margins, seed=10_000 + noise_index
        )
        neural_points, neural_correctness = _evaluate_neural_rates(
            model, codes, source, side_info, crossover, batch_size, device
        )
        points.extend(tao_points)
        points.extend(neural_points)

        def reliable_rate(rate_points: Sequence[CrossPaperPoint]) -> float | None:
            return next(
                (
                    point.syndrome_rate
                    for point in sorted(rate_points, key=lambda item: item.syndrome_rate)
                    if point.bit_error_upper_95 < target_ber
                ),
                None,
            )

        neural_rates = np.asarray([point.syndrome_rate for point in neural_points])
        first_success = np.argmax(neural_correctness, axis=0)
        any_success = neural_correctness.any(axis=0)
        oracle_rate = float(np.where(any_success, neural_rates[first_success], 1.0).mean())
        summaries.append(
            CrossPaperSummary(
                crossover=crossover,
                entropy_rate=_binary_entropy(crossover),
                tao_reliable_fixed_rate=reliable_rate(tao_points),
                neural_reliable_fixed_rate=reliable_rate(neural_points),
                neural_oracle_rate_with_uncoded_fallback=oracle_rate,
                neural_oracle_uncoded_fraction=float((~any_success).mean()),
            )
        )
    return tuple(points), tuple(summaries)


def _write_csv(path: Path, rows: Sequence[dict[str, int | float | str | None]]) -> None:
    """Write nonempty result rows with their stable dictionary field order."""
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_summaries(summaries: Sequence[CrossPaperSummary]) -> None:
    """Print the fixed-rate comparison and separately labelled neural oracle rate."""
    print("p H tao_fixed neural_fixed tao-neural neural_oracle_with_fallback fallback_fraction")
    for result in summaries:
        tao = (
            "n/a"
            if result.tao_reliable_fixed_rate is None
            else f"{result.tao_reliable_fixed_rate:.4f}"
        )
        neural = (
            "n/a"
            if result.neural_reliable_fixed_rate is None
            else f"{result.neural_reliable_fixed_rate:.4f}"
        )
        fixed = (
            f"{result.fixed_rate_difference:+.4f}"
            if result.fixed_rate_difference is not None
            else "n/a"
        )
        print(
            f"{result.crossover:.4f} {result.entropy_rate:.4f} "
            f"{tao} {neural} {fixed} "
            f"{result.neural_oracle_rate_with_uncoded_fallback:.4f} "
            f"{result.neural_oracle_uncoded_fraction:.4f}"
        )


def test_bsc_prior_is_decoder_owned_and_exact() -> None:
    """Verify the common-data prior without invoking either implementation."""
    source, side = _sample_bsc(1, 4_800, 0.11, seed=4)
    prior = _bsc_prior(0.11)(torch.zeros_like(source), side.flatten()[:, None], 0)
    assert prior.shape == (4_800, 2)
    torch.testing.assert_close(prior.sum(dim=1), torch.ones(4_800))
    expected_one = torch.where(side.flatten().bool(), 0.89, 0.11)
    torch.testing.assert_close(prior[:, 1], expected_one)


@pytest.mark.skipif(
    not bool(int(os.environ.get("SW_CROSS_RUN", "0"))),
    reason="set SW_CROSS_RUN=1 to run both installed decoder paths",
)
def test_cross_paper_smoke_comparison() -> None:
    """Exercise both codecs on one common stream and validate the result contract."""
    _require_artifacts(DEFAULT_CHECKPOINT, DEFAULT_MOTHER_CODE)
    points, summaries = compare_on_common_bsc_streams(
        checkpoint=DEFAULT_CHECKPOINT,
        mother_code=DEFAULT_MOTHER_CODE,
        crossovers=(0.11,),
        positions=4_800,
        margins=(0.0, 0.10),
        batch_size=256,
        target_ber=0.01,
        seed=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    assert len(points) == 17
    assert len(summaries) == 1
    assert {point.source_bits for point in points} == {4_800}


def _parse_args() -> argparse.Namespace:
    """Parse the standalone common-stream benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mother-code", type=Path, default=DEFAULT_MOTHER_CODE)
    parser.add_argument("--positions", type=int, default=240_000)
    parser.add_argument("--margins", type=float, nargs="+", default=DEFAULT_MARGINS)
    parser.add_argument("--crossovers", type=float, nargs="+", default=PAPER_BSC_LEVELS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-ber", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the matched comparison and optionally save fixed points and summaries."""
    args = _parse_args()
    _require_artifacts(args.checkpoint, args.mother_code)
    points, summaries = compare_on_common_bsc_streams(
        checkpoint=args.checkpoint,
        mother_code=args.mother_code,
        crossovers=args.crossovers,
        positions=args.positions,
        margins=args.margins,
        batch_size=args.batch_size,
        target_ber=args.target_ber,
        seed=args.seed,
        device=_resolve_device(args.device),
    )
    _print_summaries(summaries)
    if args.output is not None:
        _write_csv(args.output, tuple(point.row() for point in points))
        summary_path = args.output.with_name(f"{args.output.stem}_summary{args.output.suffix}")
        _write_csv(summary_path, tuple(summary.row() for summary in summaries))


if __name__ == "__main__":
    main()
