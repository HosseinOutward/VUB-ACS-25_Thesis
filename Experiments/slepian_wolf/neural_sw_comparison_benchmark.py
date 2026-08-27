"""Compare the local sum-product decoder with the authors' SW-ECCT decoder.

Both decoders receive the same random BSC source blocks, side information, and
64-bit LDPCA parity-check matrices. The neural decoder uses the LLR convention
from the authors' dataset code, while sum-product receives the equivalent
natural-log LLR required by its update equations.

The comparison requires exactly two author artifacts::

    _/slepian_wolf/Data/ldpc_4x64.txt
    _/slepian_wolf/Data/checkpoint.pt

Run a complete fixed-rate and oracle-adaptive comparison with::

    python -m Experiments.slepian_wolf.neural_sw_comparison_benchmark \
        --blocks 10000 --output /tmp/neural_sw_comparison.csv

The oracle-adaptive rate reproduces the authors' evaluation rule: each block is
assigned the first rate at which its reconstruction equals the ground truth,
with uncoded rate one as fallback. It is reported separately from the smallest
fixed rate whose held-out BER upper confidence bound is below the target.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import log, log2, sqrt
from pathlib import Path
import os
import random
from statistics import NormalDist
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from typing import Any, Final, Sequence

import numpy as np
import pytest
import torch

from FL_code.cancer_protocol.bin_coder.sw_coder import SWCoderConfig, _bp_decode


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
AUTHOR_ROOT: Final = PROJECT_ROOT / "_" / "slepian_wolf"
DEFAULT_DATA_ROOT: Final = AUTHOR_ROOT / "Data"
DEFAULT_MOTHER_CODE: Final = DEFAULT_DATA_ROOT / "ldpc_4x64.txt"
DEFAULT_CHECKPOINT: Final = DEFAULT_DATA_ROOT / "checkpoint.pt"
PAPER_BSC_LEVELS: Final = (0.013, 0.0311, 0.0532, 0.0794, 0.11, 0.146, 0.189, 0.243, 0.316)
AUTHOR_SPLIT_SEED: Final = 2
AUTHOR_MATRIX_FAMILY: Final = "ad594b1f3c92815c"


@dataclass(frozen=True, slots=True)
class DecoderPoint:
    """One fixed-rate result for a decoder, BSC level, and LDPCA matrix."""

    decoder: str
    crossover: float
    entropy_rate: float
    code_number: int
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

    @property
    def syndrome_failure_rate(self) -> float:
        return self.syndrome_failures / self.source_blocks

    @property
    def blocks_per_second(self) -> float:
        return self.source_blocks / self.seconds

    def row(self) -> dict[str, int | float | str]:
        """Return the complete CSV row, including derived reliability metrics."""
        return asdict(self) | {
            "bit_error_rate": self.bit_error_rate,
            "bit_error_upper_95": self.bit_error_upper_95,
            "block_error_rate": self.block_error_rate,
            "syndrome_failure_rate": self.syndrome_failure_rate,
            "blocks_per_second": self.blocks_per_second,
        }


@dataclass(frozen=True, slots=True)
class RateSummary:
    """Reliable fixed and oracle-adaptive rates at one BSC crossover level."""

    decoder: str
    crossover: float
    entropy_rate: float
    reliable_fixed_rate: float
    reliable_excess_rate: float
    oracle_adaptive_rate: float


def _wilson_upper(errors: int, trials: int, confidence: float = 0.95) -> float:
    """Return the one-sided Wilson upper confidence bound for a Bernoulli rate."""
    assert 0 <= errors <= trials
    assert trials > 0
    z = NormalDist().inv_cdf(confidence)
    estimate = errors / trials
    denominator = 1 + z**2 / trials
    center = estimate + z**2 / (2 * trials)
    radius = z * sqrt(estimate * (1 - estimate) / trials + z**2 / (4 * trials**2))
    return (center + radius) / denominator


def _binary_entropy(crossover: float) -> float:
    """Return the BSC conditional entropy in bits per source bit."""
    assert 0 < crossover < 0.5
    return -crossover * log2(crossover) - (1 - crossover) * log2(1 - crossover)


def _author_imports() -> tuple[Any, Any, Any]:
    """Load the untouched author model and LDPCA construction functions."""
    author_path = str(AUTHOR_ROOT)
    if author_path not in sys.path:
        sys.path.insert(0, author_path)
    from ldpca_split.split_checknodes import (  # type: ignore[import-not-found]
        code_rank,
        get_ldpca_permutation,
        load_pc_matrix,
        split_code,
    )

    # The author utility module eagerly imports the optional WZ/rANS stack. ECCT
    # only uses these exact helpers, so isolate that unrelated dependency at import.
    utility_name = "slepian_wolf.utils"
    shim_created = utility_name not in sys.modules
    if shim_created:
        def bin_to_sign(values: torch.Tensor) -> torch.Tensor:
            return 1 - 2 * values

        def llr_to_bin(llrs: torch.Tensor) -> torch.Tensor:
            return (llrs < 0).long()

        utility_shim = ModuleType(utility_name)
        utility_shim.bin_to_sign = bin_to_sign
        utility_shim.llr_to_bin = llr_to_bin
        sys.modules[utility_name] = utility_shim
    try:
        from slepian_wolf.ecct_multi import ECCTMulti  # type: ignore[import-not-found]
    finally:
        if shim_created:
            del sys.modules[utility_name]

    return (
        ECCTMulti,
        (load_pc_matrix, get_ldpca_permutation, split_code, code_rank),
        SimpleNamespace,
    )


def _load_author_codes(
    mother_code: Path,
    code_numbers: Sequence[int],
) -> dict[int, SimpleNamespace]:
    """Reproduce the authors' deterministic 64-bit LDPCA matrix family in memory."""
    _, construction, namespace = _author_imports()
    load_pc_matrix, get_ldpca_permutation, split_code, code_rank = construction
    mother = load_pc_matrix(mother_code)
    rows, block_length = mother.shape
    assert (rows, block_length) == (4, 64), (
        f"expected the author's 4x64 mother code, got {mother.shape} from {mother_code}."
    )
    rate_settings = block_length // rows
    permutation = get_ldpca_permutation(block_length, rate_settings)
    random_state = random.getstate()
    # run.sh creates the checkpoint's code family with split_checknodes.py -s 2.
    random.seed(AUTHOR_SPLIT_SEED)
    try:
        matrices = split_code(
            mother,
            rate_settings,
            permutation,
            ku=round(48 / 66 * rate_settings),
            eta=0.765,
        )
    finally:
        random.setstate(random_state)

    codes: dict[int, SimpleNamespace] = {}
    for number in code_numbers:
        matrix = np.asarray(matrices[number], dtype=np.uint8)
        assert code_rank(matrix) == matrix.shape[0], (
            f"author code {number} is not full row rank over GF(2)."
        )
        syndrome_indices = np.asarray(sorted(permutation[: matrix.shape[0]]))
        codes[number] = namespace(
            n=block_length,
            k=block_length - matrix.shape[0],
            pc_matrix=torch.as_tensor(matrix),
            syndrome_indices=torch.as_tensor(syndrome_indices),
        )
    return codes


def _load_neural_decoder(
    checkpoint_path: Path,
    mother_code: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[int, SimpleNamespace], Any]:
    """Load the exact all-rate/all-BSC SW-ECCT checkpoint and matching code family."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(checkpoint) >= {"model", "args"}, (
        f"unexpected checkpoint fields in {checkpoint_path}: {tuple(checkpoint)}."
    )
    args = checkpoint["args"]
    assert args.ecct and args.correlation == "bsc", (
        f"{checkpoint_path} is not the authors' BSC SW-ECCT checkpoint."
    )
    assert args.code == "ours-64" and tuple(args.code_numbers) == tuple(range(15)), (
        f"expected the all-rate 64-bit checkpoint, got code={args.code}, "
        f"rates={args.code_numbers}."
    )
    codes = _load_author_codes(mother_code, args.code_numbers)
    ECCTMulti, _, _ = _author_imports()
    model = ECCTMulti(
        codes=codes,
        num_heads=8,
        d_model=args.ecct_dim,
        N_dec=args.iterations,
        syndrome_average=args.syndrome_averaging,
        sep_params=args.sep_params,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return model, codes, args


def _matrix_identifier(codes: dict[int, SimpleNamespace]) -> str:
    """Return a short digest that identifies the complete generated code family."""
    digest = sha256()
    for number, code in sorted(codes.items()):
        digest.update(number.to_bytes(2, "big"))
        digest.update(code.pc_matrix.numpy().tobytes())
    return digest.hexdigest()[:16]


def _sample_bsc(
    blocks: int,
    block_length: int,
    crossover: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate uniform source words and BSC-corrupted decoder side information."""
    generator = torch.Generator().manual_seed(seed)
    source = torch.randint(0, 2, (blocks, block_length), generator=generator)
    errors = torch.rand((blocks, block_length), generator=generator) < crossover
    return source, source ^ errors.long()


def _synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before reading a wall-clock timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _evaluate_rate(
    model: torch.nn.Module,
    code_number: int,
    code: SimpleNamespace,
    source: torch.Tensor,
    side_info: torch.Tensor,
    crossover: float,
    batch_size: int,
    bp_iterations: int,
    device: torch.device,
) -> tuple[tuple[DecoderPoint, DecoderPoint], tuple[np.ndarray, np.ndarray]]:
    """Evaluate SW-ECCT and sum-product on identical blocks at one code rate."""
    matrix = code.pc_matrix.to(device=device, dtype=torch.float32)
    check_index, variable_index = torch.where(matrix.bool())
    neural_errors = bp_errors = neural_blocks = bp_blocks = 0
    neural_syndrome_failures = bp_syndrome_failures = 0
    neural_correct: list[np.ndarray] = []
    bp_correct: list[np.ndarray] = []
    neural_seconds = bp_seconds = 0.0
    natural_magnitude = log((1 - crossover) / crossover)
    author_magnitude = log2((1 - crossover) / crossover)
    bp_config = SWCoderConfig(
        block_length=code.n,
        max_bp_iterations=bp_iterations,
        damping=1.0,
        llr_clip=30.0,
    )
    model.set_code(code_number)
    warmed_up = False

    for start in range(0, source.shape[0], batch_size):
        sent = source[start : start + batch_size].to(device)
        side = side_info[start : start + batch_size].to(device)
        syndrome = (sent.float() @ matrix.t()).remainder(2)
        neural_llr = torch.where(side == 0, author_magnitude, -author_magnitude).float()
        bp_llr = torch.where(side == 0, natural_magnitude, -natural_magnitude).float()

        if not warmed_up:
            model.decode(llrs=neural_llr, syndromes=syndrome)
            _bp_decode(
                bp_llr,
                syndrome,
                variable_index,
                check_index,
                bp_config,
            )
            _synchronize(device)
            warmed_up = True

        _synchronize(device)
        began = perf_counter()
        neural_decoded = model.decode(llrs=neural_llr, syndromes=syndrome)
        _synchronize(device)
        neural_seconds += perf_counter() - began

        began = perf_counter()
        bp_decoded, _, _ = _bp_decode(
            bp_llr,
            syndrome,
            variable_index,
            check_index,
            bp_config,
        )
        _synchronize(device)
        bp_seconds += perf_counter() - began

        for decoded, error_name in ((neural_decoded, "neural"), (bp_decoded.long(), "bp")):
            differences = decoded != sent
            block_failures = differences.any(dim=1)
            decoded_syndrome = (decoded.float() @ matrix.t()).remainder(2)
            syndrome_failures = (decoded_syndrome != syndrome).any(dim=1)
            if error_name == "neural":
                neural_errors += int(differences.sum())
                neural_blocks += int(block_failures.sum())
                neural_syndrome_failures += int(syndrome_failures.sum())
                neural_correct.append((~block_failures).cpu().numpy())
            else:
                bp_errors += int(differences.sum())
                bp_blocks += int(block_failures.sum())
                bp_syndrome_failures += int(syndrome_failures.sum())
                bp_correct.append((~block_failures).cpu().numpy())

    entropy = _binary_entropy(crossover)
    common = dict(
        crossover=crossover,
        entropy_rate=entropy,
        code_number=code_number,
        syndrome_rate=matrix.shape[0] / matrix.shape[1],
        source_bits=source.numel(),
        source_blocks=source.shape[0],
    )
    return (
        (
            DecoderPoint(
                decoder="sw_ecct",
                bit_errors=neural_errors,
                block_errors=neural_blocks,
                syndrome_failures=neural_syndrome_failures,
                seconds=neural_seconds,
                **common,
            ),
            DecoderPoint(
                decoder="sum_product",
                bit_errors=bp_errors,
                block_errors=bp_blocks,
                syndrome_failures=bp_syndrome_failures,
                seconds=bp_seconds,
                **common,
            ),
        ),
        (np.concatenate(neural_correct), np.concatenate(bp_correct)),
    )


def compare_decoders(
    checkpoint: Path,
    mother_code: Path,
    crossovers: Sequence[float],
    blocks: int,
    batch_size: int,
    bp_iterations: int,
    target_ber: float,
    seed: int,
    device: torch.device,
) -> tuple[tuple[DecoderPoint, ...], tuple[RateSummary, ...]]:
    """Run fixed-rate curves and summarize reliable and oracle-adaptive rates."""
    assert blocks >= 1
    assert batch_size >= 1
    assert bp_iterations >= 1
    assert 0 < target_ber < 0.5
    model, codes, args = _load_neural_decoder(checkpoint, mother_code, device)
    print(
        f"checkpoint={checkpoint} layers={args.iterations} dim={args.ecct_dim} "
        f"matrix_family={_matrix_identifier(codes)} device={device}"
    )
    points: list[DecoderPoint] = []
    summaries: list[RateSummary] = []

    for noise_index, crossover in enumerate(crossovers):
        assert 0 < crossover < 0.5
        source, side_info = _sample_bsc(
            blocks, next(iter(codes.values())).n, crossover, seed + noise_index
        )
        correctness: dict[str, list[np.ndarray]] = {"sw_ecct": [], "sum_product": []}
        noise_points: list[DecoderPoint] = []
        for code_number, code in sorted(codes.items()):
            rate_points, rate_correctness = _evaluate_rate(
                model,
                code_number,
                code,
                source,
                side_info,
                crossover,
                batch_size,
                bp_iterations,
                device,
            )
            noise_points.extend(rate_points)
            correctness["sw_ecct"].append(rate_correctness[0])
            correctness["sum_product"].append(rate_correctness[1])
        points.extend(noise_points)

        for decoder, decoder_correctness in correctness.items():
            decoder_points = sorted(
                (point for point in noise_points if point.decoder == decoder),
                key=lambda point: point.syndrome_rate,
            )
            reliable = next(
                (
                    point.syndrome_rate
                    for point in decoder_points
                    if point.bit_error_upper_95 < target_ber
                ),
                1.0,
            )
            correct_by_rate = np.stack(decoder_correctness)
            rates = np.asarray([point.syndrome_rate for point in decoder_points])
            first_success = np.argmax(correct_by_rate, axis=0)
            any_success = correct_by_rate.any(axis=0)
            oracle_rates = np.where(any_success, rates[first_success], 1.0)
            summaries.append(
                RateSummary(
                    decoder=decoder,
                    crossover=crossover,
                    entropy_rate=_binary_entropy(crossover),
                    reliable_fixed_rate=reliable,
                    reliable_excess_rate=reliable - _binary_entropy(crossover),
                    oracle_adaptive_rate=float(oracle_rates.mean()),
                )
            )
    return tuple(points), tuple(summaries)


def _write_csv(path: Path, points: Sequence[DecoderPoint]) -> None:
    """Write fixed-rate observations to a CSV file with a stable schema."""
    rows = [point.row() for point in points]
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_results(
    points: Sequence[DecoderPoint], summaries: Sequence[RateSummary]
) -> None:
    """Print fixed-rate curves and the comparable headline rate summaries."""
    print(
        "decoder p H code rate BER upper95 BLER syndrome_fail blocks/s seconds"
    )
    for point in points:
        print(
            f"{point.decoder:>11} {point.crossover:>6.4f} {point.entropy_rate:>6.4f} "
            f"{point.code_number:>4} {point.syndrome_rate:>6.4f} "
            f"{point.bit_error_rate:>9.3%} {point.bit_error_upper_95:>9.3%} "
            f"{point.block_error_rate:>9.3%} {point.syndrome_failure_rate:>13.3%} "
            f"{point.blocks_per_second:>8.1f} {point.seconds:>8.3f}"
        )
    print("decoder p H reliable fixed_excess oracle_adaptive")
    for summary in summaries:
        print(
            f"{summary.decoder:>11} {summary.crossover:>6.4f} "
            f"{summary.entropy_rate:>6.4f} {summary.reliable_fixed_rate:>8.4f} "
            f"{summary.reliable_excess_rate:>12.4f} "
            f"{summary.oracle_adaptive_rate:>15.4f}"
        )


def _parse_args() -> argparse.Namespace:
    """Parse standalone benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mother-code", type=Path, default=DEFAULT_MOTHER_CODE)
    parser.add_argument("--blocks", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bp-iterations", type=int, default=100)
    parser.add_argument("--target-ber", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crossovers", type=float, nargs="+", default=PAPER_BSC_LEVELS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    """Resolve the requested compute device and fail clearly when CUDA is unavailable."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert name != "cuda" or torch.cuda.is_available(), "CUDA was requested but is unavailable."
    return torch.device(name)


def _require_artifacts(checkpoint: Path, mother_code: Path) -> None:
    """Fail with the exact minimal download paths needed by this comparison."""
    missing = [path for path in (checkpoint, mother_code) if not path.is_file()]
    assert not missing, "missing author artifacts: " + ", ".join(map(str, missing))


def test_author_neural_module_imports_without_wz_dependencies() -> None:
    """Ensure the comparison isolates the author's unrelated optional WZ dependencies."""
    ECCTMulti, construction, _ = _author_imports()
    assert ECCTMulti.__name__ == "ECCTMulti"
    assert len(construction) == 4


def test_matched_comparison_metric_contract() -> None:
    """Exercise both decoder paths on a full-rate identity syndrome without artifacts."""
    class IdentityDecoder(torch.nn.Module):
        def set_code(self, code_number: int) -> None:
            assert code_number == 0

        def decode(self, llrs: torch.Tensor, syndromes: torch.Tensor) -> torch.Tensor:
            del llrs
            return syndromes.long()

    code = SimpleNamespace(
        n=32,
        k=0,
        pc_matrix=torch.eye(32, dtype=torch.uint8),
        syndrome_indices=torch.arange(32),
    )
    source, side_info = _sample_bsc(32, 32, 0.11, seed=7)
    points, correctness = _evaluate_rate(
        IdentityDecoder(),
        code_number=0,
        code=code,
        source=source,
        side_info=side_info,
        crossover=0.11,
        batch_size=16,
        bp_iterations=10,
        device=torch.device("cpu"),
    )
    assert all(point.bit_errors == 0 for point in points)
    assert all(point.syndrome_failures == 0 for point in points)
    assert all(values.all() for values in correctness)


def test_author_artifact_contract() -> None:
    """Validate the checkpoint identity and exact LDPCA family when artifacts are installed."""
    if not DEFAULT_CHECKPOINT.is_file() or not DEFAULT_MOTHER_CODE.is_file():
        pytest.skip("download the BSC ECCT checkpoint and ldpc_4x64.txt to the documented paths")
    model, codes, args = _load_neural_decoder(
        DEFAULT_CHECKPOINT, DEFAULT_MOTHER_CODE, torch.device("cpu")
    )
    assert args.iterations == 15
    assert next(iter(codes.values())).n == 64
    assert _matrix_identifier(codes) == AUTHOR_MATRIX_FAMILY
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


@pytest.mark.skipif(
    not bool(int(os.environ.get("SW_NEURAL_RUN", "0"))),
    reason="set SW_NEURAL_RUN=1 after installing the two author artifacts",
)
def test_neural_and_sum_product_smoke_comparison() -> None:
    """Run a small matched-source comparison and verify the result contract."""
    _require_artifacts(DEFAULT_CHECKPOINT, DEFAULT_MOTHER_CODE)
    points, summaries = compare_decoders(
        checkpoint=DEFAULT_CHECKPOINT,
        mother_code=DEFAULT_MOTHER_CODE,
        crossovers=(0.11,),
        blocks=128,
        batch_size=64,
        bp_iterations=100,
        target_ber=0.01,
        seed=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    assert len(points) == 30
    assert {summary.decoder for summary in summaries} == {"sw_ecct", "sum_product"}
    assert all(point.source_blocks == 128 for point in points)


def main() -> None:
    """Run the author-checkpoint comparison and optionally persist the fixed-rate table."""
    args = _parse_args()
    _require_artifacts(args.checkpoint, args.mother_code)
    points, summaries = compare_decoders(
        checkpoint=args.checkpoint,
        mother_code=args.mother_code,
        crossovers=args.crossovers,
        blocks=args.blocks,
        batch_size=args.batch_size,
        bp_iterations=args.bp_iterations,
        target_ber=args.target_ber,
        seed=args.seed,
        device=_resolve_device(args.device),
    )
    _print_results(points, summaries)
    if args.output is not None:
        _write_csv(args.output, points)


if __name__ == "__main__":
    main()
