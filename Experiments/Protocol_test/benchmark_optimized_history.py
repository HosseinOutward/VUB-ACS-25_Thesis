"""Benchmark raw, preprocessed, float16, deduplicated, and disk-backed gradient history."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import gc
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.utils import GradientPreprocessMetadata, StateDictManager
from FL_code.cancer_protocol.optimized_history import (
    OptimizedGradientHistory,
    build_two_level_gradient_index,
    deduplicate_gradient_values,
)


MIB = 1024**2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commits", type=int, default=3, help="Vectors actually committed per backend.")
    parser.add_argument("--projected-count", type=int, default=150)
    parser.add_argument("--outlier-threshold", type=float, default=8.0)
    parser.add_argument("--dedup-match-bits", type=int, default=14)
    parser.add_argument("--base-code-bits", type=int, default=9)
    parser.add_argument(
        "--round-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--storage-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--indexed-values", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--elements",
        type=int,
        help="Use a synthetic equal-layer model of this size instead of the default ResNet-18.",
    )
    return parser.parse_args(argv)


def _manager(elements: int | None) -> StateDictManager:
    """Build the target ResNet-18 layout or a smaller synthetic benchmark layout."""
    if elements is None:
        return StateDictManager(resnet18(weights=None, num_classes=10))
    assert elements > 1
    layer_sizes = [elements // 8] * 8
    layer_sizes[-1] += elements - sum(layer_sizes)

    class SyntheticModel(nn.Module):
        """Parameter-only model used to benchmark a requested flat-vector size."""

        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ParameterList([nn.Parameter(torch.empty(size)) for size in layer_sizes])

    return StateDictManager(SyntheticModel())


def _delta(manager: StateDictManager, seed: int) -> torch.Tensor:
    """Generate a deterministic layer-heteroscedastic, heavy-tailed gradient-like vector."""
    generator = torch.Generator().manual_seed(seed)
    delta = torch.empty(manager.param_count, dtype=torch.float32)
    for layer_id, layer_slice in enumerate(manager.get_slices()):
        layer = torch.randn(manager.numels[layer_id], generator=generator)
        layer[::97] *= 12
        layer[::29] = 0
        delta[layer_slice] = layer * (10 ** (-6 + layer_id % 5))
    return delta


def _metadata_nbytes(metadata: GradientPreprocessMetadata) -> int:
    return (
        metadata.norm_factors.numel() * metadata.norm_factors.element_size()
        + metadata.outliers.positions.nbytes
        + metadata.outliers.packed_signs.nbytes
        + (0 if metadata.outliers.scale is None else metadata.outliers.scale.nbytes)
    )


def _index_layout(
    name: str,
    values: torch.Tensor,
    inverse: torch.Tensor,
    metadata_bytes: int,
    raw_bytes: int,
) -> dict[str, Any]:
    """Estimate dense, single-dummy, and two-level index layouts from value frequencies."""
    counts = torch.bincount(inverse, minlength=values.numel()).sort(descending=True).values
    size, unique_count = inverse.numel(), values.numel()
    code_bytes = 1 if unique_count <= 2**8 else 2 if unique_count <= 2**16 else 4
    dense_bytes = metadata_bytes + unique_count * 2 + size * code_bytes
    exceptions = size - int(counts[0])
    single_dummy_bytes = metadata_bytes + unique_count * 2 + exceptions * (4 + code_bytes)

    two_level_candidates: list[tuple[int, int, int, int]] = []
    for base_bits in range(1, math.ceil(math.log2(unique_count + 1)) + 1):
        frequent_values = min(2**base_bits - 1, unique_count)
        exception_count = size - int(counts[:frequent_values].sum())
        remaining_values = unique_count - frequent_values
        exception_bits = 0 if remaining_values <= 1 else math.ceil(math.log2(remaining_values))
        index_bytes = (size * base_bits + exception_count * exception_bits + 7) // 8
        two_level_candidates.append((
            metadata_bytes + unique_count * 2 + index_bytes,
            base_bits,
            frequent_values,
            exception_count,
        ))
    two_level_bytes, base_bits, frequent_values, exception_count = min(two_level_candidates)
    return {
        "name": name,
        "unique_values": unique_count,
        "top_value_percent": float(counts[0] / size * 100),
        "top_32_percent": float(counts[:32].sum() / size * 100),
        "dense_dictionary_mib": dense_bytes / MIB,
        "single_dummy_sparse_mib": single_dummy_bytes / MIB,
        "best_two_level_mib": two_level_bytes / MIB,
        "best_two_level_compression_factor": raw_bytes / two_level_bytes,
        "best_base_bits": base_bits,
        "frequent_values": frequent_values,
        "exception_percent": exception_count / size * 100,
    }


def _stage_benchmark(
    manager: StateDictManager,
    delta: torch.Tensor,
    threshold: float,
    storage_dtype: torch.dtype,
    dedup_match_bits: int,
    base_code_bits: int,
    round_prefix: bool,
    projected_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure each transformation separately and return retained-size projections."""
    raw_bytes = delta.numel() * delta.element_size()
    started = perf_counter()
    normalized, metadata = manager.normalize_delta(delta, threshold)
    normalization_seconds = perf_counter() - started
    metadata_bytes = _metadata_nbytes(metadata)

    started = perf_counter()
    quantized = normalized.to(storage_dtype)
    quantization_seconds = perf_counter() - started

    started = perf_counter()
    values, inverse = deduplicate_gradient_values(
        normalized, dedup_match_bits, storage_dtype, round_prefix)
    dedup_seconds = perf_counter() - started
    started = perf_counter()
    two_level_index = build_two_level_gradient_index(values, inverse, base_code_bits)
    two_level_seconds = perf_counter() - started
    exact_values, exact_inverse = deduplicate_gradient_values(normalized, 32, storage_dtype)
    restored_quantized = manager.denormalize_delta(quantized.float(), metadata, threshold)
    quantized_wmape = float((restored_quantized - delta).abs().sum() / delta.abs().sum() * 100)
    restored_dedup = manager.denormalize_delta(values[inverse], metadata, threshold)
    dedup_wmape = float((restored_dedup - delta).abs().sum() / delta.abs().sum() * 100)
    index_bytes = 1 if values.numel() <= 2**8 else 2 if values.numel() <= 2**16 else 4
    dedup_bytes = values.numel() * values.element_size() + inverse.numel() * index_bytes + metadata_bytes

    stages = (
        ("raw_float32", raw_bytes, 0.0, 0.0),
        ("layer_normalized_float32", raw_bytes + metadata_bytes, normalization_seconds, 0.0),
        (
            f"normalized_{str(storage_dtype).removeprefix('torch.')}",
            quantized.numel() * quantized.element_size() + metadata_bytes,
            quantization_seconds,
            quantized_wmape,
        ),
        (
            f"{str(storage_dtype).removeprefix('torch.')}_prefix_dense_index_{dedup_match_bits}b",
            dedup_bytes,
            dedup_seconds,
            dedup_wmape,
        ),
        (
            f"{str(storage_dtype).removeprefix('torch.')}_uint{base_code_bits}_escape_{dedup_match_bits}b",
            two_level_index.nbytes + metadata_bytes,
            two_level_seconds,
            dedup_wmape,
        ),
    )
    stage_rows = [
        {
            "stage": name,
            "seconds": seconds,
            "retained_mib_per_vector": retained_bytes / MIB,
            "projected_mib": retained_bytes * projected_count / MIB,
            "ratio_to_raw": retained_bytes / raw_bytes,
            "compression_factor": raw_bytes / retained_bytes,
            "memory_reduction_percent": (1 - retained_bytes / raw_bytes) * 100,
            "storage_wmape_percent": storage_wmape,
            "unique_values": values.numel() if "dedup" in name else None,
        }
        for name, retained_bytes, seconds, storage_wmape in stages
    ]
    layouts = [
        _index_layout(
            f"{str(storage_dtype).removeprefix('torch.')}_exact",
            exact_values,
            exact_inverse,
            metadata_bytes,
            raw_bytes,
        ),
        _index_layout(
            f"{str(storage_dtype).removeprefix('torch.')}_{dedup_match_bits}b_prefix",
            values,
            inverse,
            metadata_bytes,
            raw_bytes,
        ),
    ]
    return stage_rows, layouts


def _history_benchmark(
    manager: StateDictManager,
    base_delta: torch.Tensor,
    threshold: float,
    commits: int,
    projected_count: int,
    indexed_values: int,
    storage_dir: Path | None,
    dedup_match_bits: int,
    storage_dtype: torch.dtype,
    base_code_bits: int,
    round_prefix: bool,
) -> dict[str, Any]:
    """Measure repeated commits plus lazy indexed and full reconstruction for one backend."""
    history = OptimizedGradientHistory(
        commits,
        manager,
        outlier_threshold=threshold,
        storage_dir=storage_dir,
        dedup_match_bits=dedup_match_bits,
        storage_dtype=storage_dtype,
        base_code_bits=base_code_bits,
        round_prefix=round_prefix,
    )
    commit_times: list[float] = []
    for round_id in range(commits):
        delta = base_delta if round_id == 0 else base_delta * (1 + round_id / 1_000)
        started = perf_counter()
        history.commit(delta, CompressionRecord(round_id, 0, "benchmark"), Access.SERVER_ONLY)
        commit_times.append(perf_counter() - started)
        if round_id:
            del delta

    tensor = history.view(Access.SERVER_ONLY, 0)[0][-1].tensor
    selection = torch.linspace(
        0, manager.param_count - 1, min(indexed_values, manager.param_count), dtype=torch.long)
    indexed_times: list[float] = []
    full_times: list[float] = []
    for _ in range(5):
        started = perf_counter()
        selected = tensor[selection]
        indexed_times.append(perf_counter() - started)
        assert selected.numel() == selection.numel()
        del selected

        started = perf_counter()
        materialized = tensor.materialize()
        full_times.append(perf_counter() - started)
        assert materialized.numel() == manager.param_count
        del materialized
    per_vector_bytes = (history.resident_nbytes + history.disk_nbytes) / commits
    result = {
        "backend": "disk" if storage_dir is not None else "ram",
        "actual_commits": commits,
        "mean_commit_seconds": float(np.mean(commit_times)),
        "indexed_values": selection.numel(),
        "indexed_read_seconds": float(np.median(indexed_times)),
        "full_materialization_seconds": float(np.median(full_times)),
        "resident_mib": history.resident_nbytes / MIB,
        "disk_mib": history.disk_nbytes / MIB,
        "retained_mib_per_vector": per_vector_bytes / MIB,
        "projected_mib": per_vector_bytes * projected_count / MIB,
    }
    del tensor, history
    gc.collect()
    return result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the transformation ablation and both optimized-history storage backends."""
    assert args.commits > 0 and args.projected_count > 0 and args.indexed_values > 0
    manager = _manager(args.elements)
    delta = _delta(manager, args.seed)
    storage_dtype = getattr(torch, args.storage_dtype)
    stages, layouts = _stage_benchmark(
        manager,
        delta,
        args.outlier_threshold,
        storage_dtype,
        args.dedup_match_bits,
        args.base_code_bits,
        args.round_prefix,
        args.projected_count,
    )
    ram = _history_benchmark(
        manager,
        delta,
        args.outlier_threshold,
        args.commits,
        args.projected_count,
        args.indexed_values,
        None,
        args.dedup_match_bits,
        storage_dtype,
        args.base_code_bits,
        args.round_prefix,
    )
    with TemporaryDirectory(prefix="optimized_gradient_history_") as temporary_dir:
        disk = _history_benchmark(
            manager,
            delta,
            args.outlier_threshold,
            args.commits,
            args.projected_count,
            args.indexed_values,
            Path(temporary_dir),
            args.dedup_match_bits,
            storage_dtype,
            args.base_code_bits,
            args.round_prefix,
        )
    return {
        "vector_elements": manager.param_count,
        "raw_mib_per_vector": delta.numel() * delta.element_size() / MIB,
        "projected_count": args.projected_count,
        "outlier_threshold": args.outlier_threshold,
        "dedup_match_bits": args.dedup_match_bits,
        "storage_dtype": args.storage_dtype,
        "base_code_bits": args.base_code_bits,
        "round_prefix": args.round_prefix,
        "stages": stages,
        "unique_layouts": layouts,
        "backends": [ram, disk],
    }


def _print_results(results: dict[str, Any]) -> None:
    """Print compact human-readable benchmark tables."""
    print(
        f"elements={results['vector_elements']:,}  raw={results['raw_mib_per_vector']:.2f} MiB/vector  "
        f"projection={results['projected_count']} vectors"
    )
    print("\nTransformation                 seconds   MiB/vector   raw/size   WMAPE %   projected MiB")
    for row in results["stages"]:
        print(
            f"{row['stage']:<30} {row['seconds']:>8.3f} {row['retained_mib_per_vector']:>12.3f} "
            f"{row['compression_factor']:>10.3f}x {row['storage_wmape_percent']:>9.5f} "
            f"{row['projected_mib']:>15.1f}"
        )
    print("\nUnique layout       unique   top value   top 32   dense MiB   dummy+positions MiB   best 2-level MiB")
    for row in results["unique_layouts"]:
        print(
            f"{row['name']:<19} {row['unique_values']:>6} {row['top_value_percent']:>10.3f}% "
            f"{row['top_32_percent']:>8.3f}% {row['dense_dictionary_mib']:>11.3f} "
            f"{row['single_dummy_sparse_mib']:>21.3f} {row['best_two_level_mib']:>18.3f}"
        )
    print("\nBackend  commit s/vector   indexed read s   full read s   resident MiB   disk MiB   projected MiB")
    for row in results["backends"]:
        print(
            f"{row['backend']:<7} {row['mean_commit_seconds']:>15.3f} "
            f"{row['indexed_read_seconds']:>16.4f} {row['full_materialization_seconds']:>13.3f} "
            f"{row['resident_mib']:>14.2f} {row['disk_mib']:>10.2f} {row['projected_mib']:>15.1f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line benchmark and optionally persist its JSON result."""
    args = _parse_args(argv)
    results = run_benchmark(args)
    _print_results(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
