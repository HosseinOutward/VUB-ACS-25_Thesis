"""Benchmark exact and compressed reconstruction-history storage.

The benchmark uses a deterministic synthetic delta by default.  Pass ``--trace-delta`` to
measure a flat CPU float32 delta saved by a trace runner.  History storage is local state and is
reported separately from communication rate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from statistics import median
from time import perf_counter
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.cancer_protocol.history import (
    CompressedHistoryVector,
    CompressedReconstructionHistory,
    ReconstructionHistory,
)

MIB = 1024**2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commits", type=int, default=2)
    parser.add_argument("--elements", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--trace-delta", type=Path, help="Path to a flat float32 trace delta .pt file.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _synthetic_delta(elements: int, seed: int) -> torch.Tensor:
    """Create a deterministic heterogeneous gradient-like CPU float32 vector."""
    assert elements > 0
    generator = torch.Generator().manual_seed(seed)
    delta = torch.randn(elements, generator=generator) * 1e-4
    delta[::17] = 0
    delta[::101] *= 12
    return delta.contiguous()


def _load_delta(args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    """Load a flat CPU float32 trace delta or create the synthetic benchmark input."""
    if args.trace_delta is None:
        return _synthetic_delta(args.elements, args.seed), "synthetic"
    assert args.trace_delta.is_file(), f"Trace delta does not exist: {args.trace_delta}"
    delta = torch.load(args.trace_delta, map_location="cpu", weights_only=True)
    assert isinstance(delta, torch.Tensor) and delta.ndim == 1 and delta.dtype is torch.float32, (
        f"Trace delta {args.trace_delta} must be a flat CPU float32 tensor; got "
        f"{type(delta).__name__}, shape={getattr(delta, 'shape', None)}, "
        f"dtype={getattr(delta, 'dtype', None)}."
    )
    return delta.contiguous(), str(args.trace_delta)


def _benchmark_representation(
    history_type: type[ReconstructionHistory], delta: torch.Tensor, commits: int,
) -> dict[str, Any]:
    """Measure commit/materialization time, retained bytes per value, and storage WMAPE."""
    history = history_type(commits)
    commit_seconds: list[float] = []
    for round_id in range(commits):
        value = delta * (1 + round_id / 1_000)
        started = perf_counter()
        history.commit(value, CompressionRecord(round_id, 0, "benchmark"), Access.SERVER_ONLY)
        commit_seconds.append(perf_counter() - started)

    payloads = [entry.value for entry in history.view(Access.SERVER_ONLY, 0)[0]]
    started = perf_counter()
    restored = payloads[-1].materialize()
    first_materialize_seconds = perf_counter() - started
    materialize_seconds = [first_materialize_seconds]
    for _ in range(2):
        started = perf_counter()
        payloads[-1].materialize()
        materialize_seconds.append(perf_counter() - started)

    final = delta * (1 + (commits - 1) / 1_000)
    denominator = float(final.abs().sum())
    wmape = 0.0 if denominator == 0.0 else float((final - restored).abs().sum()) / denominator * 100.0
    retained_bytes = sum(payload.resident_nbytes for payload in payloads)
    result: dict[str, Any] = {
        "representation": f"{history_type.__module__}.{history_type.__qualname__}",
        "commit_seconds": commit_seconds,
        "mean_commit_seconds": sum(commit_seconds) / len(commit_seconds),
        "materialize_seconds": float(median(materialize_seconds)),
        "retained_bytes_per_value": retained_bytes / (len(payloads) * delta.numel()),
        "storage_wmape_percent": wmape,
    }
    compressed = [payload for payload in payloads if isinstance(payload, CompressedHistoryVector)]
    if compressed:
        chunks = [chunk for payload in compressed for chunk in payload._iter_encoded_chunks()]
        result["chunk_layouts"] = dict(Counter(chunk.layout for chunk in chunks))
        result["packed_code_bits"] = dict(Counter(
            chunk.code_bits for chunk in chunks if chunk.layout == "packed"
        ))
    return result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run both history representations against one synthetic or trace delta."""
    assert args.commits > 0
    delta, source = _load_delta(args)
    representations = (ReconstructionHistory, CompressedReconstructionHistory)
    return {
        "source": source,
        "vector_elements": delta.numel(),
        "commits": args.commits,
        "representations": [
            _benchmark_representation(item, delta, args.commits) for item in representations
        ],
    }


def _print_results(result: dict[str, Any]) -> None:
    """Print a compact human-readable benchmark table."""
    print(f"source={result['source']} elements={result['vector_elements']:,} commits={result['commits']}")
    print("\nRepresentation                         commit s   materialize s   bytes/value   WMAPE %")
    for row in result["representations"]:
        print(
            f"{row['representation']:<39} {row['mean_commit_seconds']:>8.4f} "
            f"{row['materialize_seconds']:>14.5f} {row['retained_bytes_per_value']:>13.4f} "
            f"{row['storage_wmape_percent']:>9.5f}"
        )
        if "chunk_layouts" in row:
            print(f"  layouts={row['chunk_layouts']} packed_code_bits={row['packed_code_bits']}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark and optionally save its JSON report."""
    args = _parse_args(argv)
    result = run_benchmark(args)
    _print_results(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
