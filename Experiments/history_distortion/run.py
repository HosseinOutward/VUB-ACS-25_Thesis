"""Measure history round-trip errors on real, uncompressed FL client updates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from Experiments.Protocol_test.sparse_replay import load_fl_config
from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, normalization_params
from FL_code.cancer_protocol.history import CompressedHistoryVector, CompressedReconstructionHistory, HistoryVector, ReconstructionHistory


class RoundNormalizedHistory(ReconstructionHistory):
    """Experimental ledger fitting quantizer layer factors on submissions 0, n, 2n, and so on."""

    def __init__(self, num_clients: int, slices: list[slice]) -> None:
        super().__init__(1)
        self.num_clients: int = num_clients
        self.slices: tuple[slice, ...] = tuple(slices)
        self.submissions: int = 0
        self.factors: torch.Tensor | None = None
        self.fit_seconds: float = 0.0

    def _retain(self, value: torch.Tensor, round_id: int, client_id: int) -> CompressedHistoryVector:
        """Fit on the first item in each group and give every payload its own inverse factors."""
        self.fit_seconds = 0.0
        if self.submissions % self.num_clients == 0:
            started = perf_counter()
            self.factors = torch.tensor([normalization_params(value[layer]) for layer in self.slices], dtype=torch.float32)
            self.fit_seconds = perf_counter() - started
        stored = CompressedHistoryVector(value, self.slices, self.factors)
        self.submissions += 1
        return stored


def metrics(source: np.ndarray, restored: np.ndarray) -> dict[str, float | int | None]:
    """Return additive error statistics and relative metrics, excluding exact zeros from MAPE."""
    source, restored = source.astype(np.float64), restored.astype(np.float64)
    error, magnitude = np.abs(restored - source), np.abs(source)
    nonzero = magnitude != 0
    relative = error[nonzero] / magnitude[nonzero]
    total = float(magnitude.sum())
    return {
        "count": source.size, "nonzero": int(nonzero.sum()), "abs_source": total,
        "abs_error": float(error.sum()), "source_sq": float(np.square(source).sum()),
        "error_sq": float(np.square(error).sum()), "relative_sum": float(relative.sum()),
        "wmape_pct": 100 * float(error.sum()) / total if total else None,
        "mape_pct": 100 * float(relative.mean()) if relative.size else None,
        "p99_ape_pct": 100 * float(np.quantile(relative, 0.99)) if relative.size else None,
        "max_ape_pct": 100 * float(relative.max()) if relative.size else None,
        "lost_nonzero": int((nonzero & (restored == 0)).sum()),
        "changed_zero": int((~nonzero & (restored != 0)).sum()),
        "sign_flip": int((source * restored < 0).sum()),
        "max_abs_error": float(error.max()) if error.size else 0.0,
    }


def variants(
    source: torch.Tensor, slices: list[slice], history: CompressedReconstructionHistory,
    round_id: int, client_id: int,
    round_payload: CompressedHistoryVector | None = None, round_admission_seconds: float = 0.0,
) -> Iterator[tuple[str, torch.Tensor, int, float]]:
    """Round-trip the actual history implementation and matched experimental storage variants."""
    names = ("float32", "prefix_raw", "current_5round", "current_fresh", "prefix_pow2", "float16_raw", "float16_pow2")
    for name in names + (("round_shared", "round_shared_float32") if round_payload is not None else ()):
        started = perf_counter()
        factors = torch.tensor([[1.0, 0.0]]).repeat(len(slices), 1)
        if name.endswith("pow2"):
            # A power-of-two scale introduces no centering cancellation and no division rounding
            # for normal float32 values. Store one float32 scale per layer.
            for index, layer in enumerate(slices):
                maximum = float(source[layer].abs().max())
                factors[index, 0] = 2.0 ** np.floor(np.log2(maximum)) if maximum else 1.0
        if name == "round_shared":
            assert round_payload is not None
            yield name, round_payload.materialize(), round_payload.resident_nbytes, round_admission_seconds + perf_counter() - started
            continue
        if name == "round_shared_float32":
            assert round_payload is not None
            restored = source.clone()
            for layer, (scale, center) in zip(slices, round_payload.normalization_factors, strict=True):
                restored[layer] = ((source[layer] - center) / scale) * scale + center
            yield name, restored, source.numel() * 4 + len(slices) * 8, perf_counter() - started
            continue
        if name == "current_5round":
            history.commit(source, CompressionRecord(round_id=round_id, client_id=client_id, round_name_full="history_distortion"), Access.TEMPORAL_TOO)
            payload = history.view(Access.TEMPORAL_TOO, client_id)[client_id][-1].value
        elif name in ("prefix_raw", "prefix_pow2", "current_fresh"):
            payload = CompressedHistoryVector(source, slices, None if name == "current_fresh" else factors)
        elif name == "float32":
            payload = HistoryVector(source)
        else:
            restored = source.clone()
            for layer, (scale, _) in zip(slices, factors, strict=True):
                restored[layer] = (source[layer] / scale).half().float() * scale
            yield name, restored, source.numel() * 2 + (len(slices) * 4 if name.endswith("pow2") else 0), perf_counter() - started
            continue
        restored = payload.materialize()
        yield name, restored, payload.resident_nbytes, perf_counter() - started


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool additive statistics over observations, retaining equal-weight layer diagnostics."""
    rows: list[dict[str, Any]] = []
    for method, group in frame.groupby("method", sort=False):
        sums = group.select_dtypes("number").sum()
        rows.append({
            "method": method, "wmape_pct": 100 * sums.abs_error / sums.abs_source,
            "mape_pct": 100 * sums.relative_sum / sums.nonzero,
            "relative_l2_pct": 100 * np.sqrt(sums.error_sq / sums.source_sq),
            "mean_layer_wmape_pct": group.wmape_pct.mean(),
            "worst_layer_wmape_pct": group.wmape_pct.max(),
            "mean_layer_mape_pct": group.mape_pct.mean(), "worst_layer_mape_pct": group.mape_pct.max(),
            "max_ape_pct": group.max_ape_pct.max(),
            "nonzero_to_zero_pct": 100 * sums.lost_nonzero / sums.nonzero,
            "zero_to_nonzero_pct": 100 * sums.changed_zero / (sums['count'] - sums.nonzero) if sums['count'] > sums.nonzero else None,
            "changed_zero": int(sums.changed_zero), "sign_flip": int(sums.sign_flip),
        })
    return pd.DataFrame(rows)


def per_layer_mape(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool nonzero percentage errors per parameter tensor, listing low-error tensors last."""
    pooled = frame[frame.method.isin(("current_5round", "round_shared"))].groupby(["layer", "method"])[["relative_sum", "nonzero"]].sum()
    pooled["mape_pct"] = 100 * pooled.relative_sum / pooled.nonzero
    layers = pooled.mape_pct.unstack("method")
    layers["tiny"] = (layers[["current_5round", "round_shared"]] < 0.1).all(axis=1)
    return layers.sort_values(["tiny", "current_5round"], ascending=[True, False])


def plot(frame: pd.DataFrame, bins: pd.DataFrame, output: Path) -> None:
    """Save layer-scale and within-layer magnitude diagnostics as a standalone figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = ("prefix_raw", "current_5round", "round_shared", "float16_raw", "float16_pow2")
    for method in methods:
        if method not in frame.method.values:
            continue
        group = frame[frame.method == method].groupby("layer").sum(numeric_only=True)
        axes[0].scatter(np.sqrt(group.source_sq / group["count"]), 100 * group.abs_error / group.abs_source,
                        label=method, s=20, alpha=0.8)
        group = bins[(bins.method == method) & (bins.bin != "zero")].groupby("bin", sort=False).sum(numeric_only=True)
        axes[1].plot(group.index, 100 * group.relative_sum / group.nonzero, marker="o", label=method)
    axes[0].set(xscale="log", yscale="log", xlabel="Layer RMS (raw units)", ylabel="Pooled layer WMAPE (%)")
    axes[1].set(yscale="log", xlabel="|value| / layer RMS", ylabel="Pooled nonzero MAPE (%)")
    axes[1].tick_params(axis="x", labelsize=8, labelrotation=15)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "distortion.png", dpi=170)
    figure.savefig(output / "distortion.pdf")
    plt.close(figure)


def main() -> None:
    """Regenerate a bounded real FL window and write layer, magnitude, and storage reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fl-config", type=Path, default=Path("Experiments/Protocol_test/plans/svhn_30c_fl_config_seed44.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--first-round", type=int, default=0)
    parser.add_argument("--last-round", type=int, default=5)
    parser.add_argument("--clients", type=int, nargs="+", default=[0, 10, 20])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = load_fl_config(args.fl_config)
    assert len(set(args.clients)) == len(args.clients) and all(0 <= client < cfg.num_clients for client in args.clients)
    assert not args.output.exists(), f"Use a fresh output directory: {args.output}"
    args.output.mkdir(parents=True)
    dataset = cfg.data_folder / "SVHN/train_32x32.mat"
    with dataset.open("rb") as stream:
        revision = f"torchvision-svhn-train_32x32.mat-md5:{hashlib.file_digest(stream, 'md5').hexdigest()}"
    trace = Trace(cfg, dataset_revision=revision, device=torch.device(args.device))
    manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
    slices = manager.get_slices()
    history = CompressedReconstructionHistory(1, slices)
    shared_history = RoundNormalizedHistory(cfg.num_clients, slices)
    metadata = {"trace": trace.identity(), "history": history.config_snapshot(), "clients": args.clients,
                "first_round": args.first_round, "last_round": args.last_round,
                "round_shared": {"refresh_submissions": cfg.num_clients, "fit_client": 0,
                                 "estimator": "normalization_params(raw_layer), identical to quantizer preprocessing",
                                 "admission_order": "all clients in ascending order; fit before first admission in each group",
                                 "inverse_factors": "owned by each stored vector; retained across later refreshes"},
                "source": "raw local model deltas after local training; the values compressed by FL, not minibatch gradients"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    layer_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    edges = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, float("inf"))
    for round_id, deltas in trace.stream(args.first_round, args.last_round):
        shared_payloads: dict[int, tuple[CompressedHistoryVector, float]] = {}
        # All admissions count, including clients not selected for distortion measurement.
        for client_id, source in enumerate(deltas):
            started = perf_counter()
            shared_history.commit(source, CompressionRecord(round_id, client_id, "round_shared"), Access.TEMPORAL_TOO)
            if client_id in args.clients:
                payload = shared_history.view(Access.TEMPORAL_TOO, client_id)[client_id][-1].value
                assert isinstance(payload, CompressedHistoryVector)
                shared_payloads[client_id] = (payload, perf_counter() - started)
        for client_id in args.clients:
            source = deltas[client_id]
            assert bool(torch.isfinite(source).all()), f"Nonfinite raw update: round {round_id}, client {client_id}"
            for method, restored, nbytes, seconds in variants(source, slices, history, round_id, client_id, *shared_payloads[client_id]):
                assert bool(torch.isfinite(restored).all()), f"Nonfinite {method}: round {round_id}, client {client_id}"
                if method == "float32":
                    assert torch.equal(source.view(torch.int32), restored.view(torch.int32))
                identity = {"round": round_id, "client": client_id, "method": method}
                storage_rows.append(identity | {"resident_bytes": nbytes, "bytes_per_float": nbytes / source.numel(), "roundtrip_seconds": seconds})
                for key, layer in zip(manager.keys, slices, strict=True):
                    x, y = source[layer].numpy(), restored[layer].numpy()
                    row = identity | {"layer": key} | metrics(x, y)
                    layer_rows.append(row)
                    rms = np.sqrt(row["source_sq"] / row["count"])
                    magnitude = np.abs(x.astype(np.float64)) / rms if rms else np.zeros(x.size)
                    masks = [("zero", x == 0)] + [(f"[{lower:g},{upper:g})", (x != 0) & (magnitude >= lower) & (magnitude < upper))
                                                        for lower, upper in zip(edges, edges[1:])]
                    for label, mask in masks:
                        if mask.any():
                            bin_rows.append(identity | {"layer": key, "bin": label} | metrics(x[mask], y[mask]))
            print(f"Measured round {round_id}, client {client_id}", flush=True)
        # Publish intermediate evidence so a failed later round retains its diagnostics.
        for name, rows in (("layers", layer_rows), ("magnitude_bins", bin_rows), ("storage", storage_rows)):
            with (args.output / f"{name}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    frame, bins = pd.DataFrame(layer_rows), pd.DataFrame(bin_rows)
    summary = summarize(frame).merge(pd.DataFrame(storage_rows).groupby("method", sort=False)[["bytes_per_float", "roundtrip_seconds"]].mean(), on="method")
    summary.to_csv(args.output / "summary.csv", index=False)
    per_layer_mape(frame).to_csv(args.output / "per_layer_mape.csv")
    plot(frame, bins, args.output)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
