"""Count changed coordinates and histogram history/full-SI errors before target normalization."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from Experiments.history_distortion.run import RoundNormalizedHistory, load_fl_config
from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager
from FL_code.cancer_protocol.history import CompressedHistoryVector, CompressedReconstructionHistory
from FL_code.cancer_protocol.wz_quantizer import SideInfoMatrix


class Distribution:
    """Streaming error counts produced from raw/reference blocks and consumed by plots and tables."""

    def __init__(self, edges: np.ndarray) -> None:
        self.edges: np.ndarray = edges
        self.bins: np.ndarray = np.zeros(edges.size - 1, dtype=np.int64)
        self.counts: Counter[str] = Counter()
        self.ape_sum: float = 0.0
        self.abs_error: float = 0.0
        self.abs_source: float = 0.0
        self.max_ape: float = 0.0

    def add(self, reference: torch.Tensor, restored: torch.Tensor) -> None:
        """Count numerical changes, retaining exact-zero errors separately from percentage errors."""
        source = reference.numpy().astype(np.float64).reshape(-1)
        error = np.abs(restored.cpu().numpy().astype(np.float64).reshape(-1) - source)
        nonzero = source != 0
        ape = 100 * error[nonzero] / np.abs(source[nonzero])
        self.bins += np.histogram(ape, self.edges)[0]
        self.counts.update({
            "coordinates": source.size, "nonzero_reference": int(nonzero.sum()),
            "changed": int((error != 0).sum()), "changed_nonzero": int((nonzero & (error != 0)).sum()),
            "changed_zero": int((~nonzero & (error != 0)).sum()),
            **{f"ape_gt_{threshold:g}": int((ape > threshold).sum()) for threshold in (0.01, 0.1, 1.0, 10.0, 100.0)},
        })
        self.ape_sum += float(ape.sum())
        self.abs_error += float(error.sum())
        self.abs_source += float(np.abs(source).sum())
        self.max_ape = max(self.max_ape, float(ape.max()) if ape.size else 0.0)

    def summary(self) -> dict[str, Any]:
        """Return exact changed fractions and threshold fractions with explicit denominators."""
        total, nonzero = self.counts["coordinates"], self.counts["nonzero_reference"]
        assert self.bins.sum() == nonzero, "Percentage histogram must account for every nonzero reference coordinate."
        return dict(self.counts) | {
            "changed_pct_all": 100 * self.counts["changed"] / total,
            "source_zero_pct_all": 100 * (total - nonzero) / total,
            "changed_zero_pct_all": 100 * self.counts["changed_zero"] / total,
            "changed_pct_nonzero": 100 * self.counts["changed_nonzero"] / nonzero,
            "mape_pct_nonzero": self.ape_sum / nonzero, "wmape_pct": 100 * self.abs_error / self.abs_source,
            "max_ape_pct": self.max_ape,
            **{f"ape_gt_{threshold:g}_pct_nonzero": 100 * self.counts[f"ape_gt_{threshold:g}"] / nonzero
               for threshold in (0.01, 0.1, 1.0, 10.0, 100.0)},
        }


def plot(distributions: dict[tuple[str, str], Distribution], output: Path, y_max: float) -> None:
    """Save two capped histograms, using nonzero coordinates as the percentage-error denominator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, (scope, title) in zip(axes, (("entry", "Single history entry: client 10, round 4"),
                                         ("full_si", "Full SI: 30 clients × 5 history entries")), strict=True):
        notes: list[str] = []
        for method, label in (("current_5round", "Current"), ("round_shared", "Round-shared")):
            distribution = distributions[scope, method]
            percentages = 100 * distribution.bins / distribution.counts["nonzero_reference"]
            axis.stairs(percentages[1:-1], distribution.edges[1:-1], label=label, linewidth=1.6)
            summary = distribution.summary()
            notes.append(f"{label}: {summary['changed_pct_all']:.3f}% numerically changed; peak bin {percentages.max():.1f}%")
        axis.set(xscale="log", xlim=(1e-6, 1e6), ylim=(0, y_max), title=title,
                 xlabel="Absolute percentage error: 100 × |stored − reference| / |reference|")
        axis.axvline(1, color="gray", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
        axis.text(0.02, 0.97, "\n".join(notes), transform=axis.transAxes, va="top", fontsize=8)
        zeros = distribution.summary()["source_zero_pct_all"]
        axis.text(0.02, 0.02, f"Exact source zeros: {zeros:.2f}% of all coordinates\nZero errors counted separately; no epsilon used",
                  transform=axis.transAxes, va="bottom", fontsize=8)
    axes[0].set_ylabel("Nonzero reference coordinates per bin (%)")
    figure.suptitle(f"Storage distortion in raw gradient units — y-axis capped at {y_max:g}% per bin")
    figure.tight_layout()
    figure.savefig(output / "histograms.png", dpi=180)
    figure.savefig(output / "histograms.pdf")
    plt.close(figure)


def main() -> None:
    """Regenerate the matched five-round history and measure all 150 full-SI entries in bounded blocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fl-config", type=Path, default=Path("Experiments/Protocol_test/plans/svhn_30c_fl_config_seed44.json"))
    parser.add_argument("--reference", type=Path, default=Path("Experiments/history_distortion/results/svhn_seed44_round_shared_20261005"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--y-max", type=float, default=0.5, help="Upper y-axis limit, in percent of nonzero coordinates per bin.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), f"Use a fresh output directory: {args.output}"
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    reference = json.loads((args.reference / "metadata.json").read_text())
    cfg = load_fl_config(args.fl_config)
    device = torch.device(args.device)
    trace = Trace(cfg, dataset_revision=reference["trace"]["dataset_revision"], device=device)
    assert trace.identity() == reference["trace"], "Histogram and earlier distortion experiments must use the same trace."
    slices = StateDictManager(initialize_model(cfg, torch.device("cpu"))).get_slices()
    histories = {"current_5round": CompressedReconstructionHistory(5, slices),
                 "round_shared": RoundNormalizedHistory(cfg.num_clients, slices)}
    histories["round_shared"].max_per_client = 5
    raw: dict[int, list[torch.Tensor]] = {client: [] for client in range(cfg.num_clients)}
    for round_id, deltas in trace.stream(0, 4):
        for client_id, source in enumerate(deltas):
            raw[client_id].append(source)
            for history in histories.values():
                history.commit(source, CompressionRecord(round_id, client_id, "histogram"), Access.TEMPORAL_TOO)
        print(f"Retained all clients' raw and compressed history for round {round_id}", flush=True)
    edges = np.r_[0.0, np.geomspace(1e-6, 1e6, 97), np.inf]
    distributions = {(scope, method): Distribution(edges) for scope in ("entry", "full_si") for method in histories}
    source_columns = [source for ledger in raw.values() for source in ledger]
    factors = torch.tensor([[1.0, 0.0]]).repeat(len(slices), 1)
    for method, history in histories.items():
        ledgers = history.view(Access.SERVER_ONLY, 10)
        single = ledgers[10][-1].value
        distributions["entry", method].add(raw[10][-1], single.materialize())
        columns = [entry.value for ledger in ledgers.values() for entry in ledger]
        assert len(columns) == len(source_columns) == cfg.num_clients * 5
        for start in range(0, len(columns), 8):
            # Identity target factors expose restored raw SI; no downstream normalization is applied.
            matrix = SideInfoMatrix(columns[start:start + 8], source_columns[0].numel(), factors, slices, device)
            for first in range(0, source_columns[0].numel(), 200_000):
                block = slice(first, min(first + 200_000, source_columns[0].numel()))
                exact = torch.stack([source[block] for source in source_columns[start:start + 8]], dim=1)
                distributions["full_si", method].add(exact, matrix[block])
            print(f"Measured {method} full-SI columns {start + 1}–{min(start + 8, len(columns))}", flush=True)
        del matrix
    rows = [{"scope": scope, "method": method} | distribution.summary() for (scope, method), distribution in distributions.items()]
    pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame([{"scope": scope, "method": method, "lower_ape_pct": lower, "upper_ape_pct": upper,
                   "count": int(count), "pct_nonzero": 100 * count / distribution.counts["nonzero_reference"]}
                  for (scope, method), distribution in distributions.items()
                  for lower, upper, count in zip(edges, edges[1:], distribution.bins)]).to_csv(args.output / "histogram_bins.csv", index=False)
    (args.output / "metadata.json").write_text(json.dumps({
        "trace": trace.identity(), "history_rounds": [0, 1, 2, 3, 4], "single_entry": {"client": 10, "round": 4},
        "si": "full server SI at start of round 5; all 30 clients, five entries each; restored raw gradient units",
        "si_decode": "SideInfoMatrix with identity target factors; no downstream normalization",
        "histogram": "APE over strictly nonzero reference coordinates, no epsilon; exact-zero errors counted separately",
        "y_axis_max_pct": args.y_max, "columns_per_batch": 8, "rows_per_batch": 200_000,
    }, indent=2) + "\n")
    plot(distributions, args.output, args.y_max)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
