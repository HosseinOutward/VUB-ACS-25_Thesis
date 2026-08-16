"""Plot the simple side-information sign-classification experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "results"
DEFAULT_CSV: Final[Path] = RESULTS_DIR / "sign_predictability.csv"
DEFAULT_PLOT: Final[Path] = RESULTS_DIR / "sign_predictability.png"
MODEL_STYLE: Final[dict[str, tuple[str, str]]] = {
    "own_history": ("Own history", "#2a78d6"),
    "decoder_si": ("All decoder SI", "#eb6834"),
}
METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("balanced_accuracy", "Balanced accuracy"),
    ("cross_entropy_bits", "Cross entropy (bits/symbol)"),
)
REQUIRED_COLUMNS: Final[set[str]] = {
    "round_id", "client_id", "model", *(metric for metric, _ in METRICS),
}


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_PLOT)
    return parser.parse_args()


def plot_results(csv_path: Path, output_path: Path) -> None:
    """Plot per-round client means and ranges for the two side-information conditions."""
    assert csv_path.is_file(), f"Missing classifier results: {csv_path}."
    results = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(results.columns)
    assert not missing, f"{csv_path} is missing columns: {sorted(missing)}."
    unknown = set(results["model"]) - set(MODEL_STYLE)
    assert not unknown, f"{csv_path} contains unknown model labels: {sorted(unknown)}."

    figure, axes = plt.subplots(1, len(METRICS), figsize=(11, 4.2), constrained_layout=True)
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        for model, (model_label, color) in MODEL_STYLE.items():
            grouped = results[results["model"].eq(model)].groupby("round_id")[metric]
            summary = grouped.agg(["mean", "min", "max"])
            assert not summary.empty, f"{csv_path} contains no {model!r} rows."
            rounds = summary.index.to_numpy()
            axis.fill_between(rounds, summary["min"], summary["max"], color=color, alpha=0.15)
            axis.plot(rounds, summary["mean"], color=color, marker="o", linewidth=2, label=model_label)
        axis.set_xlabel("Target round")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False)
    figure.suptitle("Can decoder-side information predict the next client-delta sign?")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {output_path}")


def main() -> None:
    """Render the configured results CSV."""
    args = parse_args()
    plot_results(args.csv, args.out)


if __name__ == "__main__":
    main()
