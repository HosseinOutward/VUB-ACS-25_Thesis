"""Validate learned WZ quantizer models on synthetic Gaussian side information."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
import torch.nn.functional as F

from FL_code.FL_core.codec import record_reconstruction_metrics
from FL_code.FL_core.utils import set_global_seed
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import DedupedDecodingWZQuantizerCancer, WZcfgQuant, WZQuantizerCancer


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One quantizer architecture evaluated by this experiment and consumed by its plots."""

    quantizer_class: type[WZQuantizerCancer]
    bins_per_plane: int
    num_planes: int

    @property
    def key(self) -> str:
        """Return the stable CSV and plotting key for this configuration."""
        return f"{self.quantizer_class.__name__}|{self.bins_per_plane}b_{self.num_planes}p"


@dataclass(frozen=True, slots=True)
class PlotStyle:
    """Plot styling assigned to one quantizer configuration."""

    color: tuple[float, float, float]
    marker: str
    label: str


# Keep these settings explicit so model sweeps remain easy to inspect and edit.
NUM_REPEATS = 2
DATA_SIZE = 2_000_000
NOISE_POWER = 0.01
QUANTIZER_SETTINGS = ((6, 2), (2, 2), (2, 1))
QUANTIZER_CLASSES = (WZQuantizerCancer, DedupedDecodingWZQuantizerCancer)
EXPERIMENT_CONFIGS = tuple(
    ExperimentConfig(quantizer_class, *setting)
    for quantizer_class, setting in itertools.product(QUANTIZER_CLASSES, QUANTIZER_SETTINGS)
)
RATE_TYPES = {
    "training_prior_rate": "Training-data prior rate",
    "inference_prior_rate": "Quantizer prior rate",
    "prior_rate": "Retrained prior rate",
    "marginal_rate": "Empirical marginal rate",
}


def append_to_csv(csv_path: Path, record: dict[str, Any]) -> None:
    """Append one result row, creating the CSV header when needed."""
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=record)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def run_single_experiment(config: ExperimentConfig, trial_idx: int) -> dict[str, Any]:
    """Train and evaluate one quantizer and its separately trained symbol prior."""
    seed = trial_idx * 42
    set_global_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    noise_std = float(np.sqrt(NOISE_POWER))
    training_side_information = torch.randn(DATA_SIZE, generator=generator)
    training_target = training_side_information + torch.randn(DATA_SIZE, generator=generator) * noise_std
    inference_side_information = torch.randn(DATA_SIZE, generator=generator)
    inference_target = inference_side_information + torch.randn(DATA_SIZE, generator=generator) * noise_std

    quantizer_config = WZcfgQuant(
        bins_per_plane=config.bins_per_plane,
        num_planes=config.num_planes,
        norm_slices=(slice(None),),
        reconst_ld = 50,
        training_progress_bar=True,
    )
    training_quantizer = config.quantizer_class(quantizer_config, si_size=1)
    training_quantizer.train_model(training_target, [training_side_information])

    quantizer = config.quantizer_class(quantizer_config, si_size=1)
    quantizer.coding_model = training_quantizer.coding_model
    bins, soft_codes, metadata = quantizer.encoding_process(inference_target)
    if soft_codes is None:
        soft_codes = F.one_hot(bins.long(), num_classes=config.bins_per_plane).float()
    quantizer.set_side_information([inference_side_information])
    formatted_side_information = quantizer.side_info_tensor()

    inference_prior = PriorCalculator.compute_prior_from_network(
        quantizer.coding_model, bins, formatted_side_information
    )
    prior_model = PriorCalculator.make_trained_prior_model(
        bins.long(), soft_codes, formatted_side_information, quantizer_config
    )
    retrained_prior = PriorCalculator.compute_prior_from_network(
        prior_model, bins, formatted_side_information
    )
    marginal_prior = PriorCalculator.compute_marginal_prior(
        bins, config.bins_per_plane, config.num_planes
    )
    reconstruction = quantizer.decoding_process((bins, metadata))

    return {
        "config_key": config.key,
        "trial_number": trial_idx,
        "bins_per_plane": config.bins_per_plane,
        "num_planes": config.num_planes,
        "data_size": DATA_SIZE,
        "noise_power": NOISE_POWER,
        "side_info_count": 1,
        "marginal_loss": quantizer_config.marginal_loss,
        "training_prior_rate": training_quantizer.training_prior_rate,
        "inference_prior_rate": PriorCalculator.compute_rate_from_prior_tensor(
            inference_prior, bins, config.num_planes
        ),
        "prior_rate": PriorCalculator.compute_rate_from_prior_tensor(
            retrained_prior, bins, config.num_planes
        ),
        "marginal_rate": PriorCalculator.compute_rate_from_prior_tensor(
            marginal_prior, bins, config.num_planes
        ),
        **record_reconstruction_metrics(inference_target, reconstruction),
    }


def run_experiments(out_path: Path) -> Path:
    """Run every configured quantizer trial, persisting its row and refreshed plot immediately."""
    out_path.mkdir(exist_ok=True, parents=True)
    csv_path = out_path / "quantizer_check.csv"
    total_runs = len(EXPERIMENT_CONFIGS) * NUM_REPEATS

    for current_run, (config, trial_idx) in enumerate(
        itertools.product(EXPERIMENT_CONFIGS, range(NUM_REPEATS)), start=1
    ):
        print(
            f"\n=== {config.quantizer_class.__name__} | {config.bins_per_plane} bins/plane × "
            f"{config.num_planes} planes | trial {trial_idx} ==="
        )
        result = run_single_experiment(config, trial_idx)
        print(
            f"[{current_run}/{total_runs}] MSE={result['mse']:.6f} | "
            f"Quantizer={result['inference_prior_rate']:.3f} | "
            f"Retrained={result['prior_rate']:.3f} | "
            f"Marginal={result['marginal_rate']:.3f}"
        )
        append_to_csv(csv_path, result)
        plot_path = plot_rate_comparison(csv_path)

    print(f"\nResults saved to: {csv_path}")
    return plot_path


def load_records(csv_path: Path) -> list[dict[str, str | float]]:
    """Load quantizer result rows and convert plotted metrics to floats."""
    numeric_fields = {
        "mse", "training_prior_rate", "inference_prior_rate", "prior_rate",
        "marginal_rate", "bins_per_plane", "num_planes",
    }
    with csv_path.open(newline="") as file:
        return [
            {
                key: float(value) if key in numeric_fields else value
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        ]


def compute_theoretical_bounds() -> dict[str, np.ndarray]:
    """Return Gaussian Wyner-Ziv and point-to-point rate-distortion bounds."""
    conditional_variance = NOISE_POWER
    mse_range = np.geomspace(conditional_variance * 1e-4, conditional_variance, 500)
    return {
        "distortion_db": 10 * np.log10(mse_range),
        "wz_rate": 0.5 * np.log2(conditional_variance / mse_range),
        "p2p_rate": 0.5 * np.log2((1 + NOISE_POWER) / mse_range),
    }


def plot_theoretical_bounds(axes: Sequence[Axes], bounds: dict[str, np.ndarray]) -> None:
    """Draw the reference rate-distortion bounds on every supplied axis."""
    for axis in axes:
        axis.plot(bounds["wz_rate"], bounds["distortion_db"], "k-", linewidth=2, label="WZ bound")
        axis.plot(
            bounds["wz_rate"], bounds["distortion_db"] + 1.53,
            "k:", linewidth=2, label="WZ + lattice gap",
        )
        axis.plot(
            bounds["p2p_rate"], bounds["distortion_db"],
            "g--", linewidth=1, alpha=0.5, label="P2P (no SI)",
        )


def plot_rate_comparison(csv_path: Path) -> Path:
    """Plot each quantizer/prior rate and save the figure beside the source CSV."""
    records = load_records(csv_path)
    config_keys = sorted({str(record["config_key"]) for record in records})
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "v", "D", "P")
    styles = {
        key: PlotStyle(colors[index % len(colors)], markers[index % len(markers)], key)
        for index, key in enumerate(config_keys)
    }

    figure, axes_array = plt.subplots(2, 2, figsize=(14, 11))
    axes = tuple(axes_array.flatten())
    plot_theoretical_bounds(axes, compute_theoretical_bounds())

    for axis, (rate_key, title) in zip(axes, RATE_TYPES.items(), strict=True):
        for config_key, grouped_records in itertools.groupby(
            sorted(records, key=lambda record: str(record["config_key"])),
            key=lambda record: str(record["config_key"]),
        ):
            group = list(grouped_records)
            style = styles[config_key]
            axis.scatter(
                [float(record[rate_key]) for record in group],
                [10 * np.log10(float(record["mse"])) for record in group],
                color=style.color,
                marker=style.marker,
                s=60,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                label=style.label,
            )
        axis.set(xlabel="Rate (bits/symbol)", ylabel="Distortion (dB)", title=title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="best")

    figure.tight_layout()
    plot_path = csv_path.with_suffix(".png")
    figure.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def main() -> None:
    """Run the configured experiment unless skipped, then plot available results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true", help="Skip experiments and only plot existing results")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "results"
    csv_path = out_dir / "quantizer_check.csv"
    if args.skip_run:
        assert csv_path.exists(), f"No quantizer results found at {csv_path}."
        plot_path = plot_rate_comparison(csv_path)
    else:
        plot_path = run_experiments(out_dir)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
