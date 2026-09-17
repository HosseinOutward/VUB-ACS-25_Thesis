"""Validate the production WZ protocol on paired synthetic Gaussian rounds."""

from __future__ import annotations

import argparse
import ast
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol, simulate_compression
from FL_code.FL_core.utils import StateDictManager, set_global_seed
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol
from FL_code.cancer_protocol.history import ReconstructionHistory


CodecName = Literal["O", "T"]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One quantizer architecture evaluated by this experiment and consumed by its plots."""

    bins_per_plane: int
    num_planes: int
    reconst_ld: float


# Keep these settings explicit so model sweeps remain easy to inspect and edit.
NUM_REPEATS = 2
DATA_SIZE = 2_000_000
NOISE_POWER = 0.01
TRAIN_ROUND_ID = 1
CLIENT_ID = 0
CODEC_NAMES: tuple[CodecName, ...] = ("O", "T")
QUANTIZER_SETTINGS = (  # (bins_per_plane, num_planes, reconst_ld)
    (2, 3, 120.0),
    (4, 2, 160.0),
)
EXPERIMENT_CONFIGS = tuple(ExperimentConfig(*setting) for setting in QUANTIZER_SETTINGS)
RATE_TYPES = {
    "training_prior_rate": "Jointly trained prior rate",
    "prior_rate": "Retrained prior rate",
    "marginal_rate": "Empirical marginal rate",
}


def append_to_csv(csv_path: Path, record: Mapping[str, Any]) -> None:
    """Append one result row, creating the CSV header when needed."""
    file_exists = csv_path.exists()
    if file_exists:
        with csv_path.open(newline="") as file:
            assert next(csv.reader(file)) == list(record), f"CSV schema mismatch in {csv_path}."
    with csv_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=record)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def make_round_token(codec_name: CodecName, config: ExperimentConfig) -> str:
    """Return the configured Oracle or Temporal training token used by the real protocol."""
    options: dict[str, Any] = {
        "bins_per_plane": config.bins_per_plane,
        "num_planes": config.num_planes,
        "reconst_ld": config.reconst_ld,
        "train_epochs": 180,
        "train_sample_size": 200_000,
        "train_batch_size": 1_000,
        "lr": 3e-3,
        "lr_step": 40,
        "tau": 1.0,
        "training_progress_bar": False,
        "training_log_every_epochs": 5,
        "quantizer_train_repeats": 1,
        "shared_heads": True,
        "prior_context": "hard",
        "prior_probabilities": "categorical",
        "fused_optimizer": True,
        "hidden_dim": 100,
        "layers": 3,
    }
    return codec_name + "".join(f"|{name}={value!r}" for name, value in options.items())


def populated_protocol(
    codec_name: CodecName,
    config: ExperimentConfig,
    training_side_information: torch.Tensor,
) -> WZCancerProtocol:
    """Create a scheduled production protocol with one decoder-visible synthetic history entry."""
    protocol = create_protocol("wz_cancer", sd_slices=(slice(None),))
    assert isinstance(protocol, WZCancerProtocol)
    protocol.set_schedule(TRAIN_ROUND_ID, (make_round_token(codec_name, config), "F"))

    history = ReconstructionHistory(max_per_client=1)
    history.commit(
        training_side_information,
        CompressionRecord(TRAIN_ROUND_ID - 1, CLIENT_ID, "synthetic_history"),
        Access.TEMPORAL_TOO,
    )
    protocol.max_per_client_recons_history = history.max_per_client
    protocol._recons_history = history
    protocol.histories = (history,)
    return protocol


def parse_stage_rates(record: Mapping[str, str], field: str, stages: int) -> list[float]:
    """Parse one staged metric emitted by ``CompressionRecord.append_record_to_csv``."""
    values = ast.literal_eval(record[field])
    assert isinstance(values, list) and len(values) == stages, (
        f"Protocol record field {field!r} has {len(values) if isinstance(values, list) else type(values)} "
        f"values; expected {stages}.")
    return [float(value) for value in values]


def run_single_experiment(
    config: ExperimentConfig,
    trial_idx: int,
    run_label: str,
    codec_name: CodecName,
    sd_manager: StateDictManager,
) -> list[dict[str, Any]]:
    """Run one trained round and one held-out frozen round through the production endpoint."""
    seed = trial_idx * 42
    generator = torch.Generator().manual_seed(seed)
    noise_std = float(np.sqrt(NOISE_POWER))
    training_side_information = torch.randn(DATA_SIZE, generator=generator)
    training_target = training_side_information + torch.randn(DATA_SIZE, generator=generator) * noise_std
    inference_side_information = torch.randn(DATA_SIZE, generator=generator)
    inference_target = inference_side_information + torch.randn(DATA_SIZE, generator=generator) * noise_std

    set_global_seed(seed)
    protocol = populated_protocol(codec_name, config, training_side_information)
    with TemporaryDirectory(prefix="brent_quantizer_check_") as temporary_dir:
        records_dir = Path(temporary_dir)
        simulate_compression(
            protocol,
            training_target,
            CLIENT_ID,
            TRAIN_ROUND_ID,
            sd_manager,
            records_dir,
            {},
            (),
            (),
        )

        # Preserve the real frozen-round transition while supplying the independent paired SI
        # that defines this synthetic held-out problem.
        protocol._recons_history.replace_latest(
            inference_side_information,
            round_id=TRAIN_ROUND_ID,
            client_id=CLIENT_ID,
        )
        simulate_compression(
            protocol,
            inference_target,
            CLIENT_ID,
            TRAIN_ROUND_ID + 1,
            sd_manager,
            records_dir,
            {},
            (),
            (),
        )
        with (records_dir / "compression_records.csv").open(newline="") as file:
            protocol_records = list(csv.DictReader(file))

    assert len(protocol_records) == 2
    assert [int(record["round_id"]) for record in protocol_records] == [
        TRAIN_ROUND_ID,
        TRAIN_ROUND_ID + 1,
    ]

    results: list[dict[str, Any]] = []
    for phase, record in zip(("training", "frozen"), protocol_records, strict=True):
        stage_fields = {
            field: list(itertools.accumulate(parse_stage_rates(record, field, config.num_planes)))
            for field in (
                "training_prior_stage_rates",
                "prior_stage_rates",
                "marginal_stage_rates",
            )
        }
        for stage, mse in enumerate(parse_stage_rates(record, "stage_mses", config.num_planes)):
            training_prior_rate = stage_fields["training_prior_stage_rates"][stage]
            results.append({
                "trial_number": trial_idx,
                "codec": codec_name,
                "phase": phase,
                "round_id": int(record["round_id"]),
                "round_name_full": record["round_name_full"],
                "bins_per_plane": config.bins_per_plane,
                "num_planes": stage + 1,
                "model_num_planes": config.num_planes,
                "reconst_ld": config.reconst_ld,
                "data_size": DATA_SIZE,
                "noise_power": NOISE_POWER,
                "side_info_count": 1,
                "training_prior_rate": training_prior_rate,
                "prior_rate": stage_fields["prior_stage_rates"][stage],
                "marginal_rate": stage_fields["marginal_stage_rates"][stage],
                "mse": mse,
                "training_prior_wz_gap_db": 10 * np.log10(
                    mse / (NOISE_POWER * 2 ** (-2 * training_prior_rate))
                ),
                "run_label": run_label,
            })
    return results


def run_experiments(
    out_path: Path,
    run_label: str = "protocol_two_round",
    codec_names: Sequence[CodecName] = ("O",),
) -> Path:
    """Run every selected production-protocol trial and refresh the comparison plot."""
    assert run_label and "|" not in run_label
    assert codec_names and set(codec_names) <= set(CODEC_NAMES)
    out_path.mkdir(exist_ok=True, parents=True)
    csv_path = out_path / "protocol_quantizer_check.csv"
    model = torch.nn.Module()
    model.register_parameter("synthetic_vector", torch.nn.Parameter(torch.empty(DATA_SIZE)))
    sd_manager = StateDictManager(model)
    cases = tuple(itertools.product(codec_names, EXPERIMENT_CONFIGS, range(NUM_REPEATS)))

    for current_run, (codec_name, config, trial_idx) in enumerate(cases, start=1):
        print(
            f"\n=== codec={codec_name} | {config.bins_per_plane} bins/plane × "
            f"{config.num_planes} planes | trial {trial_idx} ==="
        )
        results = run_single_experiment(
            config, trial_idx, run_label, codec_name, sd_manager)
        for result in results:
            print(
                f"[{current_run}/{len(cases)}] phase={result['phase']} | "
                f"stage={result['num_planes']} | MSE={result['mse']:.6f} | "
                f"Training prior={result['training_prior_rate']:.3f} | "
                f"WZ gap={result['training_prior_wz_gap_db']:.2f} dB | "
                f"Retrained={result['prior_rate']:.3f} | Marginal={result['marginal_rate']:.3f}"
            )
            append_to_csv(csv_path, result)
        plot_path = plot_rate_comparison(csv_path)

    print(f"\nResults saved to: {csv_path}")
    return plot_path


def plot_rate_comparison(csv_path: Path) -> Path:
    """Plot training and held-out protocol rounds for every selected codec and architecture."""
    numeric_fields = {
        "trial_number", "round_id", "bins_per_plane", "num_planes", "model_num_planes",
        "reconst_ld", "mse", *RATE_TYPES,
    }
    with csv_path.open(newline="") as file:
        records = [
            {key: float(value) if key in numeric_fields else value for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
    architecture_keys = sorted({
        (record["bins_per_plane"], record["model_num_planes"], record["reconst_ld"])
        for record in records
    })
    colors = plt.get_cmap("tab10").colors
    multiple_run_labels = len({record["run_label"] for record in records}) > 1

    figure, axes_array = plt.subplots(1, 3, figsize=(18, 5.5))
    axes = tuple(axes_array.flatten())
    minimum_mse = min(cast(float, record["mse"]) for record in records) / 10**0.2
    mse_range = np.geomspace(min(minimum_mse, NOISE_POWER * 0.1), NOISE_POWER, 500)
    bounds = 10 * np.log10(mse_range)
    for axis in axes:
        axis.plot(0.5 * np.log2(NOISE_POWER / mse_range), bounds, "k-", linewidth=2, label="WZ bound")
        axis.plot(0.5 * np.log2(NOISE_POWER / mse_range), bounds + 1.53,
                  "k:", linewidth=2, label="WZ + lattice gap")
        axis.plot(0.5 * np.log2((1 + NOISE_POWER) / mse_range), bounds,
                  "g--", linewidth=1, alpha=0.5, label="P2P (no SI)")
    group_fields = (
        "run_label", "codec", "phase", "bins_per_plane", "model_num_planes", "reconst_ld",
        "trial_number",
    )
    grouped_records = itertools.groupby(
        sorted(records, key=lambda record: tuple(record[field] for field in group_fields)),
        key=lambda record: tuple(record[field] for field in group_fields),
    )
    groups = [(key, sorted(group, key=lambda record: cast(float, record["num_planes"])))
              for key, group in grouped_records]

    for axis, (rate_key, title) in zip(axes, RATE_TYPES.items(), strict=True):
        labelled: set[tuple[str, str, str, float, float, float]] = set()
        for key, group in groups:
            run_label, codec, phase, bins, planes, reconst_ld, _ = key
            style_key = (str(run_label), str(codec), str(phase), bins, planes, reconst_ld)
            architecture = (bins, planes, reconst_ld)
            run_prefix = f"{run_label} | " if multiple_run_labels else ""
            color = colors[architecture_keys.index(architecture) % len(colors)]
            frozen = phase == "frozen"
            marker = {"O": ("o", "D"), "T": ("s", "^")}[str(codec)][frozen]
            axis.plot(
                [float(record[rate_key]) for record in group],
                [10 * np.log10(float(record["mse"])) for record in group],
                color=color,
                marker=marker,
                markerfacecolor="none" if frozen else color,
                markeredgewidth=1.8 if frozen else 0.8,
                linestyle="--" if frozen else "-",
                linewidth=1.8,
                markersize=8 if frozen else 6,
                zorder=4 if frozen else 3,
                label=(
                    f"{run_prefix}{codec} {phase} | {int(cast(float, bins))}b × "
                    f"{int(cast(float, planes))}p, λ={reconst_ld:g}"
                    if style_key not in labelled else None
                ),
            )
            labelled.add(style_key)
        axis.set(xlabel="Rate (bits/symbol)", ylabel="Distortion (dB)", title=title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc="upper right")

    figure.tight_layout()
    plot_path = csv_path.with_suffix(".png")
    figure.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return plot_path


def main() -> None:
    """Run the selected protocol codecs unless skipped, then plot available results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true", help="Skip experiments and only plot existing results")
    parser.add_argument("--run-label", default="protocol_two_round")
    parser.add_argument("--codec", choices=("O", "T", "both"), default="O")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "results"
    csv_path = out_dir / "protocol_quantizer_check.csv"
    if args.skip_run:
        assert csv_path.exists(), f"No protocol quantizer results found at {csv_path}."
        plot_path = plot_rate_comparison(csv_path)
    else:
        codec_names = CODEC_NAMES if args.codec == "both" else (cast(CodecName, args.codec),)
        plot_path = run_experiments(out_dir, args.run_label, codec_names)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
