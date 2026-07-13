"""Tune the 222 quantizer on saved temporal reconstruction-to-delta pairs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.utils import StateDictManager, set_global_seed
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant, WZQuantizerCancer
from FL_code.run_fl import FLConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "_/data"
RAW_DIR = DATA_ROOT / "federated_learning/client_deltas"
RECON_DIR = DATA_ROOT / "cancer_protocol/T_round_checkpoints"
RESULTS_DIR = Path(__file__).parent / "results/real_temporal_si_222_taguchi"
TRAIN_SI_ROUND = 15
INFERENCE_SI_ROUND = 16
TRAIN_CLIENT = 0
SCREEN_INFERENCE_CLIENTS = (0,)
CONFIRMATION_INFERENCE_CLIENTS = tuple(range(4))
BINS_PER_PLANE = 2
NUM_PLANES = 3
TRAIN_SAMPLE_SIZE = 300_000
TRAIN_BATCH_SIZE = 50_000

LEVELS: dict[str, tuple[float | int, ...]] = {
    "lambda": (100.0, 300.0, 900.0),
    "tau": (0.3, 1.0, 3.0),
    "lr": (3e-4, 1e-3, 3e-3),
    "epochs": (120, 180, 240),
}
L9 = (
    (0, 0, 0, 0),
    (0, 1, 1, 1),
    (0, 2, 2, 2),
    (1, 0, 1, 2),
    (1, 1, 2, 0),
    (1, 2, 0, 1),
    (2, 0, 2, 1),
    (2, 1, 0, 2),
    (2, 2, 1, 0),
)


@dataclass(frozen=True, slots=True)
class TrialParams:
    """One longer-training L9 configuration consumed by the real-data screen."""

    lambda_: float
    tau: float
    lr: float
    epochs: int


@dataclass(frozen=True, slots=True)
class RDPoint:
    """One aggregated held-out inference stage produced by a trained trial."""

    trial: int
    seed: int
    stage: int
    lambda_: float
    tau: float
    lr: float
    epochs: int
    rate: float
    mse: float
    distortion_db: float
    reference_mse: float
    rd_gap_db: float
    baseline_si_mse: float
    seconds: float


def taguchi_trials() -> tuple[TrialParams, ...]:
    """Expand the L9 level indices into concrete longer-training configurations."""
    keys = tuple(LEVELS)
    return tuple(
        TrialParams(
            lambda_=float(LEVELS[keys[0]][row[0]]),
            tau=float(LEVELS[keys[1]][row[1]]),
            lr=float(LEVELS[keys[2]][row[2]]),
            epochs=int(LEVELS[keys[3]][row[3]]),
        )
        for row in L9
    )


def load_tensor(directory: Path, round_id: int, client_id: int) -> torch.Tensor:
    """Load one saved CPU float32 delta or reconstruction tensor."""
    path = directory / f"round_{round_id}_client_{client_id}.pt"
    assert path.is_file(), f"Missing temporal experiment tensor: {path}."
    tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
    return tensor


def model_slices() -> tuple[slice, ...]:
    """Recover parameter boundaries used by the producing ResNet-18 FL run."""
    config_path = RECON_DIR / "fl_config.json"
    cfg = FLConfig.model_validate_json(config_path.read_text())
    manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
    return tuple(manager.get_slices())


def pareto_front(points: list[RDPoint]) -> list[RDPoint]:
    """Return held-out points not weakly dominated in both rate and MSE."""
    return sorted(
        (
            point for point in points
            if not any(
                other.rate <= point.rate
                and other.mse <= point.mse
                and (other.rate < point.rate or other.mse < point.mse)
                for other in points
            )
        ),
        key=lambda point: point.rate,
    )


def supported_front(points: list[RDPoint]) -> list[RDPoint]:
    """Return the lower convex envelope of nondominated held-out RD points."""
    hull: list[RDPoint] = []
    for point in pareto_front(points):
        while len(hull) >= 2:
            left = (hull[-1].mse - hull[-2].mse) / (hull[-1].rate - hull[-2].rate)
            right = (point.mse - hull[-1].mse) / (point.rate - hull[-1].rate)
            if right > left:
                break
            hull.pop()
        hull.append(point)
    return hull


def inference_baseline_mse(client_ids: tuple[int, ...]) -> float:
    """Return mean held-out error from using the previous reconstruction directly."""
    return float(np.mean([
        F.mse_loss(
            load_tensor(RECON_DIR, INFERENCE_SI_ROUND, client_id),
            load_tensor(RAW_DIR, INFERENCE_SI_ROUND + 1, client_id),
        ).item()
        for client_id in client_ids
    ]))


def run_trial(
    params: TrialParams,
    trial: int,
    seed: int,
    norm_slices: tuple[slice, ...],
    baseline_mse: float,
    inference_clients: tuple[int, ...],
) -> list[RDPoint]:
    """Train on round 15→16 and aggregate full-vector inference over round 16→17."""
    set_global_seed(seed)
    training_target = load_tensor(RAW_DIR, TRAIN_SI_ROUND + 1, TRAIN_CLIENT)
    training_si = load_tensor(RECON_DIR, TRAIN_SI_ROUND, TRAIN_CLIENT)
    config = WZcfgQuant(
        bins_per_plane=BINS_PER_PLANE,
        num_planes=NUM_PLANES,
        norm_slices=norm_slices,
        reconst_ld=params.lambda_,
        tau=params.tau,
        lr=params.lr,
        train_epochs=params.epochs,
        train_sample_size=TRAIN_SAMPLE_SIZE,
        train_batch_size=TRAIN_BATCH_SIZE,
        lr_step=max(1, params.epochs // 3),
        quantizer_train_repeats=1,
        fused_optimizer=True,
        mixed_precision=True,
        tf32=True,
        training_progress_bar=False,
        training_log_every_epochs=params.epochs,
    )

    started = perf_counter()
    training_quantizer = WZQuantizerCancer(config, si_size=1)
    training_quantizer.train_model(training_target, [training_si])
    stage_rates: list[list[float]] = [[] for _ in range(NUM_PLANES)]
    stage_mses: list[list[float]] = [[] for _ in range(NUM_PLANES)]

    for client_id in inference_clients:
        target = load_tensor(RAW_DIR, INFERENCE_SI_ROUND + 1, client_id)
        side_information = load_tensor(RECON_DIR, INFERENCE_SI_ROUND, client_id)
        quantizer = WZQuantizerCancer(config, si_size=1)
        quantizer.coding_model = training_quantizer.coding_model
        bins, _, metadata = quantizer.encoding_process(target)
        quantizer.set_side_information([side_information])
        formatted_si = quantizer.side_info_tensor(metadata)
        prior = PriorCalculator.compute_prior_from_network(
            quantizer.coding_model, bins, formatted_si
        )
        plane_rates = PriorCalculator.compute_plane_rates(prior, bins, NUM_PLANES)
        reconstructions = quantizer.decoding_stages_process((bins, metadata))
        for stage, reconstruction in enumerate(reconstructions):
            stage_rates[stage].append(float(sum(plane_rates[:stage + 1])))
            stage_mses[stage].append(F.mse_loss(reconstruction, target).item())

    elapsed = perf_counter() - started
    points: list[RDPoint] = []
    for stage in range(NUM_PLANES):
        rate = float(np.mean(stage_rates[stage]))
        mse = float(np.mean(stage_mses[stage]))
        reference_mse = baseline_mse * 2 ** (-2 * rate)
        points.append(RDPoint(
            trial=trial,
            seed=seed,
            stage=stage + 1,
            lambda_=params.lambda_,
            tau=params.tau,
            lr=params.lr,
            epochs=params.epochs,
            rate=rate,
            mse=mse,
            distortion_db=10 * math.log10(mse),
            reference_mse=reference_mse,
            rd_gap_db=10 * math.log10(mse / reference_mse),
            baseline_si_mse=baseline_mse,
            seconds=elapsed,
        ))
    return points


def write_results(points: list[RDPoint], out_dir: Path) -> None:
    """Persist real-data RD points, supported-front metadata, and a plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rd_points.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=asdict(points[0]))
        writer.writeheader()
        writer.writerows(asdict(point) for point in points)

    final_points = [point for point in points if point.stage == NUM_PLANES]
    nondominated = pareto_front(final_points)
    supported = supported_front(final_points)
    selected = min(supported, key=lambda point: point.rd_gap_db)
    summary: dict[str, Any] = {
        "selection_rule": (
            "Discard final-stage Pareto-dominated and unsupported points, then minimize "
            "gap to the empirical conditional-Gaussian reference D0*2^(-2R). D0 is the "
            "held-out MSE from directly using the decoder reconstruction as prediction."
        ),
        "training_pair": {"reconstruction_round": TRAIN_SI_ROUND, "raw_target_round": TRAIN_SI_ROUND + 1, "client": TRAIN_CLIENT},
        "inference_pairs": {"reconstruction_round": INFERENCE_SI_ROUND, "raw_target_round": INFERENCE_SI_ROUND + 1, "screen_clients": list(SCREEN_INFERENCE_CLIENTS), "confirmation_clients": list(CONFIRMATION_INFERENCE_CLIENTS)},
        "quantizer": {"bins_per_plane": BINS_PER_PLANE, "num_planes": NUM_PLANES},
        "selected": asdict(selected),
        "nondominated_trials": [point.trial for point in nondominated],
        "supported_trials": [point.trial for point in supported],
    }
    summary_path = out_dir / "summary.json"
    if (out_dir / "confirmation_points.csv").exists() and summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if "confirmation" in previous:
            summary["confirmation"] = previous["confirmation"]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(8, 6))
    rates = np.linspace(0, NUM_PLANES, 400)
    baseline = points[0].baseline_si_mse
    axis.plot(rates, 10 * np.log10(baseline * 2 ** (-2 * rates)), "k--", label="conditional Gaussian reference")
    for stage, marker in zip(range(1, NUM_PLANES + 1), ("o", "s", "^"), strict=True):
        stage_points = [point for point in points if point.stage == stage]
        axis.scatter([point.rate for point in stage_points], [point.distortion_db for point in stage_points], marker=marker, label=f"stage {stage}")
    axis.plot([point.rate for point in supported], [point.distortion_db for point in supported], color="tab:red", linewidth=2, label="final-stage convex front")
    axis.scatter([selected.rate], [selected.distortion_db], s=160, facecolors="none", edgecolors="black", linewidths=2)
    axis.set(xlabel="Conditional rate (bits/symbol)", ylabel="Held-out MSE (dB)", title="Real temporal SI, learned 222 quantizer")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "rd_front.png", dpi=200)
    plt.close(figure)


def confirm_finalists(
    points: list[RDPoint],
    out_dir: Path,
    norm_slices: tuple[slice, ...],
    baseline_mse: float,
) -> dict[str, Any]:
    """Retrain the two closest supported finalists on paired fresh seeds."""
    finalists = sorted(
        supported_front([point for point in points if point.stage == NUM_PLANES]),
        key=lambda point: point.rd_gap_db,
    )[:2]
    params_by_trial = {trial: params for trial, params in enumerate(taguchi_trials(), start=1)}
    confirmation = [
        point
        for seed in (20_001, 20_002)
        for finalist in finalists
        for point in run_trial(
            params_by_trial[finalist.trial], finalist.trial, seed, norm_slices,
            baseline_mse, CONFIRMATION_INFERENCE_CLIENTS,
        )
    ]
    with (out_dir / "confirmation_points.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=asdict(confirmation[0]))
        writer.writeheader()
        writer.writerows(asdict(point) for point in confirmation)

    aggregates = []
    for finalist in finalists:
        replicates = [point for point in confirmation if point.trial == finalist.trial and point.stage == NUM_PLANES]
        aggregates.append({
            "trial": finalist.trial,
            "params": asdict(params_by_trial[finalist.trial]),
            "mean_rate": float(np.mean([point.rate for point in replicates])),
            "mean_mse": float(np.mean([point.mse for point in replicates])),
            "mean_rd_gap_db": float(np.mean([point.rd_gap_db for point in replicates])),
            "replicate_rd_gap_db": [point.rd_gap_db for point in replicates],
        })
    aggregates.sort(key=lambda result: result["mean_rd_gap_db"])
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["confirmation"] = {"paired_seeds": [20_001, 20_002], "finalists": aggregates, "selected": aggregates[0]}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return aggregates[0]


def load_results(csv_path: Path) -> list[RDPoint]:
    """Load persisted real-data RD points without retraining."""
    with csv_path.open(newline="") as file:
        return [RDPoint(
            trial=int(row["trial"]), seed=int(row["seed"]), stage=int(row["stage"]),
            lambda_=float(row["lambda_"]), tau=float(row["tau"]), lr=float(row["lr"]),
            epochs=int(row["epochs"]), rate=float(row["rate"]), mse=float(row["mse"]),
            distortion_db=float(row["distortion_db"]), reference_mse=float(row["reference_mse"]),
            rd_gap_db=float(row["rd_gap_db"]), baseline_si_mse=float(row["baseline_si_mse"]),
            seconds=float(row["seconds"]),
        ) for row in csv.DictReader(file)]


def main() -> None:
    """Run, confirm, or reanalyze the longer real temporal-SI screen."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--confirm-only", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--single-epochs", type=int)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    assert sum((args.analyze_only, args.confirm_only, args.screen_only, args.single_epochs is not None)) <= 1
    csv_path = args.out_dir / "rd_points.csv"
    slices = model_slices()
    screen_baseline_mse = inference_baseline_mse(SCREEN_INFERENCE_CLIENTS)
    if args.single_epochs is not None:
        assert args.single_epochs > 0
        params = TrialParams(lambda_=300.0, tau=1.0, lr=3e-3, epochs=args.single_epochs)
        points = run_trial(
            params, 0, 30_000 + args.single_epochs, slices,
            screen_baseline_mse, SCREEN_INFERENCE_CLIENTS,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        result_path = args.out_dir / f"single_{args.single_epochs}_epochs.csv"
        with result_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=asdict(points[0]))
            writer.writeheader()
            writer.writerows(asdict(point) for point in points)
        print(f"\nSingle-run result saved to {result_path}: {asdict(points[-1])}")
        return
    if args.analyze_only or args.confirm_only:
        assert csv_path.exists(), f"No saved real-data results at {csv_path}."
        points = load_results(csv_path)
    else:
        (args.out_dir / "confirmation_points.csv").unlink(missing_ok=True)
        points = []
        for trial, params in enumerate(taguchi_trials(), start=1):
            print(f"\n=== real-SI L9 trial {trial}/{len(L9)}: {params} ===", flush=True)
            trial_points = run_trial(
                params, trial, 10_000 + trial, slices,
                screen_baseline_mse, SCREEN_INFERENCE_CLIENTS,
            )
            points.extend(trial_points)
            final = trial_points[-1]
            print(f"trial {trial}: R={final.rate:.4f}, MSE={final.mse:.6g}, reference gap={final.rd_gap_db:.2f} dB, seconds={final.seconds:.1f}", flush=True)
            write_results(points, args.out_dir)
    write_results(points, args.out_dir)
    if not (args.analyze_only or args.screen_only):
        selected = confirm_finalists(
            points, args.out_dir, slices,
            inference_baseline_mse(CONFIRMATION_INFERENCE_CLIENTS),
        )
        print(f"\nConfirmed real-SI parameters: {selected}")
    elif args.screen_only:
        selected = json.loads((args.out_dir / "summary.json").read_text())["selected"]
        print(f"\nScreen-selected real-SI parameters: {selected}")


if __name__ == "__main__":
    main()
