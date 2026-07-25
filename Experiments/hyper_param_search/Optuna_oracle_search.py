"""Refine distinct fixed-codec Oracle quantizer basins with independent Optuna studies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import optuna
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.hyper_param_search.Taguchi_oracle_search import (
    Level,
    Trial as SearchTrial,
    model_slices,
    raw_history,
    run_trial,
)


RESULTS_DIR = Path(__file__).parent / "results/round14_oracle_optuna"
DEFAULT_TRIALS = 30


@dataclass(frozen=True, slots=True)
class Basin:
    """One viable Taguchi row used to initialize an independent TPE search."""

    name: str
    source_trial: int
    center: dict[str, Level]


def basins() -> tuple[Basin, ...]:
    """Return the distinct viable fixed-codec starting regions."""
    return (
        Basin("high_lambda_add", 26, {
            "reconst_ld": 900.0, "tau": 0.3, "lr": 3e-3,
            "train_epochs": 80, "train_sample_size": 600_000,
            "train_batch_size": 25_000, "lr_step_fraction": 0.25,
            "hidden_dim": 100, "layers": 4, "source_perturbation": "add",
        }),
        Basin("low_lambda_add", 9, {
            "reconst_ld": 100.0, "tau": 3.0, "lr": 3e-3,
            "train_epochs": 120, "train_sample_size": 300_000,
            "train_batch_size": 25_000, "lr_step_fraction": 0.5,
            "hidden_dim": 50, "layers": 4, "source_perturbation": "add",
        }),
        Basin("mid_lambda_draw", 15, {
            "reconst_ld": 300.0, "tau": 0.3, "lr": 9e-3,
            "train_epochs": 120, "train_sample_size": 300_000,
            "train_batch_size": 25_000, "lr_step_fraction": 0.25,
            "hidden_dim": 150, "layers": 3, "source_perturbation": "draw_discard",
        }),
        Basin("low_lambda_none", 2, {
            "reconst_ld": 100.0, "tau": 1.0, "lr": 9e-3,
            "train_epochs": 40, "train_sample_size": 300_000,
            "train_batch_size": 100_000, "lr_step_fraction": 0.25,
            "hidden_dim": 100, "layers": 4, "source_perturbation": "none",
        }),
    )


def directional_bounds(
    center: int | float,
    levels: tuple[int | float, int | float, int | float],
    outer: tuple[int | float, int | float],
) -> tuple[int | float, int | float]:
    """Expand outward at endpoint levels and span the original range at the middle."""
    level_index = levels.index(center)
    return (
        (outer[0], levels[1]),
        (levels[0], levels[2]),
        (levels[1], outer[1]),
    )[level_index]


def suggest_float(
    trial: optuna.Trial,
    basin: Basin,
    name: str,
    levels: tuple[float, float, float],
    outer: tuple[float, float],
    *,
    log: bool = False,
) -> float:
    """Suggest a continuous value within the basin's direction-aware bounds."""
    low, high = directional_bounds(float(basin.center[name]), levels, outer)
    return trial.suggest_float(name, float(low), float(high), log=log)


def suggest_int(
    trial: optuna.Trial,
    basin: Basin,
    name: str,
    levels: tuple[int, int, int],
    outer: tuple[int, int],
    step: int,
) -> int:
    """Suggest an integer value within the basin's direction-aware bounds."""
    low, high = directional_bounds(int(basin.center[name]), levels, outer)
    return trial.suggest_int(name, int(low), int(high), step=step)


def suggest_params(trial: optuna.Trial, basin: Basin) -> dict[str, Level]:
    """Sample all RD-relevant parameters while keeping codec structure fixed."""
    batch_low, batch_high = directional_bounds(
        int(basin.center["train_batch_size"]),
        (25_000, 50_000, 100_000),
        (12_500, 200_000),
    )
    batch_choices = tuple(
        value for value in (12_500, 25_000, 50_000, 100_000, 200_000)
        if int(batch_low) <= value <= int(batch_high)
    )
    return {
        "reconst_ld": suggest_float(
            trial, basin, "reconst_ld", (100.0, 300.0, 900.0), (30.0, 2700.0), log=True),
        "tau": suggest_float(
            trial, basin, "tau", (0.3, 1.0, 3.0), (0.1, 10.0), log=True),
        "lr": suggest_float(
            trial, basin, "lr", (1e-3, 3e-3, 9e-3), (3e-4, 2e-2), log=True),
        "train_epochs": suggest_int(
            trial, basin, "train_epochs", (40, 80, 120), (20, 200), 20),
        "train_sample_size": suggest_int(
            trial, basin, "train_sample_size",
            (100_000, 300_000, 600_000), (50_000, 1_200_000), 50_000,
        ),
        "train_batch_size": trial.suggest_categorical("train_batch_size", batch_choices),
        "lr_step_fraction": suggest_float(
            trial, basin, "lr_step_fraction", (0.25, 0.5, 1.0), (0.1, 1.5)),
        "hidden_dim": suggest_int(
            trial, basin, "hidden_dim", (50, 100, 150), (26, 250), 2),
        "layers": suggest_int(
            trial, basin, "layers", (2, 3, 4), (1, 6), 1),
        "source_perturbation": trial.suggest_categorical(
            "source_perturbation", ("none", "draw_discard", "add")),
    }


def study_path(output_dir: Path, basin: Basin) -> Path:
    """Return the SQLite path for one independent basin study."""
    return output_dir / f"{basin.name}.db"


def write_summary(study: optuna.Study, basin: Basin, output_dir: Path) -> None:
    """Persist a readable basin summary beside its Optuna database."""
    complete = [
        trial for trial in study.trials if trial.state is optuna.trial.TrialState.COMPLETE
    ]
    ordered = sorted(complete, key=lambda trial: float(trial.value))
    summary = {
        "basin": asdict(basin),
        "completed_trials": len(complete),
        "best": {
            "number": study.best_trial.number,
            "objective_rate_gap": study.best_value,
            "params": study.best_params,
            "result": study.best_trial.user_attrs["result"],
        },
        "top_trials": [
            {
                "number": trial.number,
                "objective_rate_gap": trial.value,
                "params": trial.params,
            }
            for trial in ordered[:10]
        ],
    }
    (output_dir / f"{basin.name}.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_study(basin: Basin, output_dir: Path, target_trials: int, base_seed: int) -> None:
    """Run or resume one basin study to the requested total trial count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=base_seed + basin.source_trial,
        n_startup_trials=10,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        study_name=basin.name,
        storage=f"sqlite:///{study_path(output_dir, basin)}",
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    completed_centers = (
        trial.params
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE
    )
    if basin.center not in completed_centers:
        study.enqueue_trial(basin.center)

    history = raw_history()
    norm_slices = model_slices()

    def objective(optuna_trial: optuna.Trial) -> float:
        params = suggest_params(optuna_trial, basin)
        search_trial = SearchTrial(
            number=optuna_trial.number + 1,
            level_indices=(),
            values=tuple(params.items()),
        )
        try:
            result = run_trial(search_trial, history, norm_slices, (0,), base_seed)
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("Configuration exceeded GPU memory.") from error
        optuna_trial.set_user_attr("result", asdict(result))
        return result.objective_rate_gap

    attempted = sum(
        trial.state in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        )
        for trial in study.trials
    )
    remaining = max(0, target_trials - attempted)
    if remaining:
        study.optimize(objective, n_trials=remaining)
    write_summary(study, basin, output_dir)


def main() -> None:
    """Run one direction-aware Optuna basin study."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", required=True, choices=tuple(basin.name for basin in basins()))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--base-seed", type=int, default=60_000)
    args = parser.parse_args()
    assert args.trials > 0
    basin = next(basin for basin in basins() if basin.name == args.basin)
    run_study(basin, args.output_dir, args.trials, args.base_seed)


if __name__ == "__main__":
    main()
