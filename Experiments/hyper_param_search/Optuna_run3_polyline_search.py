"""Search around the production default against the actual run-3 O RD polyline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import optuna
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.hyper_param_search.Taguchi_oracle_search import (  # noqa: E402
    Level,
    Trial as SearchTrial,
    model_slices,
    parse_clients,
    raw_history,
    run_trial,
)


RESULTS_DIR = Path(__file__).parent / "results/round14_run3_polyline_optuna"
DEFAULT_TRIALS = 20
CURRENT_BEST: dict[str, Level] = {
    "reconst_ld": 36.73683767756248,
    "tau": 2.3406302935091077,
    "lr": 0.004530430161538338,
    "train_epochs": 120,
    "train_sample_size": 750_000,
    "train_batch_size": 12_500,
    "lr_step_fraction": 0.3051065332309726,
    "hidden_dim": 74,
    "layers": 6,
    "source_perturbation": "none",
}


def suggest_params(trial: optuna.Trial) -> dict[str, Level]:
    """Sample a focused neighborhood that extends every unresolved lower bound."""
    return {
        "reconst_ld": trial.suggest_float("reconst_ld", 20.0, 90.0, log=True),
        "tau": trial.suggest_float("tau", 1.5, 6.0, log=True),
        "lr": trial.suggest_float("lr", 3.5e-3, 8e-3, log=True),
        "train_epochs": trial.suggest_int("train_epochs", 60, 160, step=20),
        "train_sample_size": trial.suggest_int(
            "train_sample_size", 350_000, 750_000, step=50_000
        ),
        "train_batch_size": trial.suggest_categorical(
            "train_batch_size", (3_125, 6_250, 12_500)
        ),
        "lr_step_fraction": trial.suggest_float(
            "lr_step_fraction", 0.24, 0.38
        ),
        "hidden_dim": 74,
        "layers": 6,
        "source_perturbation": "none",
    }


def write_summary(study: optuna.Study, output_dir: Path, replica: str) -> None:
    """Persist the best and top focused trials for one independent replica."""
    complete = sorted(
        (
            trial
            for trial in study.trials
            if trial.state is optuna.trial.TrialState.COMPLETE
        ),
        key=lambda trial: float(trial.value),
    )
    summary = {
        "replica": replica,
        "objective": (
            "Integrated horizontal candidate-minus-run3 rate gap over the shared "
            "distortion span of the actual round-14 RD polylines."
        ),
        "completed_trials": len(complete),
        "pruned_trials": sum(
            trial.state is optuna.trial.TrialState.PRUNED
            for trial in study.trials
        ),
        "best": {
            "number": complete[0].number,
            "objective_rate_gap": complete[0].value,
            "params": complete[0].user_attrs["result"]["params"],
            "result": complete[0].user_attrs["result"],
        },
        "top_trials": [
            {
                "number": trial.number,
                "objective_rate_gap": trial.value,
                "params": trial.user_attrs["result"]["params"],
            }
            for trial in complete[:10]
        ],
    }
    (output_dir / f"replica_{replica}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def run_study(
    replica: str,
    output_dir: Path,
    target_trials: int,
    clients: tuple[int, ...],
    base_seed: int,
) -> None:
    """Run or resume one focused TPE replica to a fixed valid-attempt count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = output_dir / f"replica_{replica}.db"
    study = optuna.create_study(
        study_name=f"run3_polyline_{replica}",
        storage=f"sqlite:///{storage_path}",
        sampler=optuna.samplers.TPESampler(
            seed=base_seed,
            n_startup_trials=8,
            multivariate=True,
            group=True,
        ),
        direction="minimize",
        load_if_exists=True,
    )
    if not any(
        trial.state is optuna.trial.TrialState.COMPLETE
        and trial.user_attrs.get("result", {}).get("params") == CURRENT_BEST
        for trial in study.trials
    ):
        study.enqueue_trial({
            name: value
            for name, value in CURRENT_BEST.items()
            if name not in {"hidden_dim", "layers", "source_perturbation"}
        })

    history = raw_history()
    norm_slices = model_slices()

    def objective(optuna_trial: optuna.Trial) -> float:
        params = suggest_params(optuna_trial)
        search_trial = SearchTrial(
            number=optuna_trial.number + 1,
            level_indices=(),
            values=tuple(params.items()),
        )
        try:
            result = run_trial(
                search_trial, history, norm_slices, clients, base_seed
            )
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("Configuration exceeded GPU memory.") from error
        except ValueError as error:
            raise optuna.TrialPruned(str(error)) from error
        optuna_trial.set_user_attr("result", asdict(result))
        return result.objective_rate_gap

    attempted = sum(
        trial.state in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        )
        for trial in study.trials
    )
    if remaining := max(0, target_trials - attempted):
        study.optimize(objective, n_trials=remaining)
    write_summary(study, output_dir, replica)


def main() -> None:
    """Run one independent focused search replica."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", required=True)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--clients", type=parse_clients, default=(0,))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--base-seed", type=int, default=110_000)
    args = parser.parse_args()
    assert args.trials > 0
    assert Path(args.replica).name == args.replica
    run_study(
        args.replica,
        args.output_dir,
        args.trials,
        args.clients,
        args.base_seed,
    )


if __name__ == "__main__":
    main()
