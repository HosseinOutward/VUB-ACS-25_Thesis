"""Frozen study: last ran against commit bb5d9f1; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols.

Tune the round-14 Oracle LWZ quantizer against a fixed relative RD reference."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.hyper_param_search.Taguchi_real_si import (  # noqa: E402
    Factor,
    Level,
    Trial,
    build_trials,
    main_effects,
    recommended_values,
)
from FL_code.FL_core.codec import Access, CompressionRecord, ReconstructionHistory  # noqa: E402
from FL_code.FL_core.models import initialize_model  # noqa: E402
from FL_code.FL_core.utils import StateDictManager, set_global_seed  # noqa: E402
from FL_code.cancer_protocol.cancer_protocol import CancerRecord, Oracle_RoundCodec  # noqa: E402
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant  # noqa: E402
from FL_code.run_fl import FLConfig  # noqa: E402


RAW_DELTAS_DIR = PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas"
REFERENCE_RUN_DIR = (
    PROJECT_ROOT / "Experiments/Protocol_test/exp_records/run3_real_raw_o_nf4"
)
RESULTS_DIR = Path(__file__).parent / "results/round14_run3_polyline"
ROUND_ID = 14
HISTORY_DEPTH = 5
ALL_CLIENT_IDS = tuple(range(5))
BINS_PER_PLANE = 3
NUM_PLANES = 3
TRAINING_MATCH_BITS = 26

FACTORS = (
    Factor("reconst_ld", (100.0, 300.0, 900.0)),
    Factor("tau", (0.3, 1.0, 3.0)),
    Factor("lr", (1e-3, 3e-3, 9e-3)),
    Factor("train_epochs", (40, 80, 120)),
    Factor("train_sample_size", (100_000, 300_000, 600_000)),
    Factor("train_batch_size", (25_000, 50_000, 100_000)),
    Factor("lr_step_fraction", (0.25, 0.5, 1.0)),
    Factor("hidden_dim", (50, 100, 150)),
    Factor("layers", (2, 3, 4)),
    Factor("source_perturbation", ("none", "draw_discard", "add")),
)
TRIALS = build_trials(FACTORS)


@dataclass(frozen=True, slots=True)
class StageResult:
    """One averaged stage point on a measured RD curve."""

    stage: int
    rate: float
    wmape_percent: float
    distortion_db: float
    mse: float


@dataclass(frozen=True, slots=True)
class CurveComparison:
    """Horizontal rate gap integrated over the shared span of two actual RD polylines."""

    reference_rates: tuple[float, ...]
    reference_distortions_db: tuple[float, ...]
    distortion_range_db: tuple[float, float]
    reference_overlap_fraction: float
    integration_knots_db: tuple[float, ...]
    rate_gaps: tuple[float, ...]
    mean_rate_gap: float


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One fresh Oracle trial produced by the real raw history and target gradients."""

    trial: int
    seed: int
    clients: tuple[int, ...]
    params: dict[str, Level]
    resolved_config: dict[str, Any]
    stages: tuple[StageResult, ...]
    curve_comparison: CurveComparison
    objective_rate_gap: float
    seconds: float


def load_tensor(round_id: int, client_id: int) -> torch.Tensor:
    """Load one memory-mapped CPU float32 raw client delta."""
    path = RAW_DELTAS_DIR / f"round_{round_id}_client_{client_id}.pt"
    assert path.is_file(), f"Missing raw client delta: {path}."
    tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32 and tensor.device == torch.device("cpu"), (
        f"Expected a CPU float32 tensor in {path}; got {tensor.dtype} on {tensor.device}."
    )
    return tensor


def reference_curve(client_ids: tuple[int, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load the matching run-3 round-14 client rows and average their actual RD curve."""
    records_path = REFERENCE_RUN_DIR / "compression_records.csv"
    assert records_path.is_file(), f"Missing run-3 reference records: {records_path}."
    selected: dict[int, dict[str, str]] = {}
    with records_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            client_id = int(row["client_id"])
            if int(row["round_id"]) == ROUND_ID and client_id in client_ids:
                assert client_id not in selected, (
                    f"Duplicate round-{ROUND_ID} reference row for client {client_id} "
                    f"in {records_path}."
                )
                selected[client_id] = row
    assert set(selected) == set(client_ids), (
        f"Run-3 reference clients are {tuple(sorted(selected))}; expected {client_ids}."
    )
    stage_rates = np.array([
        np.cumsum(ast.literal_eval(selected[client_id]["prior_stage_rates"]))
        for client_id in client_ids
    ])
    stage_wmapes = np.array([
        ast.literal_eval(selected[client_id]["stage_wmapes"])
        for client_id in client_ids
    ])
    assert stage_rates.shape == stage_wmapes.shape == (len(client_ids), NUM_PLANES)
    rates = stage_rates.mean(axis=0)
    distortions = 20 * np.log10(stage_wmapes.mean(axis=0) / 100)
    assert (np.diff(rates) > 0).all() and (np.diff(distortions) < 0).all(), (
        f"Run-3 reference curve must improve monotonically: rates={rates}, "
        f"distortions={distortions}."
    )
    return tuple(map(float, rates)), tuple(map(float, distortions))


def compare_actual_polylines(
    reference_rates: tuple[float, ...],
    reference_distortions_db: tuple[float, ...],
    candidate_rates: tuple[float, ...],
    candidate_distortions_db: tuple[float, ...],
) -> CurveComparison:
    """Integrate candidate-minus-reference rate over the curves' shared distortion span."""
    ref_rates = np.asarray(reference_rates)
    ref_distortions = np.asarray(reference_distortions_db)
    candidate_rates_array = np.asarray(candidate_rates)
    candidate_distortions = np.asarray(candidate_distortions_db)
    assert (
        ref_rates.shape
        == ref_distortions.shape
        == candidate_rates_array.shape
        == candidate_distortions.shape
        == (NUM_PLANES,)
    )
    assert (np.diff(ref_rates) > 0).all() and (np.diff(ref_distortions) < 0).all()
    if not (
        (np.diff(candidate_rates_array) > 0).all()
        and (np.diff(candidate_distortions) < 0).all()
    ):
        raise ValueError(
            "Candidate RD stages must have increasing rate and decreasing distortion; "
            f"got rates={candidate_rates_array}, distortions={candidate_distortions}."
        )

    lower = max(float(ref_distortions[-1]), float(candidate_distortions[-1]))
    upper = min(float(ref_distortions[0]), float(candidate_distortions[0]))
    reference_span = float(ref_distortions[0] - ref_distortions[-1])
    overlap_fraction = (upper - lower) / reference_span
    if overlap_fraction < 0.75:
        raise ValueError(
            f"Candidate/reference distortion overlap is {overlap_fraction:.1%}; "
            "at least 75% is required."
        )
    knots = np.unique(np.concatenate((
        np.array((lower, upper)),
        ref_distortions[(ref_distortions > lower) & (ref_distortions < upper)],
        candidate_distortions[
            (candidate_distortions > lower) & (candidate_distortions < upper)
        ],
    )))
    reference_at_knots = np.interp(
        knots, ref_distortions[::-1], ref_rates[::-1]
    )
    candidate_at_knots = np.interp(
        knots, candidate_distortions[::-1], candidate_rates_array[::-1]
    )
    rate_gaps = candidate_at_knots - reference_at_knots
    mean_gap = float(np.trapezoid(rate_gaps, knots) / (upper - lower))
    return CurveComparison(
        reference_rates=reference_rates,
        reference_distortions_db=reference_distortions_db,
        distortion_range_db=(lower, upper),
        reference_overlap_fraction=overlap_fraction,
        integration_knots_db=tuple(map(float, knots)),
        rate_gaps=tuple(map(float, rate_gaps)),
        mean_rate_gap=mean_gap,
    )


def model_slices() -> tuple[slice, ...]:
    """Recover the run-3 model parameter boundaries used for normalization."""
    config_path = REFERENCE_RUN_DIR / "fl_config.json"
    assert config_path.is_file(), f"Missing baseline FL configuration: {config_path}."
    cfg = FLConfig.model_validate_json(config_path.read_text())
    manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
    sample = load_tensor(ROUND_ID, 0)
    assert manager.param_count == sample.numel(), (
        f"Baseline model has {manager.param_count} parameters, but {sample.numel()} "
        f"values were found in the raw round-{ROUND_ID} delta."
    )
    return tuple(manager.get_slices())


def raw_history() -> ReconstructionHistory:
    """Build the decoder-visible five-round raw history used by every Oracle trial."""
    history = ReconstructionHistory(
        max_per_client=HISTORY_DEPTH,
        routine_accesses=(Access.SERVER_ONLY,),
    )
    for history_round in range(ROUND_ID - HISTORY_DEPTH, ROUND_ID):
        for client_id in ALL_CLIENT_IDS:
            history.commit(
                load_tensor(history_round, client_id).to(dtype=history.dtype),
                CompressionRecord(
                    history_round,
                    client_id,
                    "history_seed_raw_deltas",
                ),
                Access.TEMPORAL_TOO,
            )
    visible = history.view(Access.SERVER_ONLY, 0)
    assert set(visible) == set(ALL_CLIENT_IDS)
    assert all(len(entries) == HISTORY_DEPTH for entries in visible.values())
    return history


def quantizer_config(trial_values: dict[str, Level], norm_slices: tuple[slice, ...]) -> WZcfgQuant:
    """Resolve one Taguchi row into a fresh, explicitly uncached quantizer configuration."""
    values = trial_values.copy()
    lr_step_fraction = float(values.pop("lr_step_fraction"))
    train_epochs = int(values["train_epochs"])
    return WZcfgQuant(
        **values,
        bins_per_plane=BINS_PER_PLANE,
        num_planes=NUM_PLANES,
        training_match_bits=TRAINING_MATCH_BITS,
        norm_slices=norm_slices,
        lr_step=max(1, round(train_epochs * lr_step_fraction)),
        cache_dir=None,
    )


def run_oracle_client(
    config: WZcfgQuant,
    history: ReconstructionHistory,
    client_id: int,
) -> CancerRecord:
    """Run the production Oracle codec flow once without committing its reconstruction."""
    codec = Oracle_RoundCodec(
        {
            "bins_per_plane": config.bins_per_plane,
            "num_planes": config.num_planes,
            "sd_slices": config.norm_slices,
            "cache_dir": None,
            "cache_overwrite": False,
        },
        f"O|bins_per_plane={config.bins_per_plane}|num_planes={config.num_planes}",
    )
    codec.c_cfg = config
    training_target, training_si = codec.get_training_data(history, client_id)
    codec.set_retrained_prior_si(history, client_id)
    codec.build_quantizer(training_target, training_si)
    record = codec.create_r_record(ROUND_ID, client_id)
    delta = load_tensor(ROUND_ID, client_id)
    codec.pre_encode_training_steps(delta)
    payload = codec.encode(delta, record)
    codec.decode(payload, record)
    assert record.prior_stage_rates is not None
    assert record.stage_wmapes is not None
    assert record.stage_mses is not None
    return record


def aggregate_stages(records: list[CancerRecord]) -> tuple[StageResult, ...]:
    """Average client records into one measured stage-wise RD curve."""
    rates = np.array([
        np.cumsum(record.prior_stage_rates)
        for record in records
        if record.prior_stage_rates is not None
    ])
    wmapes = np.array([
        record.stage_wmapes for record in records if record.stage_wmapes is not None
    ])
    mses = np.array([
        record.stage_mses for record in records if record.stage_mses is not None
    ])
    assert rates.shape == wmapes.shape == mses.shape == (len(records), NUM_PLANES)

    mean_rates = rates.mean(axis=0)
    mean_wmapes = wmapes.mean(axis=0)
    distortions_db = 20 * np.log10(mean_wmapes / 100)
    mean_mses = mses.mean(axis=0)
    return tuple(
        StageResult(
            stage=stage_index + 1,
            rate=float(rate),
            wmape_percent=float(wmape),
            distortion_db=float(distortion_db),
            mse=float(mse),
        )
        for stage_index, (rate, wmape, distortion_db, mse) in enumerate(
            zip(mean_rates, mean_wmapes, distortions_db, mean_mses, strict=True)
        )
    )


def run_trial(
    trial: Trial,
    history: ReconstructionHistory,
    norm_slices: tuple[slice, ...],
    client_ids: tuple[int, ...],
    base_seed: int,
) -> TrialResult:
    """Evaluate one trial against the matching clients' actual run-3 RD polyline."""
    seed = base_seed + trial.number
    set_global_seed(seed)
    config = quantizer_config(trial.as_dict(), norm_slices)
    started = perf_counter()
    records = [run_oracle_client(config, history, client_id) for client_id in client_ids]
    stages = aggregate_stages(records)
    comparison = compare_actual_polylines(
        *reference_curve(client_ids),
        tuple(stage.rate for stage in stages),
        tuple(stage.distortion_db for stage in stages),
    )
    return TrialResult(
        trial=trial.number,
        seed=seed,
        clients=client_ids,
        params=trial.as_dict(),
        resolved_config=config.model_dump(
            mode="json", exclude={"norm_slices", "pretrain_pth_dir"}
        ),
        stages=stages,
        curve_comparison=comparison,
        objective_rate_gap=comparison.mean_rate_gap,
        seconds=perf_counter() - started,
    )


def trial_path(output_dir: Path, trial_number: int) -> Path:
    """Return the independently writable result path for one L27 row."""
    return output_dir / "trials" / f"trial_{trial_number:02d}.json"


def write_trial(result: TrialResult, output_dir: Path) -> None:
    """Persist one complete trial atomically for safe sharded execution and resumption."""
    path = trial_path(output_dir, result.trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f".{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    temporary_path.replace(path)


def load_trial(path: Path) -> dict[str, Any]:
    """Load one persisted trial result."""
    result = json.loads(path.read_text())
    assert isinstance(result, dict) and int(result["trial"]) > 0, (
        f"Invalid Taguchi trial result: {path}."
    )
    return result


def write_manifest(
    output_dir: Path,
    screen_clients: tuple[int, ...],
    base_seed: int,
    factors: tuple[Factor, ...] = FACTORS,
) -> None:
    """Write or validate the experiment contract shared by all execution shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    reference_rates, reference_distortions = reference_curve(screen_clients)
    manifest = {
        "round": ROUND_ID,
        "history": {
            "source": str(RAW_DELTAS_DIR.relative_to(PROJECT_ROOT)),
            "rounds": list(range(ROUND_ID - HISTORY_DEPTH, ROUND_ID)),
            "clients": list(ALL_CLIENT_IDS),
            "dtype": "float16",
            "access": Access.SERVER_ONLY.value,
        },
        "screen_clients": list(screen_clients),
        "base_seed": base_seed,
        "cache": "disabled: every quantizer and prior fit uses cache_dir=None",
        "reference": {
            "source": (
                "Experiments/Protocol_test/exp_records/run3_real_raw_o_nf4/"
                "compression_records.csv, round 14, matching screen clients"
            ),
            "rates": reference_rates,
            "distortions_db": reference_distortions,
            "objective": (
                "candidate-minus-reference horizontal rate gap integrated over "
                "the shared distortion range of the two actual piecewise-linear "
                "RD curves; lower is better and negative improves on run3"
            ),
            "minimum_reference_distortion_overlap": 0.75,
        },
        "factors": [
            {"name": factor.name, "levels": factor.levels} for factor in factors
        ],
        "fixed_config": {
            "bins_per_plane": BINS_PER_PLANE,
            "num_planes": NUM_PLANES,
            "training_match_bits": TRAINING_MATCH_BITS,
            "normalization": (
                "Per-parameter standard-deviation slices from "
                "StateDictManager.get_slices(), identical to the real protocol."
            ),
            "quantizer_train_repeats": 2,
            "prior_train_repeats": 2,
            "outlier_threshold": None,
            "marginal_loss": False,
            "shared_heads": True,
            "prior_context": "hard",
            "prior_probabilities": "categorical",
            "deduplicate_training": True,
            "fused_optimizer": True,
            "mixed_precision": False,
            "tf32": False,
        },
    }
    manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, (
            f"Existing experiment manifest does not match this run: {manifest_path}."
        )
        return
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def analyze(
    output_dir: Path,
    factors: tuple[Factor, ...] = FACTORS,
    trials: tuple[Trial, ...] = TRIALS,
) -> dict[str, Any]:
    """Analyze a complete L27 screen and save its main-effect recommendation."""
    results = {
        int(result["trial"]): result
        for result in (
            load_trial(trial_path(output_dir, trial.number)) for trial in trials
        )
    }
    responses = {
        trial_number: float(result["objective_rate_gap"])
        for trial_number, result in results.items()
    }
    effects = main_effects(factors, trials, responses)
    ordered_trials = sorted(results.values(), key=lambda result: result["objective_rate_gap"])
    summary = {
        "best_observed": ordered_trials[0],
        "recommended_params": recommended_values(effects),
        "factor_effects": [asdict(effect) for effect in effects],
        "trials_by_objective": [
            {
                "trial": result["trial"],
                "objective_rate_gap": result["objective_rate_gap"],
                "params": result["params"],
            }
            for result in ordered_trials
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_clients(value: str) -> tuple[int, ...]:
    """Parse a comma-separated set of round-14 client IDs."""
    clients = tuple(int(item) for item in value.split(","))
    assert clients and len(set(clients)) == len(clients)
    assert set(clients) <= set(ALL_CLIENT_IDS), f"Clients must be in {ALL_CLIENT_IDS}."
    return clients


def parse_trials(value: str | None, trials: tuple[Trial, ...] = TRIALS) -> set[int]:
    """Parse comma-separated trial numbers and inclusive ranges."""
    if value is None:
        return {trial.number for trial in trials}
    selected: set[int] = set()
    for item in value.split(","):
        bounds = tuple(int(bound) for bound in item.split("-"))
        assert len(bounds) in (1, 2), f"Invalid trial selection: {item!r}."
        selected.update(
            range(bounds[0], bounds[-1] + 1) if len(bounds) == 2 else bounds
        )
    assert selected <= {trial.number for trial in trials}, (
        f"Trials must be between 1 and {len(trials)}."
    )
    return selected


def load_factors(path: Path | None) -> tuple[Factor, ...]:
    """Load an optional three-level factor set for a follow-up L27 experiment."""
    if path is None:
        return FACTORS
    raw_factors = json.loads(path.read_text())
    assert isinstance(raw_factors, list), f"Expected a factor list in {path}."
    return tuple(
        Factor(str(item["name"]), tuple(item["levels"]))
        for item in raw_factors
    )


def main() -> None:
    """Run a resumable L27 shard or analyze all completed Oracle trials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--factors-json", type=Path)
    parser.add_argument("--clients", type=parse_clients, default=(0,))
    parser.add_argument("--base-seed", type=int, default=45_000)
    parser.add_argument("--trials", help="Comma-separated trial numbers or inclusive ranges.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    factors = load_factors(args.factors_json)
    trials = build_trials(factors)
    write_manifest(args.output_dir, args.clients, args.base_seed, factors)
    if args.analyze:
        summary = analyze(args.output_dir, factors, trials)
        print(json.dumps(summary, indent=2))
        return

    assert 0 <= args.shard_index < args.shard_count
    selected = parse_trials(args.trials, trials)
    selected_trials = tuple(
        trial
        for trial in trials
        if trial.number in selected
        and (trial.number - 1) % args.shard_count == args.shard_index
    )
    assert selected_trials, "This shard has no selected Taguchi trials."
    slices = model_slices()
    history = raw_history()
    for trial in selected_trials:
        path = trial_path(args.output_dir, trial.number)
        if path.exists() and not args.overwrite:
            print(f"Skipping completed trial {trial.number}: {path}", flush=True)
            continue
        print(
            f"\n=== Oracle L27 trial {trial.number}/{len(trials)} "
            f"on clients {args.clients}: {trial.as_dict()} ===",
            flush=True,
        )
        result = run_trial(trial, history, slices, args.clients, args.base_seed)
        write_trial(result, args.output_dir)
        print(
            f"trial {trial.number}: objective={result.objective_rate_gap:+.6f} "
            f"rate-equivalent bits, seconds={result.seconds:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
