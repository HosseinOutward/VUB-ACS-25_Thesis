"""Compare 30-epoch scratch and warm retraining on seed-42069 round-22 gradients."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import traceback
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.Protocol_test.sparse_replay import load_fl_config
from FL_code.FL_core.codec import Access, CompressionRecord, simulate_compression
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, set_global_seed
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol
from FL_code.cancer_protocol.history import ReconstructionHistory
from FL_code.cancer_protocol.round_codecs import _WZRoundCodec
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant


PARENT_QUANTIZER_TRAIN_REPEATS = 2


class StudyConfig(BaseModel):
    """Reproducible trace, codec, comparison, and resource settings for this study."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    trace_seed: int
    training_repeats: int = Field(ge=10)
    parent_round: int
    evaluation_round: int
    history_rounds: int
    num_clients: int
    target_client: int
    bins_per_plane: int
    num_planes: int
    parent_epochs: int
    reduced_epochs: int
    rans_lanes: int
    bad_convergence_rate_gap_bits: float = Field(ge=0)
    warm_lr_multipliers: tuple[float, ...] | None = None
    warm_initial_taus: tuple[float, ...] | None = None
    gpu_ids: tuple[int, ...]
    max_parallel_cases: int = Field(ge=1)
    dataset_revision: str
    fl_config: Path
    output_root: Path

    @model_validator(mode="after")
    def validate_design(self) -> StudyConfig:
        """Protect the user-declared round, history, client, seed, and epoch design."""
        assert self.trace_seed == 42069
        assert (self.parent_round, self.evaluation_round) == (12, 22)
        assert self.history_rounds == 2 and self.num_clients == 5 and self.target_client == 0
        assert (self.bins_per_plane, self.num_planes) == (2, 3)
        assert (self.parent_epochs, self.reduced_epochs) == (120, 30)
        assert self.rans_lanes == 2048
        if self.warm_lr_multipliers is None or self.warm_initial_taus is None:
            assert self.warm_lr_multipliers is self.warm_initial_taus is None
        else:
            assert self.warm_lr_multipliers == (1.0, 0.3, 0.1, 0.03)
            assert self.warm_initial_taus == (5.5, 1.0, 0.5, 0.1)
        assert self.max_parallel_cases <= len(self.gpu_ids) == len(set(self.gpu_ids))
        return self


def load_config(path: Path) -> StudyConfig:
    """Load paths relative to the experiment configuration and validate FL identity."""
    payload = json.loads(path.read_text())
    for field in ("fl_config", "output_root"):
        payload[field] = (path.parent / payload[field]).resolve()
    config = StudyConfig.model_validate(payload)
    fl_config = load_fl_config(config.fl_config)
    assert fl_config.seed == config.trace_seed
    assert fl_config.num_clients == config.num_clients and fl_config.rounds > config.evaluation_round
    return config


def warm_variants(config: StudyConfig) -> dict[str, tuple[float, float]]:
    """Return stable condition names and resolved LR/tau pairs for the declared grid."""
    default_lr = float(WZcfgQuant.model_fields["lr"].default)
    if config.warm_lr_multipliers is None:
        return {"warm30": (default_lr, float(WZcfgQuant.model_fields["tau"].default))}
    assert config.warm_initial_taus is not None
    label = lambda value: f"{value:g}".replace(".", "p")
    return {
        f"warm30_lr{label(multiplier)}_tau{label(tau)}": (default_lr * multiplier, tau)
        for multiplier in config.warm_lr_multipliers for tau in config.warm_initial_taus
    }


def warm_conditions(config: StudyConfig) -> tuple[str, ...]:
    """Return warm conditions in the configuration's Cartesian-product order."""
    return tuple(warm_variants(config))


class TwoRoundHistoryProtocol(WZCancerProtocol):
    """Production WZ protocol with exactly two retained rounds per each of five clients."""

    protocol_name = "gradient_warm_two_round_history"
    max_per_client_recons_history = 2


class WarmTwoRoundHistoryProtocol(TwoRoundHistoryProtocol):
    """Two-round protocol that initializes its round-22 codec from the round-12 parent."""

    protocol_name = "gradient_warm_initialized"

    def __init__(self, parent_state: Mapping[str, torch.Tensor], sd_slices: Sequence[slice]) -> None:
        super().__init__(sd_slices=sd_slices)
        self.parent_state = {key: value.detach().cpu().clone() for key, value in parent_state.items()}

    def create_round_codec(self, round_id: int, client_id: int) -> Any:
        """Load the parent before the ordinary production training loop begins."""
        codec = super().create_round_codec(round_id, client_id)
        if round_id == 22:
            assert isinstance(codec, _WZRoundCodec)
            assert codec.quantizer.side_info_list_used is None
            codec.quantizer.coding_model.load_state_dict(self.parent_state)
        return codec


def round_token(config: StudyConfig, epochs: int, training_seed: int,
                quantizer_train_repeats: int | None = None,
                overrides: Mapping[str, float] | None = None) -> str:
    """Return the production Oracle-rANS token with only declared overrides."""
    token = (
        f"OR|bins_per_plane={config.bins_per_plane}|num_planes={config.num_planes}"
        f"|train_epochs={epochs}|training_seed={training_seed}|rans_lanes={config.rans_lanes}"
    )
    if quantizer_train_repeats is not None:
        token += f"|quantizer_train_repeats={quantizer_train_repeats}"
    for key, value in (overrides or {}).items():
        assert key in {"lr", "tau"}
        token += f"|{key}={value}"
    return token


def condition_hyperparameters(config: StudyConfig, condition: str) -> tuple[float, float]:
    """Return the resolved LR and initial Gumbel temperature for one condition."""
    default_lr = float(WZcfgQuant.model_fields["lr"].default)
    default_tau = float(WZcfgQuant.model_fields["tau"].default)
    return warm_variants(config).get(condition, (default_lr, default_tau))


def seed_history(protocol: WZCancerProtocol, rounds: Sequence[int],
                 values: Mapping[int, Sequence[torch.Tensor]], config: StudyConfig) -> None:
    """Replace decoder-visible history with raw five-client gradients from selected rounds."""
    assert len(rounds) == config.history_rounds
    history = ReconstructionHistory(max_per_client=config.history_rounds)
    for round_id in rounds:
        assert len(values[round_id]) == config.num_clients
        for client_id, gradient in enumerate(values[round_id]):
            history.commit(
                gradient, CompressionRecord(round_id, client_id, "raw_trace_history"),
                Access.TEMPORAL_TOO,
            )
    protocol._recons_history = history
    protocol.histories = (history,)


def generate_trace_artifacts(config: StudyConfig) -> dict[int, tuple[torch.Tensor, ...]]:
    """Generate the exact five-client FL trajectory, retaining only required raw gradients."""
    path = config.output_root / "trace" / "required_gradients.pt"
    trace_dir = config.output_root / "trace"
    fl_config = load_fl_config(config.fl_config)
    trace = Trace(
        trace_dir / "checkpoints", fl_config, dataset_revision=config.dataset_revision,
        device=torch.device(f"cuda:{config.gpu_ids[0]}"),
    )
    # TorchVersion is a str subclass that torch's safe loader otherwise treats as a custom global.
    identity = json.loads(json.dumps(trace.identity()))
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["identity"] == identity, f"Cached trace identity differs at {path}."
        return {int(key): tuple(values) for key, values in payload["rounds"].items()}
    trace_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        config.parent_round - 2, config.parent_round - 1, config.parent_round,
        config.evaluation_round - 2, config.evaluation_round - 1, config.evaluation_round,
    }
    retained = {
        round_id: tuple(value.detach().cpu().clone() for value in gradients)
        for round_id, gradients in trace.stream(min(wanted), max(wanted))
        if round_id in wanted
    }
    assert set(retained) == wanted
    temporary = path.with_suffix(".tmp")
    torch.save({"identity": identity, "rounds": retained}, temporary)
    temporary.replace(path)
    return retained


def manager(config: StudyConfig) -> StateDictManager:
    """Construct the StateDictManager matching every retained gradient vector."""
    return StateDictManager(initialize_model(load_fl_config(config.fl_config), torch.device("cpu")))


def parse_stage_values(record: Mapping[str, str], field: str, count: int) -> list[float]:
    """Parse one production staged record field and enforce complete curves."""
    values = ast.literal_eval(record[field])
    assert isinstance(values, list) and len(values) == count
    return [float(value) for value in values]


def record_rows(record: Mapping[str, str], config: StudyConfig, condition: str, repeat: int,
                phase: str) -> list[dict[str, Any]]:
    """Expand a production record into cumulative rANS hard-symbol RD stage rows."""
    rates = list(itertools.accumulate(parse_stage_values(
        record, "coder_stage_rates", config.num_planes)))
    wmapes = parse_stage_values(record, "stage_wmapes", config.num_planes)
    mses = parse_stage_values(record, "stage_mses", config.num_planes)
    return [{
        "condition": condition, "repeat": repeat,
        "training_seed": config.trace_seed if repeat < 0 else config.trace_seed + repeat,
        "phase": phase, "round_id": int(record["round_id"]), "stage": stage + 1,
        "bins_per_plane": config.bins_per_plane, "num_planes": config.num_planes,
        "train_epochs": 0 if condition == "frozen_parent" and phase == "frozen_r22" else (
            config.parent_epochs if condition in {"parent_r12", "scratch120"}
            or phase == "trained_r12" else config.reduced_epochs),
        "side_info_count": config.history_rounds * config.num_clients,
        "quantizer_train_repeats": (
            PARENT_QUANTIZER_TRAIN_REPEATS if condition in {"parent_r12", "frozen_parent"} else 1
        ),
        "coder_rate": rates[stage], "wmape": wmapes[stage],
        "wmape_db": 20 * np.log10(wmapes[stage] / 100),
        "mse": mses[stage], "mse_db": 10 * np.log10(mses[stage]),
        "coder_overhead_rate": float(record["coder_overhead_rate"]),
        "symbol_error_rate": float(record["symbol_error_rate"]),
        "pre_encode_training_seconds": float(record["pre_encode_training_seconds"]),
        "total_seconds": float(record["total_seconds"]),
    } for stage in range(config.num_planes)]


def run_simulation(protocol: WZCancerProtocol, target: torch.Tensor, round_id: int,
                   state_manager: StateDictManager) -> dict[str, str]:
    """Run one production compression endpoint and return its sole record."""
    with TemporaryDirectory(prefix="gradient_warm_") as directory:
        simulate_compression(protocol, target, 0, round_id, state_manager, Path(directory), {}, (), ())
        with (Path(directory) / "compression_records.csv").open(newline="") as file:
            records = list(csv.DictReader(file))
    assert len(records) == 1
    return records[0]


def train_parent(config: StudyConfig, gradients: Mapping[int, Sequence[torch.Tensor]],
                 state_manager: StateDictManager) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Train/cache the round-12 parent and evaluate it frozen at round 22 with raw recent SI."""
    checkpoint = config.output_root / "parent" / "round12_parent.pt"
    records_path = config.output_root / "parent" / "raw_stage_results.csv"
    expected_token = round_token(
        config,
        config.parent_epochs,
        config.trace_seed,
        quantizer_train_repeats=PARENT_QUANTIZER_TRAIN_REPEATS,
    )
    if checkpoint.exists() and records_path.exists():
        parent = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert parent["trace_seed"] == config.trace_seed
        assert parent["round"] == config.parent_round and parent["codec_token"] == expected_token
        with records_path.open(newline="") as file:
            return list(csv.DictReader(file)), parent["state_dict"]
    protocol = TwoRoundHistoryProtocol(
        sd_slices=state_manager.get_slices(), cache_dir=config.output_root / "cache")
    protocol.set_schedule(config.parent_round, (
        expected_token, "F"))
    seed_history(protocol, (10, 11), gradients, config)
    set_global_seed(config.trace_seed)
    trained = run_simulation(
        protocol, gradients[config.parent_round][config.target_client],
        config.parent_round, state_manager)
    codec = protocol.last_trained_codec[config.target_client]
    parent_state = {
        key: value.detach().cpu().clone()
        for key, value in codec.quantizer.coding_model.state_dict().items()
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    assert codec._retrained_prior_model is not None
    checkpoint_temporary = checkpoint.with_suffix(".tmp")
    torch.save({
        "trace_seed": config.trace_seed, "round": config.parent_round,
        "codec_token": expected_token,
        "state_dict": parent_state,
        "retrained_prior_state_dict": codec._retrained_prior_model.state_dict(),
        "training_prior_stage_rates": codec.quantizer.training_prior_stage_rates,
        "side_info_count": config.history_rounds * config.num_clients,
        "normalization": "StateDictManager.get_slices()",
    }, checkpoint_temporary)
    checkpoint_temporary.replace(checkpoint)
    # Skipped rounds never enter protocol history: round 22 sees raw rounds 20 and 21 only.
    seed_history(protocol, (20, 21), gradients, config)
    frozen = run_simulation(
        protocol, gradients[config.evaluation_round][config.target_client],
        config.evaluation_round, state_manager)
    return (
        record_rows(trained, config, "parent_r12", -1, "trained_r12")
        + record_rows(frozen, config, "frozen_parent", -1, "frozen_r22"),
        parent_state,
    )


def run_repeat(config_path: Path, repeat: int, gpu_id: int) -> Path:
    """Run paired round-22 scratch/warm conditions in one disposable GPU worker."""
    torch.cuda.set_device(gpu_id)
    config = load_config(config_path)
    case_dir = config.output_root / "cases" / f"repeat_{repeat:02d}"
    completion = case_dir / "complete.json"
    if completion.exists():
        return completion
    assert not case_dir.exists(), f"Incomplete output requires inspection: {case_dir}"
    case_dir.mkdir(parents=True)
    gradients_payload = torch.load(
        config.output_root / "trace" / "required_gradients.pt", map_location="cpu", weights_only=True)
    gradients = {int(key): tuple(values) for key, values in gradients_payload["rounds"].items()}
    parent = torch.load(
        config.output_root / "parent" / "round12_parent.pt", map_location="cpu", weights_only=True)
    assert parent["trace_seed"] == config.trace_seed
    assert parent["round"] == config.parent_round
    assert parent["codec_token"] == round_token(
        config,
        config.parent_epochs,
        config.trace_seed,
        quantizer_train_repeats=PARENT_QUANTIZER_TRAIN_REPEATS,
    )
    assert parent["side_info_count"] == config.history_rounds * config.num_clients
    assert parent["normalization"] == "StateDictManager.get_slices()"
    parent_state = parent["state_dict"]
    state_manager = manager(config)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    training_seed = config.trace_seed + repeat
    conditions: tuple[tuple[str, int, bool, dict[str, float]], ...] = (
        (("scratch30", config.reduced_epochs, False, {}),)
        + tuple(
            (condition, config.reduced_epochs, True, {"lr": lr, "tau": tau})
            for condition, (lr, tau) in warm_variants(config).items()
        )
        + (("scratch120", config.parent_epochs, False, {}),)
    )
    for condition, epochs, warm, overrides in conditions:
        started = perf_counter()
        try:
            protocol = (
                WarmTwoRoundHistoryProtocol(parent_state, state_manager.get_slices())
                if warm else TwoRoundHistoryProtocol(sd_slices=state_manager.get_slices())
            )
            protocol.set_schedule(config.evaluation_round, (
                round_token(
                    config, epochs, training_seed, quantizer_train_repeats=1,
                    overrides=overrides), "F"))
            seed_history(protocol, (20, 21), gradients, config)
            set_global_seed(training_seed)
            record = run_simulation(
                protocol, gradients[config.evaluation_round][config.target_client],
                config.evaluation_round, state_manager)
            codec = protocol.last_trained_codec[config.target_client]
            expected_lr, expected_tau = condition_hyperparameters(config, condition)
            assert codec.c_cfg.lr == expected_lr and codec.c_cfg.tau == expected_tau
            assert codec.c_cfg.train_epochs == epochs and codec.c_cfg.quantizer_train_repeats == 1
            assert codec.rans_config.lanes == config.rans_lanes
            rows.extend(record_rows(record, config, condition, repeat, "trained_r22"))
        except (AssertionError, RuntimeError, ValueError, OSError) as error:
            failures.append({
                "repeat": repeat, "condition": condition,
                "failure_type": type(error).__name__, "message": str(error),
            })
            (case_dir / f"{condition}_traceback.txt").write_text(traceback.format_exc())
        finally:
            print(
                f"repeat={repeat} condition={condition} elapsed_seconds={perf_counter() - started:.1f}",
                flush=True,
            )
            torch.cuda.empty_cache()
    write_csv(case_dir / "raw_stage_results.csv", rows)
    write_csv(case_dir / "failures.csv", failures)
    temporary = completion.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "repeat": repeat, "finished": datetime.now().astimezone().isoformat(),
        "conditions_completed": sorted({row["condition"] for row in rows}),
        "failures": len(failures),
    }, indent=2))
    temporary.replace(completion)
    return completion


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write one homogeneous result CSV when it has rows."""
    if not rows:
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_provenance() -> dict[str, Any]:
    """Fingerprint source relevant to trace generation, protocol execution, and analysis."""
    watched = (
        "FL_code", "Experiments/Protocol_test", "Experiments/warm_start_gradient_retraining")
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout
    diff = git("diff", "HEAD", "--", *watched)
    changed_tracked = sorted(git("diff", "HEAD", "--name-only", "--", *watched).splitlines())
    untracked = sorted(
        path for path in git("ls-files", "--others", "--exclude-standard", "--", *watched).splitlines()
        if not path.startswith("Experiments/warm_start_gradient_retraining/results/"))
    digest = hashlib.sha256(diff.encode())
    for relative in untracked:
        digest.update((PROJECT_ROOT / relative).read_bytes())
    return {
        "git_head": git("rev-parse", "HEAD").strip(),
        "dirty_source_fingerprint": digest.hexdigest(),
        "dirty_source_paths": changed_tracked + untracked,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }


def run(config_path: Path) -> None:
    """Generate one trace and parent, then run paired stochastic repeats on bounded GPUs."""
    config = load_config(config_path)
    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest = config.output_root / "manifest.json"
    declared = json.loads(config_path.read_text())
    declared_fl = json.loads(config.fl_config.read_text())
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        assert stored["config"] == declared and stored["fl_config"] == declared_fl
    else:
        manifest.write_text(json.dumps({
            "started": datetime.now().astimezone().isoformat(), "config": declared,
            "fl_config": declared_fl,
            "provenance": source_provenance(),
            "skipped_round_semantics": "raw FL advances; no codec calls in rounds 13-21",
            "round22_resolved_settings": {
                condition: {
                    "epochs": config.parent_epochs if condition == "scratch120" else config.reduced_epochs,
                    "lr": condition_hyperparameters(config, condition)[0],
                    "initial_tau": condition_hyperparameters(config, condition)[1],
                    "quantizer_train_repeats": 1,
                    "rans_lanes": config.rans_lanes,
                }
                for condition in ("scratch30", *warm_conditions(config), "scratch120")
            },
        }, indent=2))
    gradients = generate_trace_artifacts(config)
    parent_rows, _parent_state = train_parent(config, gradients, manager(config))
    write_csv(config.output_root / "parent" / "raw_stage_results.csv", parent_rows)
    del gradients, parent_rows, _parent_state
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.max_parallel_cases, mp_context=context, max_tasks_per_child=1,
    ) as pool:
        for start in range(0, config.training_repeats, config.max_parallel_cases):
            repeats = range(start, min(start + config.max_parallel_cases, config.training_repeats))
            futures = [
                pool.submit(run_repeat, config_path, repeat,
                            config.gpu_ids[(start + offset) % len(config.gpu_ids)])
                for offset, repeat in enumerate(repeats)
            ]
            for future in futures:
                future.result()
            subprocess.run(
                (sys.executable, "-m", "Experiments.warm_start_gradient_retraining.analyze",
                 str(config_path)), cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Validate or run the declared study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.validate_only:
        print(config.model_dump_json(indent=2))
        return
    run(args.config.resolve())


if __name__ == "__main__":
    main()
