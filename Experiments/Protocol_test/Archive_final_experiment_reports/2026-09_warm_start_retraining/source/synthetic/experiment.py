"""Run the full Brent-scale learned-WZ warm-start retraining experiment."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
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
import time
import traceback
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.codec import Access, CompressionRecord, simulate_compression
from FL_code.FL_core.utils import StateDictManager, parse_configurable_name, set_global_seed
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol
from FL_code.cancer_protocol.history import ReconstructionHistory
from FL_code.cancer_protocol.round_codecs import _WZRoundCodec


class Architecture(BaseModel):
    """One Brent architecture consumed by the experiment runner and analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    bins_per_plane: int = Field(gt=1)
    num_planes: int = Field(gt=0)
    reconst_ld: float = Field(gt=0)

    @property
    def key(self) -> str:
        """Return a filesystem-safe architecture identifier."""
        return f"b{self.bins_per_plane}_p{self.num_planes}_ld{self.reconst_ld:g}"


class Variant(BaseModel):
    """One scratch or warm-retraining condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    warm_start: bool
    lr: float = Field(gt=0)
    tau: float = Field(gt=0)


class ExperimentConfig(BaseModel):
    """Reproducible full-scale study configuration consumed by runner and analyser."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    repeats: int = Field(ge=10)
    data_size: int
    noise_power: float
    si_shift: float
    train_round_id: int
    client_id: int
    codec: Literal["O"]
    architectures: tuple[Architecture, ...]
    training: dict[str, Any]
    variants: tuple[Variant, ...]
    bad_convergence_rate_gap_bits: float = Field(ge=0)
    gpu_ids: tuple[int, ...]
    max_parallel_cases: int = Field(ge=1)
    output_root: Path

    @model_validator(mode="after")
    def validate_design(self) -> ExperimentConfig:
        """Reject accidental drift from the authoritative Brent-scale design."""
        expected_training = {
            "train_epochs": 180, "train_sample_size": 200_000, "train_batch_size": 1_000,
            "lr": 3e-3, "lr_step": 40, "tau": 1.0, "training_progress_bar": False,
            "training_log_every_epochs": 5, "quantizer_train_repeats": 1,
            "shared_heads": True, "prior_context": "hard",
            "prior_probabilities": "categorical", "fused_optimizer": True,
            "hidden_dim": 100, "layers": 3,
        }
        assert self.data_size == 2_000_000 and self.noise_power == 0.01
        assert self.train_round_id == 1 and self.client_id == 0
        assert self.architectures == (
            Architecture(bins_per_plane=2, num_planes=3, reconst_ld=120.0),
            Architecture(bins_per_plane=4, num_planes=2, reconst_ld=160.0),
        )
        assert self.training == expected_training, "Training settings differ from the Brent baseline."
        assert tuple(variant.name for variant in self.variants) == (
            "scratch", "warm_naive", "warm_lr10", "warm_tau10", "warm_lr10_tau10")
        assert len(set(self.gpu_ids)) == len(self.gpu_ids)
        assert self.max_parallel_cases <= len(self.gpu_ids)
        return self


def load_config(path: Path) -> ExperimentConfig:
    """Load the experiment configuration with paths relative to its file."""
    payload = json.loads(path.read_text())
    payload["output_root"] = (path.parent / payload["output_root"]).resolve()
    config = ExperimentConfig.model_validate(payload)
    verify_against_brent(config)
    return config


def verify_against_brent(config: ExperimentConfig) -> None:
    """Verify preserved settings directly against the authoritative Brent module."""
    from Experiments.brent_quantizer_check_paper import quantizer_check as brent

    assert config.data_size == brent.DATA_SIZE
    assert config.noise_power == brent.NOISE_POWER
    assert config.train_round_id == brent.TRAIN_ROUND_ID
    assert config.client_id == brent.CLIENT_ID
    assert tuple(
        (item.bins_per_plane, item.num_planes, item.reconst_ld) for item in config.architectures
    ) == brent.QUANTIZER_SETTINGS
    for architecture in config.architectures:
        brent_config = brent.ExperimentConfig(
            architecture.bins_per_plane, architecture.num_planes, architecture.reconst_ld)
        baseline = parse_configurable_name(brent.make_round_token("O", brent_config), "round codec")
        candidate = parse_configurable_name(round_token(architecture, config.training), "round codec")
        assert candidate == baseline, f"Round token drifted from Brent: {candidate!r} != {baseline!r}."


def round_token(architecture: Architecture, training: Mapping[str, Any]) -> str:
    """Build the exact Brent production Oracle round token."""
    options = {
        "bins_per_plane": architecture.bins_per_plane,
        "num_planes": architecture.num_planes,
        "reconst_ld": architecture.reconst_ld,
        **training,
    }
    return "O" + "".join(f"|{name}={value!r}" for name, value in options.items())


def populate_protocol(protocol: WZCancerProtocol, config: ExperimentConfig,
                      side_information: torch.Tensor, schedule: tuple[str, ...]) -> None:
    """Install the production schedule and its single decoder-visible SI vector."""
    protocol.set_schedule(config.train_round_id, schedule)
    history = ReconstructionHistory(max_per_client=1)
    history.commit(
        side_information,
        CompressionRecord(config.train_round_id - 1, config.client_id, "synthetic_history"),
        Access.TEMPORAL_TOO,
    )
    protocol.max_per_client_recons_history = history.max_per_client
    protocol._recons_history = history
    protocol.histories = (history,)


class WarmStartProtocol(WZCancerProtocol):
    """Experiment-only production protocol that initializes its training codec from a parent."""

    protocol_name = "warm_start_retraining_experiment"

    def __init__(self, parent_state: Mapping[str, torch.Tensor]) -> None:
        super().__init__(sd_slices=(slice(None),))
        self.parent_state = {key: value.detach().cpu().clone() for key, value in parent_state.items()}

    def create_round_codec(self, round_id: int, client_id: int) -> Any:
        """Create the normal codec, loading the parent before its ordinary training loop runs."""
        codec = super().create_round_codec(round_id, client_id)
        if round_id == 1:
            assert isinstance(codec, _WZRoundCodec)
            assert codec.quantizer.side_info_list_used is None
            codec.quantizer.coding_model.load_state_dict(self.parent_state)
        return codec


def generated_data(config: ExperimentConfig, repeat: int) -> tuple[torch.Tensor, ...]:
    """Generate tensors in exactly the same order as the authoritative Brent experiment."""
    generator = torch.Generator().manual_seed(repeat * 42)
    noise_std = float(np.sqrt(config.noise_power))
    training_si = torch.randn(config.data_size, generator=generator)
    training_target = training_si + torch.randn(config.data_size, generator=generator) * noise_std
    inference_si = torch.randn(config.data_size, generator=generator)
    inference_target = inference_si + torch.randn(config.data_size, generator=generator) * noise_std
    return training_si, training_target, inference_si, inference_target


def parse_list(record: Mapping[str, str], field: str, count: int) -> list[float]:
    """Parse and validate a production staged record field."""
    values = ast.literal_eval(record[field])
    assert isinstance(values, list) and len(values) == count
    return [float(value) for value in values]


def stage_rows(record: Mapping[str, str], config: ExperimentConfig, architecture: Architecture,
               repeat: int, condition: str, phase: str) -> list[dict[str, Any]]:
    """Convert one production record into complete cumulative hard-symbol stage points."""
    cumulative_rates = {
        output: list(itertools.accumulate(parse_list(record, source, architecture.num_planes)))
        for output, source in {
            "training_prior_rate": "training_prior_stage_rates",
            "prior_rate": "prior_stage_rates",
            "marginal_rate": "marginal_stage_rates",
        }.items()
    }
    mses = parse_list(record, "stage_mses", architecture.num_planes)
    wmapes = parse_list(record, "stage_wmapes", architecture.num_planes)
    return [{
        "repeat": repeat,
        "seed": repeat * 42,
        "condition": condition,
        "phase": phase,
        "round_id": int(record["round_id"]),
        "round_name_full": record["round_name_full"],
        "bins_per_plane": architecture.bins_per_plane,
        "stage": stage + 1,
        "model_num_planes": architecture.num_planes,
        "reconst_ld": architecture.reconst_ld,
        "data_size": config.data_size,
        "noise_power": config.noise_power,
        "si_shift": config.si_shift,
        **{field: values[stage] for field, values in cumulative_rates.items()},
        "mse": mses[stage],
        "mse_db": 10 * np.log10(mses[stage]),
        "wmape": wmapes[stage],
        "wmape_db": 20 * np.log10(wmapes[stage] / 100),
        "pre_encode_training_seconds": float(record["pre_encode_training_seconds"]),
        "total_seconds": float(record["total_seconds"]),
    } for stage in range(architecture.num_planes)]


def simulate_condition(config: ExperimentConfig, architecture: Architecture, repeat: int,
                       condition: str, protocol: WZCancerProtocol, training_si: torch.Tensor,
                       training_target: torch.Tensor, inference_si: torch.Tensor,
                       inference_target: torch.Tensor, manager: StateDictManager,
                       parent_diagnostic: bool = False) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Run one training and frozen evaluation sequence through production simulation."""
    token = round_token(architecture, protocol.training_options)  # type: ignore[attr-defined]
    schedule = (token, "F", "F") if parent_diagnostic else (token, "F")
    populate_protocol(protocol, config, training_si, schedule)
    with TemporaryDirectory(prefix=f"wz_warm_{condition}_") as directory:
        records_dir = Path(directory)
        simulate_compression(
            protocol, training_target, config.client_id, config.train_round_id,
            manager, records_dir, {}, (), (),
        )
        parent_state = {
            key: value.detach().cpu().clone()
            for key, value in protocol.last_trained_codec[config.client_id].quantizer.coding_model.state_dict().items()
        }
        protocol._recons_history.replace_latest(
            inference_si, round_id=config.train_round_id, client_id=config.client_id)
        simulate_compression(
            protocol, inference_target, config.client_id, config.train_round_id + 1,
            manager, records_dir, {}, (), (),
        )
        if parent_diagnostic:
            protocol._recons_history.replace_latest(
                inference_si + config.si_shift,
                round_id=config.train_round_id + 1,
                client_id=config.client_id,
            )
            simulate_compression(
                protocol, inference_target, config.client_id, config.train_round_id + 2,
                manager, records_dir, {}, (), (),
            )
        with (records_dir / "compression_records.csv").open(newline="") as file:
            records = list(csv.DictReader(file))
    phases = (
        ("training_unshifted", "heldout_unshifted", "heldout_shifted")
        if parent_diagnostic else ("training_shifted", "heldout_shifted")
    )
    assert len(records) == len(phases)
    return [
        row
        for phase, record in zip(phases, records, strict=True)
        for row in stage_rows(record, config, architecture, repeat, condition, phase)
    ], parent_state


def protocol_for(training: Mapping[str, Any], parent_state: Mapping[str, torch.Tensor] | None,
                 cache_dir: Path | None = None) -> WZCancerProtocol:
    """Create a production protocol for scratch, parent, or warm training."""
    protocol = (
        WZCancerProtocol(sd_slices=(slice(None),), cache_dir=cache_dir)
        if parent_state is None else WarmStartProtocol(parent_state)
    )
    protocol.training_options = dict(training)  # type: ignore[attr-defined]
    return protocol


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write a homogeneous raw or failure CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_case(config_path: Path, architecture_index: int, repeat: int, gpu_id: int) -> Path:
    """Run one paired architecture/repeat case in a disposable GPU process."""
    torch.cuda.set_device(gpu_id)
    config = load_config(config_path)
    architecture = config.architectures[architecture_index]
    case_dir = config.output_root / "cases" / architecture.key / f"repeat_{repeat:02d}"
    completion = case_dir / "complete.json"
    if completion.exists():
        return completion
    assert not case_dir.exists(), f"Incomplete case output requires inspection: {case_dir}"
    case_dir.mkdir(parents=True)
    training_si, training_target, inference_si, inference_target = generated_data(config, repeat)
    model = torch.nn.Module()
    model.register_parameter("synthetic_vector", torch.nn.Parameter(torch.empty(config.data_size)))
    manager = StateDictManager(model)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    parent_state: dict[str, torch.Tensor] | None = None

    conditions: list[tuple[str, dict[str, Any], bool, bool]] = [
        ("frozen_parent", dict(config.training), False, True),
        *[
            (variant.name, config.training | {"lr": variant.lr, "tau": variant.tau},
             variant.warm_start, False)
            for variant in config.variants
        ],
    ]
    for condition, training, warm_start, parent_diagnostic in conditions:
        if warm_start and parent_state is None:
            failures.append({
                "repeat": repeat, "architecture": architecture.key, "condition": condition,
                "failure_type": "missing_parent", "message": "Parent training failed.",
            })
            continue
        set_global_seed(repeat * 42)
        started = time.perf_counter()
        try:
            condition_training_si = training_si if parent_diagnostic else training_si + config.si_shift
            condition_inference_si = inference_si if parent_diagnostic else inference_si + config.si_shift
            protocol = protocol_for(
                training,
                parent_state if warm_start else None,
                config.output_root / "cache" if parent_diagnostic else None,
            )
            condition_rows, trained_state = simulate_condition(
                config, architecture, repeat, condition, protocol,
                condition_training_si, training_target, condition_inference_si,
                inference_target, manager, parent_diagnostic,
            )
            rows.extend(condition_rows)
            if parent_diagnostic:
                parent_state = trained_state
                torch.save({
                    "repeat": repeat,
                    "architecture": architecture.model_dump(),
                    "training": config.training,
                    "state_dict": parent_state,
                }, case_dir / "frozen_parent.pt")
        except (AssertionError, RuntimeError, ValueError, OSError) as error:
            failures.append({
                "repeat": repeat, "architecture": architecture.key, "condition": condition,
                "failure_type": type(error).__name__, "message": str(error),
            })
            (case_dir / f"{condition}_traceback.txt").write_text(traceback.format_exc())
        finally:
            elapsed = time.perf_counter() - started
            print(
                f"case={architecture.key}/repeat_{repeat:02d} condition={condition} "
                f"elapsed_seconds={elapsed:.1f}", flush=True,
            )
            torch.cuda.empty_cache()

    write_csv(case_dir / "raw_stage_results.csv", rows)
    write_csv(case_dir / "failures.csv", failures)
    temporary = completion.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "finished": datetime.now().astimezone().isoformat(),
        "architecture": architecture.model_dump(), "repeat": repeat,
        "conditions_completed": sorted({row["condition"] for row in rows}),
        "failures": len(failures),
    }, indent=2))
    temporary.replace(completion)
    return completion


def git_provenance() -> dict[str, Any]:
    """Record the repository state that produced the experiment."""
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        ).stdout
    watched = ("FL_code", "Experiments/brent_quantizer_check_paper",
               "Experiments/warm_start_retraining", "Experiments/Protocol_test/rd_metrics.py")
    diff = git("diff", "HEAD", "--", *watched)
    untracked = sorted(git("ls-files", "--others", "--exclude-standard", "--", *watched).splitlines())
    digest = hashlib.sha256(diff.encode())
    for relative in untracked:
        digest.update((PROJECT_ROOT / relative).read_bytes())
    return {
        "git_head": git("rev-parse", "HEAD").strip(),
        "dirty_source_fingerprint": digest.hexdigest(),
        "dirty_source_paths": sorted(git("diff", "HEAD", "--name-only", "--", *watched).splitlines()) + untracked,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }


def run(config_path: Path) -> None:
    """Run all full-scale cases with bounded GPU parallelism and refresh analysis per batch."""
    config = load_config(config_path)
    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest = config.output_root / "manifest.json"
    payload = {
        "config": json.loads(config_path.read_text()), "provenance": git_provenance(),
        "started": datetime.now().astimezone().isoformat(),
    }
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        assert stored["config"] == payload["config"], f"Config differs from {manifest}."
    else:
        manifest.write_text(json.dumps(payload, indent=2))

    jobs = list(itertools.product(range(len(config.architectures)), range(config.repeats)))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.max_parallel_cases, mp_context=context, max_tasks_per_child=1,
    ) as pool:
        for start in range(0, len(jobs), config.max_parallel_cases):
            batch = jobs[start:start + config.max_parallel_cases]
            futures = [
                pool.submit(run_case, config_path, architecture_index, repeat,
                            config.gpu_ids[(start + offset) % len(config.gpu_ids)])
                for offset, (architecture_index, repeat) in enumerate(batch)
            ]
            for future in futures:
                future.result()
            subprocess.run(
                (sys.executable, "-m", "Experiments.warm_start_retraining.analyze", str(config_path)),
                cwd=PROJECT_ROOT, check=True,
            )


def main() -> None:
    """Validate or execute the declared full-scale experiment."""
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
