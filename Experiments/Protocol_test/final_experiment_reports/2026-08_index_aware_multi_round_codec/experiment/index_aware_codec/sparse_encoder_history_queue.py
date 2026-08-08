from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import pandas as pd
import torch


REPORT_ROOT = Path(__file__).resolve().parents[2]
INDEX_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path[:0] = [str(PROJECT_ROOT), str(INDEX_EXPERIMENT_ROOT)]

from index_aware_codec.run_replay import (
    MULTI_MODES,
    V2_MODE_QUANT_OVERRIDES,
    V2_QUANT_OVERRIDES,
    RunMode,
    _protocol_name,
    build_protocol,
)
from index_aware_codec.index_quantizer import IndexAwareDedupedWZQuantizer
from Experiments.Protocol_test.live_outputs import atomic_write, refresh_until, start_refresh_process
from Experiments.Protocol_test.replay import (
    HistorySource,
    build_state_manager,
    load_replay_config,
    load_replay_tensor,
    replace_latest_reconstruction,
    seed_history,
)
from FL_code.FL_core.codec import (
    Access, BaseProtocol, CompressionRecord, simulate_compression)
from FL_code.FL_core.utils import set_global_seed, write_fl_config_snapshot


FROZEN_ROUNDS = 12
SEED = 43
DELTAS = PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas"
FL_CONFIG = INDEX_EXPERIMENT_ROOT / "replay_fl_config.json"
RECORDS_ROOT = REPORT_ROOT / "results/sparse_encoder_history/records"
CACHE_ROOT = PROJECT_ROOT / "_/data/protocol_cache/sparse_codec_queue"
ANALYSIS_SCRIPT = INDEX_EXPERIMENT_ROOT / "analyse_sparse_encoder_history.py"
GPU_IDS = (0, 1)
REFERENCE_WORKERS_PER_GPU = 3
INDEXED_WORKERS_PER_GPU = 1
HEALTH_CHECK_DELAYS_SECONDS = (900, 900, 1800, 1800, 1800, 1800, 1800)
M_TRAIN_OFFSETS = (1, 2)
M_TRAINING_PROBLEM_WEIGHTS = (0.50, 0.33, 0.17)


@dataclass(frozen=True, slots=True)
class Case:
    """One independently replayed client/start pair in a named repeated schedule."""

    variant_start: int
    repeat: int
    client_id: int
    start_round: int

    @property
    def name(self) -> str:
        """Return a stable unique directory name for this case."""
        return (
            f"start_{self.variant_start:02d}_repeat_{self.repeat}_"
            f"client_{self.client_id}_round_{self.start_round:02d}"
        )


@dataclass(frozen=True, slots=True)
class CodecConfig:
    """One codec evaluated with either current-only or the shared M training schedule."""

    mode: RunMode

    def __post_init__(self) -> None:
        assert self.mode in {RunMode.O, RunMode.T, *MULTI_MODES}

    @property
    def name(self) -> str:
        """Return the CSV/cache namespace for this codec configuration."""
        return self.mode.value

    @property
    def train_offsets(self) -> tuple[int, ...]:
        """Return lagged offsets; offset zero is the implicit current training problem."""
        return M_TRAIN_OFFSETS

    @property
    def training_problem_weights(self) -> tuple[float, ...] | None:
        """Return the shared current/offset-1/offset-2 weights for M codecs."""
        return M_TRAINING_PROBLEM_WEIGHTS if self.mode in MULTI_MODES else None

    @property
    def indexed(self) -> bool:
        """Return whether this configuration uses the high-memory indexed quantizer."""
        return self.mode in {RunMode.OMI, RunMode.TMI}


CASES = tuple(
    Case(variant_start, repeat, client_id, variant_start + 5 * client_id)
    for variant_start in (10, 15)
    for repeat in (1, 2)
    for client_id in range(5)
)
CONFIGS = (
    CodecConfig(RunMode.O),
    CodecConfig(RunMode.T),
    CodecConfig(RunMode.OM),
    CodecConfig(RunMode.TM),
    CodecConfig(RunMode.OM_FE),
    CodecConfig(RunMode.TM_FE),
    CodecConfig(RunMode.OMI),
    CodecConfig(RunMode.TMI),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue the pinned sparse 1-train + 12-frozen codec comparison."
    )
    parser.add_argument("--worker", nargs=2, metavar=("CONFIG", "CASE"))
    parser.add_argument("--monitor-pid", type=int)
    parser.add_argument("--refresh-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert args.dry_run + sum(
        option is not None for option in (args.worker, args.monitor_pid, args.refresh_pid)
    ) <= 1
    return args


def _config(name: str) -> CodecConfig:
    """Resolve a configured codec name used by an internal worker process."""
    matches = [config for config in CONFIGS if config.name == name]
    assert len(matches) == 1, f"Unknown codec configuration {name!r}."
    return matches[0]


def _case(name: str) -> Case:
    """Resolve a configured sparse case name used by an internal worker process."""
    matches = [case for case in CASES if case.name == name]
    assert len(matches) == 1, f"Unknown sparse case {name!r}."
    return matches[0]


def _quantizer_overrides(config: CodecConfig) -> dict[str, Any]:
    """Return the pinned final-experiment profile for the selected codec."""
    overrides = (
        {
            "training_seed": SEED,
            **V2_QUANT_OVERRIDES,
            **V2_MODE_QUANT_OVERRIDES.get(config.mode, {}),
        }
        if config.indexed
        else {}
    )
    if config.training_problem_weights is not None:
        overrides["training_problem_weights"] = config.training_problem_weights
    return overrides


def _commit_other_clients_as_raw_history(
    protocol: BaseProtocol,
    round_id: int,
    target_client_id: int,
    client_ids: range,
) -> None:
    """Advance non-target histories from raw-source gradients to preserve the trace."""
    for client_id in client_ids:
        if client_id == target_client_id:
            continue
        protocol.commit_history(
            load_replay_tensor(DELTAS, round_id, client_id),
            CompressionRecord(round_id, client_id, "history_raw_delta"),
            Access.TEMPORAL_TOO,
        )


def _case_csv(config: CodecConfig, case: Case) -> Path:
    """Return the isolated compression-record path for a case."""
    return RECORDS_ROOT / "cases" / config.name / case.name / "compression_records.csv"


def _case_complete(config: CodecConfig, case: Case) -> bool:
    """Validate and report whether one resumable case already has all requested rows."""
    path = _case_csv(config, case)
    if not path.is_file():
        return False
    records = pd.read_csv(path)
    expected_rounds = list(range(case.start_round, case.start_round + FROZEN_ROUNDS + 1))
    assert records["round_id"].tolist() == expected_rounds[: len(records)], (
        f"Unexpected rounds in {path}."
    )
    assert records["client_id"].tolist() == [case.client_id] * len(records), (
        f"Unexpected clients in {path}."
    )
    return len(records) == len(expected_rounds)


def run_case(config: CodecConfig, case: Case) -> Path:
    """Train once and replay twelve frozen rounds for one client/start case."""
    if _case_complete(config, case):
        print(f"SKIP {config.name} {case.name}", flush=True)
        return _case_csv(config, case)

    set_global_seed(SEED)
    case_dir = _case_csv(config, case).parent
    cfg = load_replay_config(
        FL_CONFIG,
        _protocol_name(config.mode, case.start_round, config.train_offsets),
        f"sparse_queue_{config.name}_{case.name}",
        case_dir,
        SEED,
    )
    assert 0 <= case.client_id < cfg.num_clients
    cache_dir = CACHE_ROOT / config.name
    cfg.protocol_cache_dir = cache_dir
    state_manager = build_state_manager(cfg)
    protocol = build_protocol(
        config.mode,
        case.start_round,
        _quantizer_overrides(config),
        state_manager.get_slices(),
        cache_dir,
        train_offsets=config.train_offsets,
    )
    client_ids = range(cfg.num_clients)
    seed_history(
        protocol,
        HistorySource.RAW_DELTAS,
        case.start_round,
        client_ids,
        DELTAS,
        DELTAS,
    )

    case_dir.mkdir(parents=True, exist_ok=True)
    write_fl_config_snapshot(cfg, case_dir)
    (case_dir / "protocol_config.json").write_text(
        json.dumps(protocol.config_snapshot(), indent=2)
    )
    for round_id in range(case.start_round, case.start_round + FROZEN_ROUNDS + 1):
        delta = load_replay_tensor(DELTAS, round_id, case.client_id)
        assert delta.numel() == state_manager.param_count
        print(f"RUN {config.name} {case.name} round={round_id}", flush=True)
        simulate_compression(
            protocol,
            delta,
            case.client_id,
            round_id,
            state_manager,
            case_dir,
            {},
            (),
            (),
        )
        replace_latest_reconstruction(protocol, delta, round_id, case.client_id)
        _commit_other_clients_as_raw_history(
            protocol, round_id, case.client_id, client_ids
        )
    assert _case_complete(config, case)
    return _case_csv(config, case)


def aggregate(config: CodecConfig) -> Path:
    """Atomically combine every completed sparse trajectory into one codec CSV."""
    frames: list[pd.DataFrame] = []
    for case_id, case in enumerate(CASES):
        if not _case_complete(config, case):
            continue
        records = pd.read_csv(_case_csv(config, case))
        records.insert(0, "start_round", case.start_round)
        records.insert(0, "client_repeat", case.repeat)
        records.insert(0, "variant_start", case.variant_start)
        records.insert(0, "case_id", case_id)
        records.insert(
            0,
            "training_problem_weights",
            json.dumps(config.training_problem_weights or (1.0,)),
        )
        records.insert(
            0,
            "training_round_offsets",
            json.dumps((0, *config.train_offsets) if config.mode in MULTI_MODES else (0,)),
        )
        records.insert(0, "codec_config", config.name)
        frames.append(records)
    assert frames, f"Cannot aggregate {config.name}: it has no completed cases."
    output_dir = RECORDS_ROOT / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "compression_records.csv"
    combined = pd.concat(frames, ignore_index=True)
    assert len(combined) == len(frames) * (FROZEN_ROUNDS + 1)
    atomic_write(output, lambda temporary: combined.to_csv(temporary, index=False))
    return output


def _completed_case_signature() -> tuple[tuple[str, str], ...]:
    """Return the completed case identities that determine aggregate and plot contents."""
    return tuple(
        (config.name, case.name)
        for config in CONFIGS
        for case in CASES
        if _case_complete(config, case)
    )


def _refresh_outputs() -> None:
    """Refresh every non-empty partial aggregate and all sparse-analysis plots."""
    completed_by_config = {
        config.name: sum(_case_complete(config, case) for case in CASES)
        for config in CONFIGS
    }
    outputs = [
        aggregate(config)
        for config in CONFIGS
        if completed_by_config[config.name]
    ]
    assert outputs, "Cannot refresh sparse outputs before the first case completes."
    print(
        f"REFRESH aggregates={completed_by_config}",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "-u", str(ANALYSIS_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=os.environ | {"MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )


def _worker_command(config: CodecConfig, case: Case) -> list[str]:
    """Build the isolated command for one queued case."""
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", config.name, case.name]


def _run_job(config: CodecConfig, case: Case, gpu: int) -> None:
    """Run one case on a selected GPU and retain its complete log."""
    log_path = RECORDS_ROOT / "logs" / config.name / f"{case.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"START gpu={gpu} {config.name} {case.name}", flush=True)
    with log_path.open("w") as log:
        subprocess.run(
            _worker_command(config, case),
            cwd=PROJECT_ROOT,
            env=os.environ
            | {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"DONE gpu={gpu} {config.name} {case.name}", flush=True)


def _run_slot(gpu: int, jobs: Sequence[tuple[CodecConfig, Case]]) -> list[str]:
    """Drain one fixed GPU slot so per-device concurrency cannot exceed its limit."""
    failures: list[str] = []
    for config, case in jobs:
        try:
            _run_job(config, case, gpu)
        except subprocess.CalledProcessError as error:
            failures.append(f"{config.name}/{case.name}: exit {error.returncode}")
    return failures


def _run_phase(configs: Sequence[CodecConfig], workers_per_gpu: int) -> None:
    """Run a codec phase at its safe maximum GPU concurrency, with repeats in separate waves."""
    slots = tuple(gpu for gpu in GPU_IDS for _ in range(workers_per_gpu))
    failures: list[str] = []
    for repeat in (1, 2):
        jobs = [(config, case) for config in configs for case in CASES if case.repeat == repeat]
        slot_jobs = tuple(jobs[index::len(slots)] for index in range(len(slots)))
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            failures.extend(
                failure
                for slot_failures in pool.map(_run_slot, slots, slot_jobs)
                for failure in slot_failures
            )

    missing = [
        (config, case)
        for config in configs
        for case in CASES
        if not _case_complete(config, case)
    ]
    if missing:
        print(f"RECOVER missing_cases={len(missing)} prior_failures={failures}", flush=True)
        slot_jobs = tuple(missing[index::len(slots)] for index in range(len(slots)))
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            recovery_failures = [
                failure
                for slot_failures in pool.map(_run_slot, slots, slot_jobs)
                for failure in slot_failures
            ]
        still_missing = [
            f"{config.name}/{case.name}"
            for config, case in missing
            if not _case_complete(config, case)
        ]
        assert not still_missing, (
            f"Recovery left incomplete cases {still_missing}; failures: {recovery_failures}."
        )


def _manifest() -> dict[str, Any]:
    """Return the exact shared case and codec configuration for provenance and resume checks."""
    storage_manifest = json.loads((DELTAS / "storage_manifest.json").read_text())
    return {
        "schema_version": 1,
        "indexed_candidate_failure_tolerance": (
            IndexAwareDedupedWZQuantizer.candidate_failure_tolerance
        ),
        "history_source": HistorySource.RAW_DELTAS.value,
        "replace_commits_with_raw": True,
        "all_client_history_advanced": True,
        "frozen_rounds": FROZEN_ROUNDS,
        "seed": SEED,
        "deltas": str(DELTAS.resolve()),
        "delta_storage_manifest": storage_manifest,
        "cases": [
            {
                "name": case.name,
                "variant_start": case.variant_start,
                "repeat": case.repeat,
                "client_id": case.client_id,
                "start_round": case.start_round,
                "end_round": case.start_round + FROZEN_ROUNDS,
            }
            for case in CASES
        ],
        "codec_configs": [
            {
                "name": config.name,
                "mode": config.mode.value,
                "training_round_offsets": (
                    [0, *config.train_offsets] if config.mode in MULTI_MODES else [0]
                ),
                "training_problem_weights": list(config.training_problem_weights or (1.0,)),
                "quantizer_overrides": _quantizer_overrides(config),
            }
            for config in CONFIGS
        ],
    }


def run_queue() -> None:
    """Run current-only, weighted reference, then indexed codecs as resumable phases."""
    assert torch.cuda.device_count() >= len(GPU_IDS), "The pinned queue requires two GPUs."
    manifest = json.loads(json.dumps(_manifest()))
    RECORDS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RECORDS_ROOT / "test_config.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        assert {
            key: value for key, value in existing.items() if key != "codec_configs"
        } == {
            key: value for key, value in manifest.items() if key != "codec_configs"
        }, f"Existing queue configuration differs: {manifest_path}."
        expected_configs = {
            config["name"]: config for config in manifest["codec_configs"]
        }
        assert all(
            expected_configs.get(config["name"]) == config
            for config in existing["codec_configs"]
        ), f"Existing codec configuration differs: {manifest_path}."
        if existing != manifest:
            manifest_path.write_text(json.dumps(manifest, indent=2))
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    refresh_log = RECORDS_ROOT / "refresh.log"
    refresher_pid = start_refresh_process(
        Path(__file__).resolve(), os.getpid(), PROJECT_ROOT, refresh_log
    )
    print(f"REFRESHER pid={refresher_pid} log={refresh_log}", flush=True)
    _run_phase(CONFIGS[:2], REFERENCE_WORKERS_PER_GPU)
    _run_phase(CONFIGS[2:6], REFERENCE_WORKERS_PER_GPU)
    _run_phase(CONFIGS[6:], INDEXED_WORKERS_PER_GPU)


def _health_snapshot(queue_pid: int, check_minute: int) -> dict[str, Any]:
    """Capture process, output, error-log, GPU, and disk health for the detached queue."""
    try:
        os.kill(queue_pid, 0)
        queue_alive = True
    except ProcessLookupError:
        queue_alive = False
    error_markers = (
        "Traceback",
        "out of memory",
        "Queued phase failed",
        "REFRESH FAILED",
    )
    logs = (
        *((RECORDS_ROOT / "logs").glob("*/*.log")),
        RECORDS_ROOT / "queue.log",
        RECORDS_ROOT / "refresh.log",
    )
    error_logs = [
        str(path.relative_to(RECORDS_ROOT))
        for path in logs
        if path.is_file() and any(marker in path.read_text(errors="replace") for marker in error_markers)
    ]
    worker_count = int(
        subprocess.run(
            ("pgrep", "-fc", f"{Path(__file__).name} --worker"),
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        or 0
    )
    gpu_status = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    completed_cases = len(_completed_case_signature())
    aggregates = len(tuple(RECORDS_ROOT.glob("*/compression_records.csv")))
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    return {
        "check_minute": check_minute,
        "timestamp": pd.Timestamp.now(tz="Europe/Brussels").isoformat(),
        "queue_pid": queue_pid,
        "queue_alive": queue_alive,
        "worker_count": worker_count,
        "completed_cases": completed_cases,
        "expected_cases": len(CONFIGS) * len(CASES),
        "aggregate_csvs": aggregates,
        "expected_aggregate_csvs": len(CONFIGS),
        "error_logs": error_logs,
        "gpu_status": gpu_status,
        "free_disk_gib": round(free_gib, 2),
        "healthy": not error_logs
        and (
            (queue_alive and worker_count > 0)
            or completed_cases == len(CONFIGS) * len(CASES)
        ),
    }


def monitor_queue(queue_pid: int) -> None:
    """Record checks at 15, 30, then every 30 minutes through three hours."""
    health_path = RECORDS_ROOT / "health_checks.jsonl"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed_seconds = 0
    with health_path.open("a") as health_log:
        for delay in HEALTH_CHECK_DELAYS_SECONDS:
            time.sleep(delay)
            elapsed_seconds += delay
            snapshot = _health_snapshot(queue_pid, elapsed_seconds // 60)
            line = json.dumps(snapshot)
            health_log.write(f"{line}\n")
            health_log.flush()
            print(line, flush=True)


def main() -> None:
    """Run a worker, monitor, output refresher, dry run, or the complete queue."""
    args = _parse_args()
    if args.worker is not None:
        run_case(_config(args.worker[0]), _case(args.worker[1]))
    elif args.monitor_pid is not None:
        monitor_queue(args.monitor_pid)
    elif args.refresh_pid is not None:
        refresh_until(args.refresh_pid, _completed_case_signature, _refresh_outputs)
    elif args.dry_run:
        print(json.dumps(_manifest(), indent=2))
    else:
        run_queue()


if __name__ == "__main__":
    main()
