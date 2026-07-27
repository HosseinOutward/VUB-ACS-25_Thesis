from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.codec import BaseProtocol, create_protocol, simulate_compression
from FL_code.FL_core.utils import _prepare_records_dir, set_global_seed, write_fl_config_snapshot
from test_protocol import (
    HistorySource,
    _build_state_manager,
    _load_replay_config,
    _load_tensor,
    replace_latest_reconstruction,
    seed_history,
)


BINS_PER_PLANE = 3
NUM_PLANES = 3
F_ROUNDS = 2  # the schedule under test is one training round followed by two frozen rounds

DEFAULT_DELTAS = PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas"
DEFAULT_RECORDS_ROOT = Path(__file__).parent / "exp_records"
DEFAULT_RAW_ROOT = Path(__file__).parent / "exp_records_multistart"
DEFAULT_FL_CONFIG = DEFAULT_RECORDS_ROOT / "run3_real_raw_o_nf4" / "fl_config.json"
DEFAULT_PROTOCOL_CACHE_ROOT = PROJECT_ROOT / "_/data/protocol_cache/multistart"

# Both sets are pinned rather than leaning on WZcfgQuant defaults: which set the defaults
# happen to hold has already changed once, and it must not silently redefine an experiment.
RUN3_QUANT_OVERRIDES: dict[str, Any] = {  # runs 3 and 4
    "train_epochs": 140,
    "reconst_ld": 72.31601255189986,
    "train_sample_size": 550_000,
    "train_batch_size": 6250,
    "lr": 0.0056728196862653865,
    "lr_step": 43,
    "tau": 5.5,
}
RUN5_QUANT_OVERRIDES: dict[str, Any] = {  # runs 5 and 6
    "train_epochs": 120,
    "reconst_ld": 36.73683767756248,
    "train_sample_size": 750_000,
    "train_batch_size": 12_500,
    "lr": 0.004530430161538338,
    "lr_step": 37,
    "tau": 2.3406302935091077,
}

EXPERIMENTS: dict[str, tuple[str, str, Mapping[str, Any]]] = {
    "run3hp_off_o": ("wz_cancer", "O", RUN3_QUANT_OVERRIDES),
    "run3hp_off_t": ("wz_cancer|replace_O_with='T'", "T", RUN3_QUANT_OVERRIDES),
    "run5hp_off_o": ("wz_cancer", "O", RUN5_QUANT_OVERRIDES),
    "run5hp_off_t": ("wz_cancer|replace_O_with='T'", "T", RUN5_QUANT_OVERRIDES),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved client deltas through an O/T + F*n schedule from many start rounds. "
            "Without --start-round this orchestrates every start round of every experiment in "
            "parallel and then aggregates each experiment into one records file."
        )
    )
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        choices=tuple(EXPERIMENTS),
        help="Experiment to run; repeatable. Defaults to all of them.",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        help="Run only this one replay in the current process, as spawned by the orchestrator.",
    )
    parser.add_argument("--first-start-round", type=int, default=12)
    parser.add_argument("--last-round", type=int, default=20, help="Last round with saved deltas.")
    parser.add_argument("--deltas", type=Path, default=DEFAULT_DELTAS)
    parser.add_argument("--fl-config", type=Path, default=DEFAULT_FL_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--protocol-cache-root", type=Path, default=DEFAULT_PROTOCOL_CACHE_ROOT)
    parser.add_argument(
        "--quant-override",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Extra WZcfgQuant override applied to every experiment, mainly for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--per-gpu", type=int, default=3, help="Concurrent replays per visible GPU.")
    parser.add_argument(
        "--skip-aggregate", action="store_true", help="Leave the per-start records unaggregated."
    )
    args = parser.parse_args()
    args.experiments = args.experiments or list(EXPERIMENTS)
    return args


def _extra_overrides(tokens: list[str]) -> dict[str, Any]:
    """Parse repeated FIELD=VALUE command-line overrides into WZcfgQuant keyword arguments."""
    overrides: dict[str, Any] = {}
    for token in tokens:
        field, separator, value = token.partition("=")
        assert separator and field, f"--quant-override {token!r} must be FIELD=VALUE."
        overrides[field] = ast.literal_eval(value)
    return overrides


def _round_ids(start_round: int, last_round: int) -> range:
    """Return the training round and the frozen rounds that still fit before the saved deltas end."""
    return range(start_round, min(start_round + 1 + F_ROUNDS, last_round + 1))


def _train_round_name(codec: str, overrides: Mapping[str, Any]) -> str:
    """Build the training round selector, carrying quantizer overrides as extra option tokens."""
    return "|".join((
        f"{codec}|bins_per_plane={BINS_PER_PLANE}|num_planes={NUM_PLANES}",
        *(f"{field}={value!r}" for field, value in overrides.items()),
    ))


def build_protocol(
    experiment: str,
    overrides: Mapping[str, Any],
    sd_slices: list[slice] | None = None,
    cache_dir: Path | None = None,
    start_round: int | None = None,
) -> BaseProtocol:
    """Create the experiment protocol with its routine cycle replaced by one training round plus F rounds.

    The protocol picks a round codec from the round id modulo the routine length, so landing the
    training round on an arbitrary start round means rotating the cycle rather than the replay
    window. The warmup rounds are left untouched: every replay starts well past them.
    """
    protocol_name, codec, experiment_overrides = EXPERIMENTS[experiment]
    protocol = create_protocol(protocol_name, sd_slices, cache_dir=cache_dir)
    cycle = (_train_round_name(codec, {**experiment_overrides, **overrides}), *("F",) * F_ROUNDS)
    phase = (
        0 if start_round is None
        else len(cycle) - (start_round - len(protocol.warmup_round_codecs)) % len(cycle)
    )
    protocol.routine_round_codecs = cycle[phase:] + cycle[:phase]
    protocol._round_codec_classes = protocol._validate_round_codecs()
    assert start_round is None or protocol._get_curr_round_codec_name(start_round)[2] == cycle[0]
    return protocol


def run_single(experiment: str, start_round: int, args: argparse.Namespace) -> None:
    """Replay one start round of one experiment, seeding history with the preceding raw deltas."""
    records_dir = args.raw_root / experiment / f"start_{start_round:02d}"
    rounds = _round_ids(start_round, args.last_round)
    set_global_seed(args.seed)
    cfg = _load_replay_config(
        args.fl_config,
        EXPERIMENTS[experiment][0],
        f"{experiment}_start_{start_round:02d}",
        records_dir,
        args.seed,
    )
    cfg.protocol_cache_dir = args.protocol_cache_root / experiment
    cfg.protocol_cache_overwrite = False

    csv_path = records_dir / "compression_records.csv"
    expected_rows = len(rounds) * cfg.num_clients
    if csv_path.is_file():
        if len(pd.read_csv(csv_path)) == expected_rows:
            print(f"{experiment} start {start_round}: already complete, skipping.")
            return
        print(f"{experiment} start {start_round}: discarding an incomplete {csv_path}.")
        csv_path.unlink()

    state_manager = _build_state_manager(cfg)
    protocol = build_protocol(
        experiment,
        _extra_overrides(args.quant_override),
        state_manager.get_slices(),
        cfg.protocol_cache_dir,
        start_round,
    )
    client_ids = range(cfg.num_clients)
    seed_history(protocol, HistorySource.RAW_DELTAS, start_round, client_ids, args.deltas, args.deltas)

    records_dir.mkdir(parents=True, exist_ok=True)
    write_fl_config_snapshot(cfg, records_dir)
    (records_dir / "protocol_config.json").write_text(json.dumps(protocol.config_snapshot(), indent=2))

    for round_id in rounds:
        for client_id in client_ids:
            delta = _load_tensor(args.deltas, round_id, client_id)
            assert delta.numel() == state_manager.param_count, (
                f"round {round_id}, client {client_id} has {delta.numel()} values; "
                f"the configured model has {state_manager.param_count}."
            )
            print(f"Simulating {experiment} start {start_round}: round {round_id}, client {client_id}")
            simulate_compression(
                protocol, delta, client_id, round_id, state_manager, records_dir, {}, (), ()
            )
            # Runs 3 to 6 kept the inference side-information window on raw gradients; match them.
            replace_latest_reconstruction(protocol, delta, round_id, client_id)

    print(f"\nRecords written to {csv_path}")


def aggregate(experiment: str, args: argparse.Namespace) -> Path:
    """Concatenate every start-round replay of one experiment into a single run directory."""
    start_dirs = sorted(p for p in (args.raw_root / experiment).glob("start_*") if p.is_dir())
    assert start_dirs, f"No per-start records to aggregate under {args.raw_root / experiment}."
    records = pd.concat(
        [
            pd.read_csv(start_dir / "compression_records.csv").assign(
                start_round=int(start_dir.name.removeprefix("start_"))
            )
            for start_dir in start_dirs
        ],
        ignore_index=True,
    )
    records.insert(0, "start_round", records.pop("start_round"))

    cfg = _load_replay_config(
        args.fl_config, EXPERIMENTS[experiment][0], experiment, args.records_root, args.seed
    )
    cfg.protocol_cache_dir = args.protocol_cache_root / experiment
    _prepare_records_dir(cfg)
    records.to_csv(cfg.records_dir / "compression_records.csv", index=False)
    # _prepare_records_dir snapshots the unmodified protocol schedule, so replace it with the
    # cycle actually replayed, shown unrotated because the phase differs per start round.
    (cfg.records_dir / "protocol_config.json").write_text(
        json.dumps(
            build_protocol(experiment, _extra_overrides(args.quant_override)).config_snapshot(),
            indent=2,
        )
    )
    print(f"Aggregated {len(records)} rows from {len(start_dirs)} start rounds into {cfg.records_dir}")
    return cfg.records_dir


def _run_job(experiment: str, start_round: int, gpu: str, args: argparse.Namespace) -> None:
    """Run one replay in its own process so its memory and GPU context are released afterwards."""
    log_path = args.raw_root / experiment / f"start_{start_round:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--experiment", experiment,
        "--start-round", str(start_round),
        "--first-start-round", str(args.first_start_round),
        "--last-round", str(args.last_round),
        "--deltas", str(args.deltas),
        "--fl-config", str(args.fl_config),
        "--raw-root", str(args.raw_root),
        "--records-root", str(args.records_root),
        "--protocol-cache-root", str(args.protocol_cache_root),
        "--seed", str(args.seed),
        *(token for override in args.quant_override for token in ("--quant-override", override)),
    ]
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": gpu,
        "OMP_NUM_THREADS": str(args.cpu_threads),
        "MKL_NUM_THREADS": str(args.cpu_threads),
    }
    with log_path.open("w") as log_file:
        subprocess.run(
            command, check=True, cwd=PROJECT_ROOT, env=environment,
            stdout=log_file, stderr=subprocess.STDOUT,
        )


def orchestrate(args: argparse.Namespace) -> None:
    """Run every start round of every experiment across the visible GPUs, then aggregate."""
    gpu_count = max(torch.cuda.device_count(), 1)
    gpu_slots: queue.Queue[str] = queue.Queue()
    for slot in range(gpu_count * args.per_gpu):
        gpu_slots.put(str(slot % gpu_count))
    args.cpu_threads = max(1, (os.cpu_count() or 1) // gpu_slots.qsize())

    start_rounds = range(args.first_start_round, args.last_round + 1)
    jobs = [(experiment, start) for start in start_rounds for experiment in args.experiments]
    print(
        f"Running {len(jobs)} replays over {gpu_count} GPU(s) at {args.per_gpu} each "
        f"({args.cpu_threads} CPU threads per replay); logs in {args.raw_root}."
    )

    def worker(job: tuple[str, int]) -> tuple[str, int] | None:
        gpu = gpu_slots.get()
        try:
            _run_job(*job, gpu, args)
        except subprocess.CalledProcessError:
            return job
        finally:
            gpu_slots.put(gpu)
        return None

    with ThreadPoolExecutor(max_workers=gpu_slots.qsize()) as pool:
        failures = [job for job in pool.map(worker, jobs) if job is not None]

    failed_experiments = {experiment for experiment, _start in failures}
    for job in failures:
        print(f"FAILED: experiment {job[0]} start round {job[1]}; see its log under {args.raw_root}.")
    if not args.skip_aggregate:
        for experiment in args.experiments:
            if experiment in failed_experiments:
                print(f"Skipping aggregation of {experiment}: some of its replays failed.")
                continue
            aggregate(experiment, args)
    assert not failures, f"{len(failures)} of {len(jobs)} replays failed."


def main() -> None:
    """Replay one start round, or orchestrate and aggregate all of them."""
    args = _parse_args()
    if args.start_round is None:
        orchestrate(args)
        return
    assert len(args.experiments) == 1, "--start-round runs exactly one experiment."
    run_single(args.experiments[0], args.start_round, args)


if __name__ == "__main__":
    main()
