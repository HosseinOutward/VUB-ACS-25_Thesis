from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[6]
REPORT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from .rd_compare import (
    compare_baseline,
    compare_experiment,
)
from .round_codecs import (
    OM_FE_RoundCodec,
    OM_RoundCodec,
    OMI_RoundCodec,
    OI_RoundCodec,
    TM_FE_RoundCodec,
    TM_RoundCodec,
    TMI_RoundCodec,
    TI_RoundCodec,
)
from .index_quantizer import IndexAwareWZcfgQuant
from Experiments.Protocol_test.replay import (
    HistorySource,
    build_state_manager,
    load_replay_config,
    load_replay_tensor,
    seed_history,
)
from FL_code.FL_core.codec import BaseProtocol, create_protocol, simulate_compression
from FL_code.FL_core.utils import set_global_seed, write_fl_config_snapshot


BINS_PER_PLANE = 3
NUM_PLANES = 3
DEFAULT_DELTAS = PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas"
DEFAULT_FL_CONFIG = Path(__file__).resolve().parents[1] / "replay_fl_config.json"
DEFAULT_RECORDS_ROOT = REPORT_ROOT / "results/archive/index_aware_v1"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "_/data/protocol_cache/index_aware/v1"
DEFAULT_V2_RECORDS_ROOT = REPORT_ROOT / "results/records"
DEFAULT_V2_CACHE_ROOT = PROJECT_ROOT / "_/data/protocol_cache/flagship_comparison"
RUNTIME_SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[1] / "replay.py",
    *(Path(__file__).resolve().parent / name for name in (
        "encoder_history_quantizer.py",
        "index_aware_model.py",
        "index_quantizer.py",
        "round_codecs.py",
    )),
    *(PROJECT_ROOT / "FL_code/cancer_protocol" / name for name in (
        "brent_wz_models.py",
        "cancer_protocol.py",
        "prior_code.py",
        "round_codecs.py",
        "wz_quantizer.py",
    )),
)
DEFAULT_ORIGINAL_O = REPORT_ROOT / "results/records/O/compression_records.csv"
DEFAULT_ORIGINAL_T = REPORT_ROOT / "results/records/T/compression_records.csv"


class RunMode(StrEnum):
    """Replay mode selected by a user or by the two-GPU orchestrator."""

    BASELINE = "baseline"
    O = "O"
    T = "T"
    OI = "OI"
    TI = "TI"
    OM = "OM"
    TM = "TM"
    OM_FE = "OM_FE"
    TM_FE = "TM_FE"
    OMI = "OMI"
    TMI = "TMI"
    FULL = "full"
    FULL_M = "full_m"


class ExperimentProfile(StrEnum):
    """Pinned indexed-model experiment configuration and storage namespace."""

    V1 = "v1"
    V2 = "v2"


V1_QUANT_OVERRIDES: dict[str, Any] = {
    "index_cache_version": "index-side-information/v1",
    "hash_table_count": 6,
    "hash_bucket_count": 65_536,
    "hash_dim": 16,
    "route_count": 48,
    "embedding_dim": 16,
    "hash_seed": 0,
    "routing_temperature": 1.0,
    "hidden_dim": 88,
    "layers": 7,
    "train_epochs": 140,
}
V2_QUANT_OVERRIDES: dict[str, Any] = {
    "index_cache_version": "index-side-information/v2",
    "hash_table_count": 8,
    "hash_dim": 24,
    "route_count": 64,
    "embedding_dim": 24,
    "hidden_dim": 112,
    "layers": 8,
    "train_epochs": 180,
    "raw_index_bits": 24,
    "raw_index_scale": 0.01,
    "routing_temperature_start": 2.0,
    "routing_temperature_end": 0.8,
    "routing_dense_warmup_fraction": 0.1,
    "routing_top_k": 64,
    "routing_balance_loss_weight": 1e-3,
    "routing_z_loss_weight": 1e-3,
    "embedding_diversity_loss_weight": 1e-4,
    "index_dropout": 0.1,
    "routed_scale_initial": 0.01,
    "routed_scale_max": 0.05,
    "content_routing": False,
    "current_problem_sampling_fraction": 0.75,
    "training_match_bits": 32,
    "quantizer_train_repeats": 2,
    "prior_train_repeats": 2,
}
V2_MODE_QUANT_OVERRIDES: dict[RunMode, dict[str, Any]] = {
    RunMode.TMI: {
        "routed_scale_initial": 0.0002,
        "routed_scale_max": 0.001,
        "current_problem_sampling_fraction": 0.95,
    },
}
V2_VALIDATION_OVERRIDES: dict[str, Any] = {
    "train_epochs": 2,
    "train_sample_size": 4_096,
    "train_batch_size": 512,
    "quantizer_train_repeats": 1,
    "prior_train_repeats": 1,
}
ORACLE_MODES = frozenset({
    RunMode.BASELINE, RunMode.O, RunMode.OI, RunMode.OM, RunMode.OMI, RunMode.OM_FE
})
INDEX_MODES = frozenset({RunMode.OI, RunMode.TI, RunMode.OMI, RunMode.TMI})
REFERENCE_MODES = frozenset({
    RunMode.O, RunMode.T, RunMode.OM, RunMode.TM, RunMode.OM_FE, RunMode.TM_FE
})
MULTI_MODES = frozenset({
    RunMode.OM, RunMode.TM, RunMode.OMI, RunMode.TMI, RunMode.OM_FE, RunMode.TM_FE
})
ORCHESTRATION_MODES = frozenset({RunMode.FULL, RunMode.FULL_M})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay index-aware OI/TI or multi-target OMI/TMI codecs from rounds 12 through 20."
        )
    )
    parser.add_argument("--mode", type=RunMode, choices=tuple(RunMode), default=RunMode.FULL)
    parser.add_argument(
        "--profile",
        type=ExperimentProfile,
        choices=tuple(ExperimentProfile),
        default=ExperimentProfile.V1,
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Run one client/round with the v2 architecture and short training settings.",
    )
    parser.add_argument("--client-id", type=int, action="append", dest="client_ids")
    parser.add_argument("--start-round", type=int, default=12)
    parser.add_argument("--end-round", type=int, default=20)
    parser.add_argument("--deltas", type=Path, default=DEFAULT_DELTAS)
    parser.add_argument("--fl-config", type=Path, default=DEFAULT_FL_CONFIG)
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--original-o", type=Path, default=DEFAULT_ORIGINAL_O)
    parser.add_argument("--original-t", type=Path, default=DEFAULT_ORIGINAL_T)
    parser.add_argument(
        "--train-offset",
        type=int,
        action="append",
        dest="train_offsets",
        help="Positive lagged training offset; repeatable. Defaults to 1, 2, and 3.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cache-overwrite", action="store_true")
    parser.add_argument("--overwrite-records", action="store_true")
    parser.add_argument(
        "--quant-override",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Typed IndexAwareWZcfgQuant override; repeat for multiple fields.",
    )
    args = parser.parse_args()
    train_offsets = tuple(sorted(args.train_offsets or (1, 2, 3)))
    assert all(offset > 0 for offset in train_offsets), "train offsets must be positive."
    assert len(set(train_offsets)) == len(train_offsets), "train offsets must be unique."
    args.train_offsets = train_offsets
    args.records_root = args.records_root or (
        DEFAULT_V2_RECORDS_ROOT if args.profile is ExperimentProfile.V2 else DEFAULT_RECORDS_ROOT
    )
    args.cache_root = args.cache_root or (
        DEFAULT_V2_CACHE_ROOT if args.profile is ExperimentProfile.V2 else DEFAULT_CACHE_ROOT
    )
    if args.profile is ExperimentProfile.V2:
        assert args.records_root.resolve() != DEFAULT_RECORDS_ROOT.resolve(), (
            "v2 records-root must be isolated from the v1 records root."
        )
        assert args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve(), (
            "v2 cache-root must be isolated from the v1 cache root."
        )
    if args.validation:
        assert args.profile is ExperimentProfile.V2, "--validation requires --profile v2."
        assert args.mode in INDEX_MODES, "--validation runs one direct indexed codec mode."
        assert args.client_ids in (None, [0]), "--validation is pinned to client 0."
        args.client_ids = [0]
        args.end_round = args.start_round
        args.records_root /= "validation"
        args.cache_root /= "validation"
    return args


def _quant_overrides(tokens: Sequence[str]) -> dict[str, Any]:
    """Parse FIELD=VALUE tokens with Python literal values."""
    overrides: dict[str, Any] = {}
    for token in tokens:
        field, separator, value = token.partition("=")
        assert separator and field and field not in overrides, (
            f"--quant-override {token!r} must name one previously unset FIELD=VALUE."
        )
        overrides[field] = ast.literal_eval(value)
    return overrides


def _resolved_quant_overrides(args: argparse.Namespace, mode: RunMode) -> dict[str, Any]:
    """Merge explicit overrides with the pinned profile without weakening profile semantics."""
    if mode is RunMode.BASELINE:
        return {}
    explicit = _quant_overrides(args.quant_override)
    if mode in REFERENCE_MODES:
        return explicit
    if args.profile is ExperimentProfile.V1:
        return {"training_seed": args.seed} | V1_QUANT_OVERRIDES | explicit

    pinned = V2_QUANT_OVERRIDES | V2_MODE_QUANT_OVERRIDES.get(mode, {}) | (
        V2_VALIDATION_OVERRIDES if args.validation else {}
    )
    conflicts = set(explicit) & set(pinned)
    assert not conflicts, (
        f"v2 profile fields are pinned and cannot be overridden: {sorted(conflicts)}."
    )
    return {"training_seed": args.seed} | pinned | explicit


def _source_revision() -> str:
    """Return the repository revision recorded in an experiment manifest."""
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_tree_hash() -> str:
    """Hash the runtime sources so dirty-tree experiments cannot reuse stale records."""
    digest = hashlib.sha256()
    for path in RUNTIME_SOURCE_FILES:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _experiment_manifest(
    args: argparse.Namespace,
    mode: RunMode,
    client_ids: tuple[int, ...],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the complete v2 experiment identity used to validate record reuse."""
    quantizer_config = IndexAwareWZcfgQuant(
        bins_per_plane=BINS_PER_PLANE,
        num_planes=NUM_PLANES,
        **overrides,
    ).model_dump(
        mode="json",
        exclude={"norm_slices", "cache_dir", "cache_overwrite"},
    )
    return {
        "schema_version": 1,
        "profile": args.profile.value,
        "validation": args.validation,
        "source_revision": _source_revision(),
        "source_tree_hash": _source_tree_hash(),
        "mode": mode.value,
        "protocol_name": _protocol_name(mode, args.start_round, args.train_offsets),
        "rounds": [args.start_round, args.end_round],
        "client_ids": list(client_ids),
        "train_offsets": list(args.train_offsets),
        "seed": args.seed,
        "deltas": str(args.deltas.resolve()),
        "fl_config": str(args.fl_config.resolve()),
        "records_root": str(args.records_root.resolve()),
        "cache_dir": str((args.cache_root / mode.value).resolve()),
        "quantizer_config": quantizer_config,
    }


def _validate_manifest(output_dir: Path, expected: Mapping[str, Any]) -> None:
    """Require existing v2 records to have exactly the resolved experiment identity."""
    path = output_dir / "experiment_manifest.json"
    assert path.is_file(), (
        f"Existing v2 output is missing {path}; use --overwrite-records in the isolated v2 root."
    )
    actual = json.loads(path.read_text())
    assert actual == expected, (
        f"Existing v2 manifest does not match the requested experiment: {path}. "
        "Use a new v2 records root or --overwrite-records."
    )


def _protocol_name(
    mode: RunMode,
    start_round: int,
    train_offsets: tuple[int, ...],
) -> str:
    assert mode not in ORCHESTRATION_MODES
    if mode in (RunMode.BASELINE, RunMode.O):
        return f"wz_cancer|train_at_round={start_round}"
    options = f"replace_O_with={mode.value!r}|train_at_round={start_round}"
    if mode in MULTI_MODES:
        options += f"|train_offsets={train_offsets!r}"
    return f"wz_cancer|{options}"


def _training_round_name(mode: RunMode, overrides: Mapping[str, Any]) -> str:
    codec = "O" if mode is RunMode.BASELINE else mode.value
    return "|".join((
        f"{codec}|bins_per_plane={BINS_PER_PLANE}|num_planes={NUM_PLANES}",
        *(f"{field}={value!r}" for field, value in overrides.items()),
    ))


def _output_name(mode: RunMode) -> str:
    """Return the concise directory name used by final comparison records."""
    return "O_sanity" if mode is RunMode.BASELINE else mode.value


def build_protocol(
    mode: RunMode,
    start_round: int,
    overrides: Mapping[str, Any],
    sd_slices: Sequence[slice],
    cache_dir: Path,
    cache_overwrite: bool = False,
    train_offsets: tuple[int, ...] = (1, 2, 3),
) -> BaseProtocol:
    """Create a protocol that trains exactly once at start_round and freezes thereafter."""
    protocol = create_protocol(
        _protocol_name(mode, start_round, train_offsets),
        sd_slices,
        cache_dir=cache_dir,
        cache_overwrite=cache_overwrite,
    )
    if mode in MULTI_MODES:
        assert protocol.train_offsets == tuple(sorted(train_offsets))  # type: ignore[attr-defined]
    protocol.warmup_round_codecs = (
        *("I" for _ in range(start_round)),
        _training_round_name(mode, overrides),
    )
    protocol.routine_round_codecs = ("F",)
    protocol._round_codec_classes = protocol._validate_round_codecs()
    assert protocol._get_curr_round_codec_name(start_round)[2].split("|", 1)[0] == (
        "O" if mode is RunMode.BASELINE else mode.value
    )
    return protocol


def _requested_keys(
    mode: RunMode,
    start_round: int,
    end_round: int,
    client_ids: tuple[int, ...],
) -> set[tuple[int, int]]:
    rounds = (start_round,) if mode is RunMode.BASELINE else range(start_round, end_round + 1)
    return {(round_id, client_id) for round_id in rounds for client_id in client_ids}


def _complete_records(csv_path: Path, expected: set[tuple[int, int]]) -> bool:
    if not csv_path.is_file():
        return False
    records = pd.read_csv(csv_path)
    actual = set(zip(records["round_id"], records["client_id"], strict=True))
    assert len(records) == len(actual), f"Duplicate round-client rows in {csv_path}."
    if actual == expected:
        return True
    raise AssertionError(
        f"Incomplete existing replay {csv_path}: expected {len(expected)} keys, found {len(actual)}. "
        "Pass --overwrite-records to replace it."
    )


def _history_client_ids(
    mode: RunMode,
    selected: tuple[int, ...],
    num_clients: int,
) -> tuple[int, ...]:
    """Return every client for oracle-visible history and selected clients for temporal history."""
    return tuple(range(num_clients)) if mode in ORACLE_MODES else selected


def _protocol_snapshot(
    protocol: BaseProtocol,
    mode: RunMode,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return protocol provenance with the actual indexed quantizer defaults."""
    snapshot = protocol.config_snapshot()
    if mode in INDEX_MODES:
        snapshot["quantizer_defaults"] = IndexAwareWZcfgQuant(
            bins_per_plane=1,
            num_planes=1,
        ).model_dump(
            mode="json",
            exclude={
                "bins_per_plane",
                "num_planes",
                "norm_slices",
                "cache_dir",
                "cache_overwrite",
            },
        )
    if manifest is not None:
        snapshot["experiment_profile"] = manifest["profile"]
        snapshot["resolved_quantizer_config"] = manifest["quantizer_config"]
    return snapshot


def run_replay(args: argparse.Namespace, mode: RunMode) -> Path:
    """Replay one baseline or index-aware trajectory, retaining normal decoded history."""
    assert mode not in ORCHESTRATION_MODES
    assert 0 <= args.start_round <= args.end_round
    set_global_seed(args.seed)
    name = _output_name(mode)
    output_dir = args.records_root / name
    csv_path = output_dir / "compression_records.csv"
    protocol_name = _protocol_name(mode, args.start_round, args.train_offsets)
    cfg = load_replay_config(
        args.fl_config, protocol_name, name, args.records_root, args.seed
    )
    default_clients = (0, 1) if mode is RunMode.BASELINE else tuple(range(cfg.num_clients))
    client_ids = tuple(args.client_ids) if args.client_ids is not None else default_clients
    if mode is RunMode.BASELINE:
        client_ids = (0, 1)
    assert client_ids and len(set(client_ids)) == len(client_ids)
    assert all(0 <= client_id < cfg.num_clients for client_id in client_ids)
    overrides = _resolved_quant_overrides(args, mode)
    manifest = (
        _experiment_manifest(args, mode, client_ids, overrides)
        if args.profile is ExperimentProfile.V2 and mode is not RunMode.BASELINE
        else None
    )
    expected = _requested_keys(mode, args.start_round, args.end_round, client_ids)
    if args.overwrite_records and csv_path.exists():
        csv_path.unlink()
    if manifest is not None and csv_path.exists():
        _validate_manifest(output_dir, manifest)
    if _complete_records(csv_path, expected):
        print(f"{mode.value}: existing complete replay at {csv_path}; reusing it.")
        return csv_path

    cache_dir = args.cache_root / mode.value
    cfg.protocol_cache_dir = cache_dir
    cfg.protocol_cache_overwrite = args.cache_overwrite
    state_manager = build_state_manager(cfg)
    protocol = build_protocol(
        mode,
        args.start_round,
        overrides,
        state_manager.get_slices(),
        cache_dir,
        args.cache_overwrite,
        args.train_offsets,
    )
    history_client_ids = _history_client_ids(mode, client_ids, cfg.num_clients)
    seed_history(
        protocol,
        HistorySource.RAW_DELTAS,
        args.start_round,
        history_client_ids,  # type: ignore[arg-type]
        args.deltas,
        args.deltas,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (output_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
    write_fl_config_snapshot(cfg, output_dir)
    snapshot = _protocol_snapshot(protocol, mode, manifest)
    (output_dir / "protocol_config.json").write_text(json.dumps(snapshot, indent=2))

    rounds = (
        (args.start_round,)
        if mode is RunMode.BASELINE
        else range(args.start_round, args.end_round + 1)
    )
    for round_id in rounds:
        for client_id in client_ids:
            delta = load_replay_tensor(args.deltas, round_id, client_id)
            assert delta.numel() == state_manager.param_count
            print(f"Simulating {mode.value}: round {round_id}, client {client_id}", flush=True)
            simulate_compression(
                protocol, delta, client_id, round_id, state_manager, output_dir, {}, (), ()
            )

    assert _complete_records(csv_path, expected)
    if mode in INDEX_MODES:
        assert not (cache_dir / "priors").exists(), (
            f"Index-aware separate prior unexpectedly created a cache at {cache_dir / 'priors'}."
        )
    return csv_path


def _child_command(args: argparse.Namespace, mode: RunMode) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parents[1] / "replay.py"),
        "--mode",
        mode.value,
        "--profile",
        args.profile.value,
        "--start-round",
        str(args.start_round),
        "--end-round",
        str(args.end_round),
        "--deltas",
        str(args.deltas),
        "--fl-config",
        str(args.fl_config),
        "--records-root",
        str(args.records_root),
        "--cache-root",
        str(args.cache_root),
        "--seed",
        str(args.seed),
        *(token for offset in args.train_offsets for token in ("--train-offset", str(offset))),
        *(token for client_id in (args.client_ids or ()) for token in ("--client-id", str(client_id))),
        *(token for override in args.quant_override for token in ("--quant-override", override)),
    ]
    if args.cache_overwrite:
        command.append("--cache-overwrite")
    if args.overwrite_records:
        command.append("--overwrite-records")
    if args.validation:
        command.append("--validation")
    return command


def _validate_baseline(args: argparse.Namespace) -> None:
    """Reuse or run the two-client O sanity check before either two-GPU experiment."""
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.overwrite_records = False
    baseline_csv = run_replay(baseline_args, RunMode.BASELINE)
    compare_baseline(
        baseline_csv,
        args.original_o,
        args.records_root / _output_name(RunMode.BASELINE) / "rd_sanity",
        round_id=args.start_round,
    )


def _run_gpu_pair(args: argparse.Namespace, modes: tuple[RunMode, RunMode]) -> None:
    """Run two learned modes concurrently, one isolated process per GPU."""
    assert torch.cuda.device_count() >= 2, "Two-GPU orchestration requires at least two GPUs."

    def run_child(mode_gpu: tuple[RunMode, int]) -> None:
        mode, gpu = mode_gpu
        output_dir = args.records_root / _output_name(mode)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "replay.log").open("w") as log:
            subprocess.run(
                _child_command(args, mode),
                cwd=PROJECT_ROOT,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_child, zip(modes, (0, 1), strict=True)))


def run_full(args: argparse.Namespace) -> None:
    """Gate on original O, then run OI and TI concurrently on separate GPUs and compare RD."""
    _validate_baseline(args)
    _run_gpu_pair(args, (RunMode.OI, RunMode.TI))

    compare_experiment(
        args.records_root / _output_name(RunMode.OI) / "compression_records.csv",
        args.original_o,
        args.records_root / _output_name(RunMode.OI) / "rd_comparison.csv",
    )
    compare_experiment(
        args.records_root / _output_name(RunMode.TI) / "compression_records.csv",
        args.original_t,
        args.records_root / _output_name(RunMode.TI) / "rd_comparison.csv",
    )


def run_full_m(args: argparse.Namespace) -> None:
    """Gate on O, run OMI/TMI, and compare each with indexed and original baselines."""
    _validate_baseline(args)
    indexed_baselines = {
        mode: args.records_root / _output_name(mode) / "compression_records.csv"
        for mode in (RunMode.OI, RunMode.TI)
    }
    missing = [path for path in indexed_baselines.values() if not path.is_file()]
    assert not missing, (
        f"full_m reuses completed OI/TI baselines; missing {missing}. Run --mode full first."
    )
    cfg = load_replay_config(
        args.fl_config,
        _protocol_name(RunMode.OI, args.start_round, args.train_offsets),
        "validate_indexed_baselines",
        args.records_root,
        args.seed,
    )
    client_ids = tuple(args.client_ids or range(cfg.num_clients))
    for mode, path in indexed_baselines.items():
        assert _complete_records(
            path,
            _requested_keys(mode, args.start_round, args.end_round, client_ids),
        )
    _run_gpu_pair(args, (RunMode.OMI, RunMode.TMI))

    comparisons = (
        (RunMode.OMI, indexed_baselines[RunMode.OI], "rd_vs_OI.csv"),
        (RunMode.OMI, args.original_o, "rd_vs_O.csv"),
        (RunMode.TMI, indexed_baselines[RunMode.TI], "rd_vs_TI.csv"),
        (RunMode.TMI, args.original_t, "rd_vs_T.csv"),
    )
    for mode, reference, filename in comparisons:
        output_dir = args.records_root / _output_name(mode)
        compare_experiment(
            output_dir / "compression_records.csv",
            reference,
            output_dir / filename,
        )


def main() -> None:
    """Execute the selected replay or the complete sanity-gated experiment."""
    args = _parse_args()
    if args.mode is RunMode.FULL:
        run_full(args)
    elif args.mode is RunMode.FULL_M:
        run_full_m(args)
    else:
        run_replay(args, args.mode)


if __name__ == "__main__":
    main()
