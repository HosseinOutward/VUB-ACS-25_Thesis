"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from . import run_replay as replay_module
from .run_replay import (
    ExperimentProfile,
    RunMode,
    V1_QUANT_OVERRIDES,
    V2_MODE_QUANT_OVERRIDES,
    V2_QUANT_OVERRIDES,
    V2_VALIDATION_OVERRIDES,
    _child_command,
    _experiment_manifest,
    _parse_args,
    _protocol_snapshot,
    _resolved_quant_overrides,
    _validate_manifest,
    build_protocol,
)


def _args(tmp_path: Path, **updates: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "profile": ExperimentProfile.V2,
        "validation": False,
        "client_ids": None,
        "start_round": 12,
        "end_round": 20,
        "deltas": tmp_path / "deltas",
        "fl_config": tmp_path / "fl_config.json",
        "records_root": tmp_path / "records_v2",
        "cache_root": tmp_path / "cache_v2",
        "original_o": tmp_path / "original_o.csv",
        "original_t": tmp_path / "original_t.csv",
        "train_offsets": (1, 2, 3),
        "seed": 43,
        "cache_overwrite": False,
        "overwrite_records": False,
        "quant_override": [],
    }
    return argparse.Namespace(**(values | updates))


def test_v2_defaults_use_isolated_roots_and_validation_narrows_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_replay", "--profile", "v2", "--mode", "OMI", "--validation"],
    )
    args = _parse_args()
    assert args.profile is ExperimentProfile.V2
    assert args.client_ids == [0]
    assert (args.start_round, args.end_round) == (12, 12)
    assert args.records_root.name == "validation"
    assert args.records_root.parent.name == "records"
    assert args.cache_root.name == "validation"
    assert args.cache_root.parent.name == "flagship_comparison"


def test_v1_profile_explicitly_pins_the_legacy_cache_semantics(tmp_path: Path) -> None:
    resolved = _resolved_quant_overrides(
        _args(tmp_path, profile=ExperimentProfile.V1), RunMode.OMI
    )

    assert {field: resolved[field] for field in V1_QUANT_OVERRIDES} == (
        V1_QUANT_OVERRIDES
    )
    assert resolved["training_seed"] == 43


def test_v2_profile_pins_capacity_duration_and_fast_validation(tmp_path: Path) -> None:
    full = _resolved_quant_overrides(_args(tmp_path), RunMode.OMI)
    assert {field: full[field] for field in V2_QUANT_OVERRIDES} == V2_QUANT_OVERRIDES
    assert full["train_epochs"] == 180

    validation = _resolved_quant_overrides(
        _args(tmp_path, validation=True), RunMode.TMI
    )
    assert {
        field: validation[field] for field in V2_MODE_QUANT_OVERRIDES[RunMode.TMI]
    } == V2_MODE_QUANT_OVERRIDES[RunMode.TMI]
    assert validation["hidden_dim"] == 112
    assert validation["layers"] == 8
    assert validation["embedding_dim"] == 24
    assert {field: validation[field] for field in V2_VALIDATION_OVERRIDES} == (
        V2_VALIDATION_OVERRIDES
    )


def test_v2_profile_rejects_overrides_of_pinned_fields(tmp_path: Path) -> None:
    args = _args(tmp_path, quant_override=["hidden_dim=64"])
    with pytest.raises(AssertionError, match="pinned"):
        _resolved_quant_overrides(args, RunMode.OMI)


def test_manifest_requires_exact_resolved_experiment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay_module, "_source_revision", lambda: "revision")
    args = _args(tmp_path)
    overrides = _resolved_quant_overrides(args, RunMode.OMI)
    manifest = _experiment_manifest(args, RunMode.OMI, (0,), overrides)
    assert manifest["profile"] == "v2"
    assert manifest["quantizer_config"]["index_cache_version"] == (
        "index-side-information/v2"
    )
    assert manifest["cache_dir"].endswith("cache_v2/OMI")

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    _validate_manifest(output_dir, manifest)
    with pytest.raises(AssertionError, match="does not match"):
        _validate_manifest(output_dir, manifest | {"seed": 44})


def test_v2_protocol_snapshot_contains_resolved_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay_module, "_source_revision", lambda: "revision")
    args = _args(tmp_path)
    overrides = _resolved_quant_overrides(args, RunMode.OMI)
    manifest = _experiment_manifest(args, RunMode.OMI, (0,), overrides)
    protocol = build_protocol(
        RunMode.OMI,
        12,
        overrides,
        [slice(0, 4)],
        args.cache_root / "OMI",
        train_offsets=args.train_offsets,
    )
    snapshot = _protocol_snapshot(protocol, RunMode.OMI, manifest)
    assert snapshot["experiment_profile"] == "v2"
    assert snapshot["resolved_quantizer_config"]["hidden_dim"] == 112
    assert snapshot["resolved_quantizer_config"]["train_epochs"] == 180


def test_complete_v2_records_reuse_only_with_matching_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay_module, "_source_revision", lambda: "revision")
    monkeypatch.setattr(
        replay_module,
        "load_replay_config",
        lambda *_args, **_kwargs: argparse.Namespace(num_clients=5),
    )
    args = _args(tmp_path, client_ids=[0], end_round=12)
    overrides = _resolved_quant_overrides(args, RunMode.OMI)
    manifest = _experiment_manifest(args, RunMode.OMI, (0,), overrides)
    output_dir = args.records_root / "OMI"
    output_dir.mkdir(parents=True)
    pd.DataFrame([{"round_id": 12, "client_id": 0}]).to_csv(
        output_dir / "compression_records.csv", index=False
    )
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    assert replay_module.run_replay(args, RunMode.OMI) == (
        output_dir / "compression_records.csv"
    )
    with pytest.raises(AssertionError, match="does not match"):
        replay_module.run_replay(
            argparse.Namespace(**(vars(args) | {"seed": 44})), RunMode.OMI
        )


def test_v2_child_command_preserves_profile_and_isolated_paths(tmp_path: Path) -> None:
    args = _args(tmp_path)
    command = _child_command(args, RunMode.TMI)
    assert command[command.index("--profile") + 1] == "v2"
    assert command[command.index("--records-root") + 1].endswith("records_v2")
    assert command[command.index("--cache-root") + 1].endswith("cache_v2")
