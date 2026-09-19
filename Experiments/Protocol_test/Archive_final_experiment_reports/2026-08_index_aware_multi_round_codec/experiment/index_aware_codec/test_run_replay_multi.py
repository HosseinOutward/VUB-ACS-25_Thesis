"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pytest

from . import run_replay as replay_module
from .round_codecs import OM_RoundCodec, OMI_RoundCodec, TM_RoundCodec, TMI_RoundCodec
from .run_replay import (
    ExperimentProfile,
    RunMode,
    _child_command,
    _history_client_ids,
    _parse_args,
    _protocol_name,
    _protocol_snapshot,
    build_protocol,
    run_full_m,
)


def _args(tmp_path: Path, **updates: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "client_ids": None,
        "profile": ExperimentProfile.V1,
        "validation": False,
        "start_round": 12,
        "end_round": 13,
        "deltas": tmp_path / "deltas",
        "fl_config": tmp_path / "fl_config.json",
        "records_root": tmp_path / "records",
        "cache_root": tmp_path / "cache",
        "original_o": tmp_path / "original_o.csv",
        "original_t": tmp_path / "original_t.csv",
        "train_offsets": (1, 2, 3),
        "seed": 43,
        "cache_overwrite": False,
        "overwrite_records": False,
        "quant_override": [],
    }
    return argparse.Namespace(**(values | updates))


def test_parse_train_offsets_defaults_and_repeated_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_replay"])
    assert _parse_args().train_offsets == (1, 2, 3)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_replay", "--train-offset", "2", "--train-offset", "5"],
    )
    assert _parse_args().train_offsets == (2, 5)

    for invalid in ((0,), (2, 2)):
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_replay", *(token for offset in invalid for token in ("--train-offset", str(offset)))],
        )
        with pytest.raises(AssertionError):
            _parse_args()


def test_only_multi_protocol_names_include_train_offsets() -> None:
    offsets = (2, 5)
    assert _protocol_name(RunMode.BASELINE, 12, offsets) == "wz_cancer|train_at_round=12"
    assert _protocol_name(RunMode.O, 12, offsets) == "wz_cancer|train_at_round=12"
    assert _protocol_name(RunMode.T, 12, offsets) == (
        "wz_cancer|replace_O_with='T'|train_at_round=12"
    )
    assert _protocol_name(RunMode.OI, 12, offsets) == (
        "wz_cancer|replace_O_with='OI'|train_at_round=12"
    )
    assert _protocol_name(RunMode.TI, 12, offsets) == (
        "wz_cancer|replace_O_with='TI'|train_at_round=12"
    )
    assert _protocol_name(RunMode.OMI, 12, offsets).endswith("train_offsets=(2, 5)")
    assert _protocol_name(RunMode.TMI, 12, offsets).endswith("train_offsets=(2, 5)")
    assert _protocol_name(RunMode.OM, 12, offsets).endswith("train_offsets=(2, 5)")
    assert _protocol_name(RunMode.TM, 12, offsets).endswith("train_offsets=(2, 5)")


@pytest.mark.parametrize(
    ("mode", "codec_class"),
    [
        (RunMode.OM, OM_RoundCodec),
        (RunMode.TM, TM_RoundCodec),
        (RunMode.OMI, OMI_RoundCodec),
        (RunMode.TMI, TMI_RoundCodec),
    ],
)
def test_multi_protocol_allocates_lagged_history_before_schedule_use(
    tmp_path: Path,
    mode: RunMode,
    codec_class: type[OM_RoundCodec | TM_RoundCodec | OMI_RoundCodec | TMI_RoundCodec],
) -> None:
    protocol = build_protocol(
        mode,
        12,
        {},
        [slice(0, 4)],
        tmp_path,
        train_offsets=(2, 4),
    )
    assert protocol.train_offsets == (2, 4)  # type: ignore[attr-defined]
    assert len(protocol.histories()) == 2
    assert protocol.histories()[1].max_per_client == 9
    assert protocol._get_curr_round_codec_name(12)[0] is codec_class
    assert protocol._get_curr_round_codec_name(13)[2] == "F"


def test_protocol_snapshot_records_index_architecture(tmp_path: Path) -> None:
    indexed = build_protocol(RunMode.OMI, 12, {}, [slice(0, 4)], tmp_path)
    defaults = _protocol_snapshot(indexed, RunMode.OMI)["quantizer_defaults"]
    assert defaults["index_cache_version"] == "index-side-information/v2"
    assert defaults["hidden_dim"] == 112
    assert defaults["layers"] == 8
    assert defaults["hash_table_count"] == 8
    assert defaults["route_count"] == 64
    assert defaults["embedding_dim"] == 24

    baseline = build_protocol(RunMode.BASELINE, 12, {}, [slice(0, 4)], tmp_path)
    baseline_defaults = _protocol_snapshot(baseline, RunMode.BASELINE)["quantizer_defaults"]
    assert baseline_defaults["hidden_dim"] == 74
    assert "hash_table_count" not in baseline_defaults


def test_oracle_modes_seed_all_clients_and_temporal_modes_do_not() -> None:
    selected = (1, 3)
    assert _history_client_ids(RunMode.BASELINE, selected, 5) == tuple(range(5))
    assert _history_client_ids(RunMode.O, selected, 5) == tuple(range(5))
    assert _history_client_ids(RunMode.OI, selected, 5) == tuple(range(5))
    assert _history_client_ids(RunMode.OMI, selected, 5) == tuple(range(5))
    assert _history_client_ids(RunMode.OM, selected, 5) == tuple(range(5))
    assert _history_client_ids(RunMode.TI, selected, 5) == selected
    assert _history_client_ids(RunMode.T, selected, 5) == selected
    assert _history_client_ids(RunMode.TMI, selected, 5) == selected
    assert _history_client_ids(RunMode.TM, selected, 5) == selected


def test_child_command_preserves_multi_mode_offsets_and_separate_roots(tmp_path: Path) -> None:
    args = _args(tmp_path, train_offsets=(2, 4))
    command = _child_command(args, RunMode.OMI)
    assert command[command.index("--mode") + 1] == "OMI"
    assert command.count("--train-offset") == 2
    assert command[command.index("--records-root") + 1] == str(args.records_root)
    assert command[command.index("--cache-root") + 1] == str(args.cache_root)


def test_full_m_uses_two_multi_modes_and_writes_four_comparisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    for mode in (RunMode.OI, RunMode.TI):
        path = args.records_root / mode.value / "compression_records.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame([
            {"round_id": round_id, "client_id": client_id}
            for round_id in (12, 13)
            for client_id in (0, 1)
        ]).to_csv(path, index=False)

    calls: list[tuple[Path, Path, Path]] = []
    gpu_pairs: list[tuple[RunMode, RunMode]] = []
    monkeypatch.setattr(replay_module, "_validate_baseline", lambda _args: None)
    monkeypatch.setattr(
        replay_module,
        "load_replay_config",
        lambda *_args, **_kwargs: argparse.Namespace(num_clients=2),
    )
    monkeypatch.setattr(
        replay_module,
        "_run_gpu_pair",
        lambda _args, modes: gpu_pairs.append(modes),
    )
    monkeypatch.setattr(
        replay_module,
        "compare_experiment",
        lambda candidate, reference, output: calls.append((candidate, reference, output)),
    )

    run_full_m(args)

    assert gpu_pairs == [(RunMode.OMI, RunMode.TMI)]
    assert [output.name for _, _, output in calls] == [
        "rd_vs_OI.csv",
        "rd_vs_O.csv",
        "rd_vs_TI.csv",
        "rd_vs_T.csv",
    ]
    assert calls[0][0].parent.name == "OMI"
    assert calls[2][0].parent.name == "TMI"
