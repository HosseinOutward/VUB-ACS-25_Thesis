from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pydantic import ValidationError
import pytest
import torch

from Experiments.Protocol_test import sparse_replay
from Experiments.Protocol_test.replay import HistorySource


class _StateManager:
    param_count = 1

    def get_slices(self) -> list[slice]:
        return [slice(None)]


class _Config(SimpleNamespace):
    def model_dump(self, mode: str) -> dict[str, object]:
        assert mode == "json"
        return vars(self)


class _Protocol:
    def __init__(self, name: str, schedule: tuple[str, ...], events: list[tuple[str, int, int]]) -> None:
        self.protocol_name_full = name
        self.schedule = schedule
        self.events = events
        self.histories = (SimpleNamespace(dtype=torch.float32, commit=self._commit),)

    def _commit(self, _delta: torch.Tensor, record: Any, _access: object) -> None:
        self.events.append(("context", record.round_id, record.client_id))

    def config_snapshot(self) -> dict[str, object]:
        return {"warmup_round_codecs": (), "routine_round_codecs": self.schedule}


def _case() -> sparse_replay.ReplayCase:
    return sparse_replay.ReplayCase(
        case_id="rounds_2_3", codec_id="codec", protocol="test", start_round=2,
        client_ids=(1,), round_codecs=("A", "B"),
    )


def _plan(tmp_path: Path, cases: tuple[sparse_replay.ReplayCase, ...] | None = None) -> sparse_replay.ReplayPlan:
    return sparse_replay.ReplayPlan(
        name="test", dataset_name="DATA", model_name="MODEL", fl_config=tmp_path / "fl.json",
        deltas=tmp_path / "deltas", context=tmp_path / "context", output_root=tmp_path / "output",
        cache_root=tmp_path / "cache", cases=cases or (_case(),), context_source=HistorySource.RAW_DELTAS,
        replace_selected_with_raw=True,
    )


def test_plan_rejects_duplicate_cases_and_unknown_comparisons(tmp_path: Path) -> None:
    case = _case()
    with pytest.raises(ValidationError, match="unique"):
        _plan(tmp_path, (case, case))
    with pytest.raises(ValidationError, match="Invalid comparison"):
        sparse_replay.ReplayPlan.model_validate(
            _plan(tmp_path).model_dump() | {"comparisons": (("codec", "missing"),)}
        )


def test_case_preserves_all_client_order_and_resumes_only_complete_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, plan = _case(), _plan(tmp_path)
    events: list[tuple[str, int, int]] = []
    protocol = _Protocol(case.protocol, case.round_codecs, events)
    monkeypatch.setattr(sparse_replay, "set_global_seed", lambda _seed: None)
    monkeypatch.setattr(
        sparse_replay, "load_replay_config",
        lambda *_args: _Config(dataset_name=plan.dataset_name, model_name=plan.model_name, num_clients=3),
    )
    monkeypatch.setattr(sparse_replay, "build_state_manager", lambda _cfg: _StateManager())
    monkeypatch.setattr(sparse_replay, "seed_history", lambda *_args: None)
    monkeypatch.setattr(
        sparse_replay, "load_replay_tensor",
        lambda _path, round_id, client_id: torch.tensor([10 * round_id + client_id], dtype=torch.float32),
    )
    monkeypatch.setattr(
        sparse_replay, "replace_latest_reconstruction",
        lambda _protocol, _delta, round_id, client_id: events.append(("replace", round_id, client_id)),
    )

    def simulate(
        _protocol: object, _delta: torch.Tensor, client_id: int, round_id: int,
        _state_manager: object, save_dir: Path, *_metrics: object,
    ) -> None:
        events.append(("simulate", round_id, client_id))
        path = save_dir / "compression_records.csv"
        with path.open("a", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=("round_id", "client_id", "round_name_full"))
            if output.tell() == 0:
                writer.writeheader()
            writer.writerow({"round_id": round_id, "client_id": client_id,
                             "round_name_full": case.round_codecs[round_id - case.start_round]})

    monkeypatch.setattr(sparse_replay, "simulate_compression", simulate)
    factory = cast(sparse_replay.ProtocolFactory, lambda *_args: protocol)
    records = sparse_replay.run_case(plan, case, factory)

    assert records.is_file()
    assert events == [
        ("context", 2, 0), ("simulate", 2, 1), ("replace", 2, 1), ("context", 2, 2),
        ("context", 3, 0), ("simulate", 3, 1), ("replace", 3, 1), ("context", 3, 2),
    ]
    sparse_replay.run_case(plan, case, factory)
    assert len(events) == 8
    (records.parent / "complete.json").unlink()
    with pytest.raises(AssertionError, match="Incomplete case directory"):
        sparse_replay.run_case(plan, case, factory)
