from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from Experiments.Protocol_test import records, sparse_replay
from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.cancer_protocol.history import ReconstructionHistory
from temp_nsplit import nsplit_quantize

SEED = 43
TRAIN_TOKEN = f"T|training_seed={SEED}"


class _FakeTrace:
    """Mimics `Trace`'s disk-persisted identity without any real FL training.

    `ensure`/`identity` state is keyed by directory (a plain dict, standing in for the marker
    file `Trace` persists to disk), so a second `_FakeTrace` instance opened on the same
    directory - exactly what `run_case` does - observes a prior `ensure()` the same way the real
    class would.
    """

    _ensured_dirs: set[str] = set()

    def __init__(self, directory: Path, cfg: Any, *, dataset_revision: str) -> None:
        self.directory, self.cfg, self.dataset_revision = directory, cfg, dataset_revision

    def ensure(
        self, rounds: Any, device: torch.device, *, checkpoint_rounds: Any = (),
    ) -> None:
        assert isinstance(device, torch.device)
        assert all(round_id > 0 for round_id in checkpoint_rounds)
        type(self)._ensured_dirs.add(str(self.directory))

    def identity(self) -> dict[str, Any]:
        assert str(self.directory) in type(self)._ensured_dirs, "identity() requested before ensure()"
        return {"trace_version": 1, "fl_config": {"num_clients": self.cfg.num_clients},
                "dataset_revision": self.dataset_revision, "device": {"type": "cpu"}}

    def load_delta(self, round_id: int, client_id: int) -> torch.Tensor:
        return torch.tensor([10.0 * round_id + client_id], dtype=torch.float32)


class _FakeProtocol:
    """Orchestration-only stand-in for `_WZProtocol`: real history, fake schedule/snapshot."""

    def __init__(self, depth: int = 2, refresh_frozen_si: bool = True, snapshot_tag: str = "v1") -> None:
        self.max_per_client_recons_history = depth
        self.refresh_frozen_si = refresh_frozen_si
        self._snapshot_tag = snapshot_tag
        self.histories = (ReconstructionHistory(depth),)

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "implementation": "test",
            "tag": self._snapshot_tag,
            "history": self.histories[0].config_snapshot(),
        }

    def install_histories(self, histories: tuple[ReconstructionHistory, ...]) -> None:
        assert len(histories) == 1
        self.histories = histories


def _write_plan(tmp_path: Path, cases: tuple[sparse_replay.ReplayCase, ...], **overrides: Any) -> Path:
    (tmp_path / "fl.json").write_text(json.dumps({"protocol": "identity", "num_clients": 3, "num_classes": 2}))
    payload: dict[str, Any] = {
        "name": "test", "fl_config": "fl.json", "trace": "trace", "dataset_revision": "rev-1",
        "trace_device": "cpu", "output_root": "output", "history_cache_root": "history_cache",
        "gpu_ids": [], "seed": SEED,
        "cases": [json.loads(case.model_dump_json()) for case in cases],
    }
    payload.update(overrides)
    plan_path = tmp_path / f"plan_{len(list(tmp_path.glob('plan_*.json')))}.json"
    plan_path.write_text(json.dumps(payload))
    return plan_path


def _case(case_id: str = "start5", codec_id: str = "T", start_round: int = 5,
          client_ids: tuple[int, ...] = (1,), round_codecs: tuple[str, ...] = (TRAIN_TOKEN, "F")) -> sparse_replay.ReplayCase:
    return sparse_replay.ReplayCase(case_id=case_id, codec_id=codec_id, start_round=start_round,
                                     client_ids=client_ids, round_codecs=round_codecs)


def _install_fakes(monkeypatch: pytest.MonkeyPatch, protocols: dict[str, _FakeProtocol],
                    simulate: Any) -> None:
    monkeypatch.setattr(sparse_replay, "Trace", _FakeTrace)
    monkeypatch.setattr(sparse_replay, "initialize_model", lambda *_args: nn.Linear(1, 1, bias=False))
    monkeypatch.setattr(sparse_replay, "case_protocol", lambda _plan, case, _manager=None: protocols[case.case_id])
    monkeypatch.setattr(sparse_replay, "simulate_compression", simulate)


def _recording_simulate(seen: list[tuple[int, int, list[float]]]) -> Any:
    """A `simulate_compression` fake that snapshots visible history and writes one CSV row.

    A case's first measured call always lands on its training round (rounds run in increasing
    order within a case), so the training token is recognized dynamically per output directory
    rather than assuming a fixed `start_round` shared by every case in a test.
    """
    first_round: dict[Path, int] = {}

    def _simulate(protocol: _FakeProtocol, delta: torch.Tensor, client_id: int, round_id: int,
                  _manager: object, output: Path, *_metrics: object) -> torch.Tensor:
        history = protocol.histories[0]
        view = history.view(Access.SERVER_ONLY, client_id)
        seen.append((
            round_id,
            client_id,
            [view[c][-1].value.materialize().item() for c in sorted(view)],
        ))
        reconst = delta * 0 - 1  # deliberately distinct from the raw delta, so replace_latest is observable
        history.commit(reconst, CompressionRecord(round_id, client_id, "T"), Access.TEMPORAL_TOO)
        start_round = first_round.setdefault(output, round_id)
        token = TRAIN_TOKEN if round_id == start_round else "F"
        with (output / "compression_records.csv").open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("round_id", "client_id", "round_name_full"))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow({"round_id": round_id, "client_id": client_id, "round_name_full": token})
        return reconst

    return _simulate


# --- Malformed schedules: rejected at model construction, before any expensive work --- #

def test_case_rejects_a_schedule_with_no_frozen_round() -> None:
    with pytest.raises(ValidationError, match="at least two"):
        _case(round_codecs=(TRAIN_TOKEN,))
    with pytest.raises(ValidationError, match="frozen round"):
        _case(round_codecs=(TRAIN_TOKEN, TRAIN_TOKEN))


def test_case_accepts_a_second_training_cycle() -> None:
    case = _case(round_codecs=(TRAIN_TOKEN, "F", TRAIN_TOKEN, "F"))
    assert case.round_codecs[2] == TRAIN_TOKEN


def test_nsplit_quantize_uses_quantile_bins_and_bin_means() -> None:
    values = torch.tensor([-3.0, -2.0, 1.0, 4.0])
    assert torch.equal(nsplit_quantize(values, 2), torch.tensor([-2.5, -2.5, 2.5, 2.5]))


def test_plan_rejects_a_training_seed_that_does_not_match_the_plan_seed(tmp_path: Path) -> None:
    case = _case(round_codecs=("T|training_seed=99", "F"))
    with pytest.raises(ValidationError, match="training_seed=43"):
        sparse_replay.load_plan(_write_plan(tmp_path, (case,)))


def test_plan_rejects_duplicate_cases_and_unknown_comparisons(tmp_path: Path) -> None:
    case = _case()
    with pytest.raises(ValidationError, match="unique"):
        sparse_replay.load_plan(_write_plan(tmp_path, (case, case)))
    with pytest.raises(ValidationError, match="Invalid comparison"):
        sparse_replay.load_plan(_write_plan(tmp_path, (case,), comparisons=[["T", "missing"]]))


def test_gpu_cases_use_one_serial_disposable_worker_queue_per_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingPool:
        instances: list[RecordingPool] = []

        def __init__(self, **options: Any) -> None:
            self.options = options
            self.submissions: list[tuple[int, str]] = []
            type(self).instances.append(self)

        def __enter__(self) -> RecordingPool:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self, _function: Any, _plan: object, gpu_id: int, case: object, _provenance: object,
        ) -> object:
            case_id = case.case_id
            self.submissions.append((gpu_id, case_id))
            return SimpleNamespace(result=lambda: Path(case_id))

    monkeypatch.setattr(sparse_replay, "ProcessPoolExecutor", RecordingPool)
    plan = SimpleNamespace(
        gpu_ids=(2, 5),
        cases=tuple(SimpleNamespace(case_id=f"case{index}") for index in range(5)),
    )

    results = sparse_replay._run_gpu_cases(plan, {})

    assert [pool.options["max_workers"] for pool in RecordingPool.instances] == [1, 1]
    assert [pool.options["max_tasks_per_child"] for pool in RecordingPool.instances] == [1, 1]
    assert [pool.submissions for pool in RecordingPool.instances] == [
        [(2, "case0"), (2, "case2"), (2, "case4")],
        [(5, "case1"), (5, "case3")],
    ]
    assert results == tuple(Path(f"case{index}") for index in range(5))


def test_run_sparse_plan_rejects_client_id_out_of_range_before_trace_ensure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(client_ids=(5,))
    plan_path = _write_plan(tmp_path, (case,))
    _install_fakes(monkeypatch, {"start5": _FakeProtocol()}, _recording_simulate([]))
    with pytest.raises(AssertionError):
        sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)
    assert str((tmp_path / "trace").resolve()) not in _FakeTrace._ensured_dirs


def test_run_sparse_plan_rejects_start_round_below_history_depth_before_trace_ensure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(start_round=1)
    plan_path = _write_plan(tmp_path, (case,))
    _install_fakes(monkeypatch, {"start5": _FakeProtocol(depth=2)}, _recording_simulate([]))
    with pytest.raises(AssertionError, match="preceding raw rounds"):
        sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)
    assert str((tmp_path / "trace").resolve()) not in _FakeTrace._ensured_dirs


# --- Raw visibility and ordering --- #

def test_raw_visibility_matches_the_normal_client_order_for_measured_worker_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(case_id="start5", client_ids=(1,), start_round=5)
    plan_path = _write_plan(tmp_path, (case,))
    seen: list[tuple[int, int, list[float]]] = []
    protocol = _FakeProtocol(depth=2)
    _install_fakes(monkeypatch, {"start5": protocol}, _recording_simulate(seen))

    sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)

    # Just before client 1's compression at round 5: client 0 already advanced to round 5
    # (processed earlier in the same round), clients 1 and 2 are still at round 4.
    round_id, client_id, values = seen[0]
    assert (round_id, client_id) == (5, 1)
    assert values == [50.0, 41.0, 42.0]  # 10*round + client, keyed by sorted client id

    # Client 1's temporal (self-only) view contains only its own preceding entries, never
    # another client's, regardless of what the server-side ledger holds.
    temporal_view = protocol.histories[0].view(Access.TEMPORAL_TOO, 1)
    assert set(temporal_view) == {1}


def test_nsplit_initial_si_is_applied_only_to_the_seed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(client_ids=(1,)).model_copy(update={"initial_si": "nsplit"})
    plan_path = _write_plan(tmp_path, (case,), nsplit_bins=2)
    seen: list[tuple[int, int, list[float]]] = []
    _install_fakes(monkeypatch, {"start5": _FakeProtocol(depth=2)}, _recording_simulate(seen))
    monkeypatch.setattr(sparse_replay, "nsplit_quantize", lambda values, _bins: values + 1000)

    sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)

    assert seen[0] == (5, 1, [50.0, 1041.0, 1042.0])


def test_raw_history_gives_way_to_decoded_updates_after_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(case_id="start5", client_ids=(1,), start_round=5)
    plan_path = _write_plan(tmp_path, (case,))
    protocol = _FakeProtocol(depth=2)
    _install_fakes(monkeypatch, {"start5": protocol}, _recording_simulate([]))

    sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)

    # Initial history is raw, but simulate_compression's reconstruction remains in the ledger.
    # Depth 2 still retains both rounds 5 and 6, so round 5 is index 0, not the latest.
    server_view = protocol.histories[0].view(Access.SERVER_ONLY, 0)
    round5_entry = server_view[1][0]
    assert round5_entry.round_id == 5 and round5_entry.value.materialize().item() == -1.0
    assert round5_entry.access is Access.TEMPORAL_TOO


def test_worker_subset_zero_and_two_measured_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(case_id="start5", client_ids=(0, 2), start_round=5)
    plan_path = _write_plan(tmp_path, (case,))
    seen: list[tuple[int, int, list[float]]] = []
    _install_fakes(monkeypatch, {"start5": _FakeProtocol(depth=2)}, _recording_simulate(seen))

    sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)

    measured_rounds_clients = [(r, c) for r, c, _ in seen]
    assert measured_rounds_clients == [(5, 0), (5, 2), (6, 0), (6, 2)]
    # Unselected client 1 never triggers a codec: only 0 and 2 appear in `seen`.
    assert {client for _, client, _ in seen} == {0, 2}


def test_two_independent_windows_do_not_share_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case5 = _case(case_id="start5", client_ids=(1,), start_round=5)
    case8 = _case(case_id="start8", client_ids=(1,), start_round=8)
    plan_path = _write_plan(tmp_path, (case5, case8))
    seen: list[tuple[int, int, list[float]]] = []
    _install_fakes(monkeypatch, {"start5": _FakeProtocol(depth=2), "start8": _FakeProtocol(depth=2)},
                    _recording_simulate(seen))

    sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)

    by_case = {(r, c): values for r, c, values in seen}
    # Case start8's first measured round sees only its own window's raw rounds (6, 7), with no
    # trace of case start5's rounds (3, 4).
    assert by_case[(8, 1)] == [80.0, 71.0, 72.0]
    assert by_case[(5, 1)] == [50.0, 41.0, 42.0]


# --- Completion and reuse --- #

def test_completed_case_is_reused_without_recompressing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    plan_path = _write_plan(tmp_path, (case,))
    calls = []
    simulate = _recording_simulate(calls)
    _install_fakes(monkeypatch, {"start5": _FakeProtocol(depth=2)}, simulate)

    plan = sparse_replay.load_plan(plan_path)
    first = sparse_replay.run_sparse_plan(plan, plan_path)
    assert len(calls) == 2  # one measured client x two scheduled rounds (train + frozen)

    second = sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)
    assert second == first
    assert len(calls) == 2, "a completed case must not re-run compression"


def test_history_cache_location_change_still_reuses_a_completed_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    plan_path = _write_plan(tmp_path, (case,))
    calls: list[Any] = []
    _install_fakes(monkeypatch, {"start5": _FakeProtocol(depth=2)}, _recording_simulate(calls))
    sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)
    assert len(calls) == 2

    changed_path = _write_plan(tmp_path, (case,), gpu_ids=[], history_cache_root="history_cache2")
    # output_root stays "output" so the same case directory is reused.
    sparse_replay.run_sparse_plan(sparse_replay.load_plan(changed_path), changed_path)
    assert len(calls) == 2, "gpu_ids/history_cache_root must not force recompression"


def test_changed_trace_identity_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    plan_path = _write_plan(tmp_path, (case,))
    _install_fakes(monkeypatch, {"start5": _FakeProtocol(depth=2)}, _recording_simulate([]))
    sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)

    changed_path = _write_plan(tmp_path, (case,), dataset_revision="rev-2")
    with pytest.raises(AssertionError, match="differ"):
        sparse_replay.run_sparse_plan(sparse_replay.load_plan(changed_path), changed_path)


def test_changed_semantic_codec_setting_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    plan_path = _write_plan(tmp_path, (case,))
    _install_fakes(monkeypatch, {"start5": _FakeProtocol(depth=2, snapshot_tag="v1")}, _recording_simulate([]))
    sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)

    monkeypatch.setattr(sparse_replay, "case_protocol",
                         lambda _plan, _case, _manager=None: _FakeProtocol(depth=2, snapshot_tag="v2"))
    with pytest.raises(AssertionError, match="differ"):
        sparse_replay.run_sparse_plan(sparse_replay.load_plan(plan_path), plan_path)


def test_identity_never_embeds_execution_only_fields(tmp_path: Path) -> None:
    case = _case()
    plan_path = _write_plan(tmp_path, (case,))
    plan = sparse_replay.load_plan(plan_path)
    protocol = _FakeProtocol()
    identity = sparse_replay._case_identity(plan, case, {"dataset_revision": "rev-1"}, protocol)
    plan_identity = sparse_replay._plan_identity(plan, {"dataset_revision": "rev-1"})
    forbidden = {"output_root", "history_cache_root", "gpu_ids", "plan_path"}
    assert forbidden.isdisjoint(identity) and forbidden.isdisjoint(plan_identity)


def test_initial_history_identity_separates_nsplit_but_preserves_raw_cache(
    tmp_path: Path,
) -> None:
    raw = _case()
    classical = raw.model_copy(update={"initial_si": "nsplit"})
    first = sparse_replay.load_plan(_write_plan(tmp_path, (raw,), nsplit_bins=2))
    second = sparse_replay.load_plan(_write_plan(tmp_path, (raw,), nsplit_bins=3))
    protocol = _FakeProtocol()
    trace_identity = {"dataset_revision": "rev-1"}
    assert (
        sparse_replay._initial_history_cache_dir(first, raw, trace_identity, protocol)
        == sparse_replay._initial_history_cache_dir(second, raw, trace_identity, protocol)
    )
    assert (
        sparse_replay._initial_history_cache_dir(first, classical, trace_identity, protocol)
        != sparse_replay._initial_history_cache_dir(second, classical, trace_identity, protocol)
    )


# --- records.py: the shared completion contract --- #

def test_validate_case_records_rejects_missing_rows(tmp_path: Path) -> None:
    path = tmp_path / "compression_records.csv"
    path.write_text("round_id,client_id,round_name_full\n5,1,T\n")  # missing round 6 row
    with pytest.raises(AssertionError, match="Incomplete"):
        records.validate_case_records(path, {"start_round": 5, "round_codecs": (TRAIN_TOKEN, "F"), "client_ids": (1,)})


def test_validate_case_records_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "compression_records.csv"
    path.write_text(f"round_id,client_id,round_name_full\n5,1,{TRAIN_TOKEN}\n5,1,{TRAIN_TOKEN}\n6,1,F\n")
    with pytest.raises(AssertionError, match="Incomplete"):
        records.validate_case_records(path, {"start_round": 5, "round_codecs": (TRAIN_TOKEN, "F"), "client_ids": (1,)})


def test_read_completed_case_without_marker_fails(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="unfinished"):
        records.read_completed_case(tmp_path)


def test_read_completed_case_enriches_rows_with_placement_and_phase(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / "T" / "start5"
    case_dir.mkdir(parents=True)
    (case_dir / "compression_records.csv").write_text(
        f"round_id,client_id,round_name_full\n5,1,{TRAIN_TOKEN}\n6,1,F\n")
    identity = {"case": {"protocol": "wz_cancer", "start_round": 5, "client_ids": [1],
                          "initial_si": "raw",
                          "round_codecs": [TRAIN_TOKEN, "F"]},
                "trace": {"dataset_revision": "rev-1"}}
    (case_dir / "complete.json").write_text(json.dumps({"identity": identity, "provenance": {}}))

    frame = records.read_completed_case(case_dir)
    assert frame["codec_id"].tolist() == ["T", "T"]
    assert frame["case_id"].tolist() == ["start5", "start5"]
    assert frame["phase"].tolist() == ["trained", "frozen"]
    assert frame["cycle"].tolist() == [0, 0]
    assert frame["relative_round"].tolist() == [0, 1]
    assert frame["client_ids"].tolist() == [(1,), (1,)]


def test_read_completed_case_marks_every_non_f_token_as_trained(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / "T" / "start5"
    case_dir.mkdir(parents=True)
    tokens = [TRAIN_TOKEN, "F", TRAIN_TOKEN, "F"]
    rows = "\n".join(f"{round_id},1,{token}" for round_id, token in enumerate(tokens, 5))
    (case_dir / "compression_records.csv").write_text(
        f"round_id,client_id,round_name_full\n{rows}\n")
    identity = {"case": {"protocol": "wz_cancer", "start_round": 5, "client_ids": [1],
                         "initial_si": "raw",
                         "round_codecs": tokens},
                "trace": {"dataset_revision": "rev-1"}}
    (case_dir / "complete.json").write_text(json.dumps({"identity": identity, "provenance": {}}))

    assert records.read_completed_case(case_dir)["phase"].tolist() == [
        "trained", "frozen", "trained", "frozen"]
    assert records.read_completed_case(case_dir)["cycle"].tolist() == [0, 0, 1, 1]
