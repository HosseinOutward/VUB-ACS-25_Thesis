"""Run protocol cases at arbitrary positions in a saved federated-learning trace."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from importlib import import_module
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import torch

from Experiments.Protocol_test.replay import (
    HistorySource,
    build_state_manager,
    load_replay_config,
    load_replay_tensor,
    replace_latest_reconstruction,
    seed_history,
)
from FL_code.FL_core.codec import Access, BaseProtocol, CompressionRecord, create_protocol, simulate_compression
from FL_code.cancer_protocol.cancer_protocol import _WZProtocol
from FL_code.FL_core.utils import StateDictManager, set_global_seed, write_fl_config_snapshot


class ReplayCase(BaseModel):
    """One protocol placement, including its absolute FL position and evaluated clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    codec_id: str
    protocol: str
    start_round: int = Field(ge=0)
    client_ids: tuple[int, ...]
    round_codecs: tuple[str, ...]

    @field_validator("case_id", "codec_id")
    @classmethod
    def validate_path_component(cls, value: str) -> str:
        """Require stable identifiers that cannot escape their result directories."""
        assert value not in {"", ".", ".."} and Path(value).name == value, (
            f"Replay identifiers must be single path components; got {value!r}."
        )
        return value

    @model_validator(mode="after")
    def validate_case(self) -> ReplayCase:
        """Require ordered unique clients and a non-empty exact codec schedule."""
        assert self.client_ids and tuple(sorted(set(self.client_ids))) == self.client_ids, (
            f"Case {self.case_id!r} must select unique clients in ascending order."
        )
        assert all(client_id >= 0 for client_id in self.client_ids)
        assert self.round_codecs and all(self.round_codecs)
        return self


class ReplayPlan(BaseModel):
    """The complete sparse experiment configuration consumed by workers and plotting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dataset_name: str
    model_name: str
    fl_config: Path
    deltas: Path
    context: Path
    output_root: Path
    cache_root: Path
    cases: tuple[ReplayCase, ...]
    comparisons: tuple[tuple[str, str], ...] = ()
    context_source: HistorySource = HistorySource.RECONSTRUCTIONS
    replace_selected_with_raw: bool = False
    codec_packages: tuple[str, ...] = ()
    protocol_factory: str | None = None
    seed: int = Field(default=43, ge=0)
    gpu_ids: tuple[int, ...] = (0,)
    workers_per_gpu: int = Field(default=1, ge=1)
    cache_overwrite: bool = False

    @model_validator(mode="after")
    def validate_plan(self) -> ReplayPlan:
        """Reject ambiguous cases, devices, and comparisons."""
        assert self.name and self.dataset_name and self.model_name
        identities = [(case.codec_id, case.case_id) for case in self.cases]
        assert self.cases and len(set(identities)) == len(identities), (
            "Replay cases must have unique (codec_id, case_id) identities."
        )
        assert self.gpu_ids and len(set(self.gpu_ids)) == len(self.gpu_ids)
        assert all(gpu_id >= 0 for gpu_id in self.gpu_ids)
        codecs = {case.codec_id for case in self.cases}
        assert len(set(self.comparisons)) == len(self.comparisons)
        for candidate, reference in self.comparisons:
            assert candidate != reference and {candidate, reference} <= codecs, (
                f"Invalid comparison ({candidate!r}, {reference!r}); codecs are {sorted(codecs)}."
            )
        return self


ProtocolFactory = Callable[[ReplayCase, StateDictManager, Path, bool], _WZProtocol]


def load_plan(path: Path) -> ReplayPlan:
    """Load a JSON plan, resolving its filesystem paths relative to the plan file."""
    assert path.is_file(), f"Missing sparse replay plan: {path}."
    payload = json.loads(path.read_text())
    for field in ("fl_config", "deltas", "context", "output_root", "cache_root"):
        value = Path(payload[field])
        payload[field] = value if value.is_absolute() else path.parent / value
    return ReplayPlan.model_validate(payload)


def _atomic_write(output_path: Path, write: Callable[[Path], None]) -> None:
    """Publish a derived artifact only after its temporary file is complete."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}")
    try:
        write(temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _case_dir(plan: ReplayPlan, case: ReplayCase) -> Path:
    return plan.output_root / "cases" / case.codec_id / case.case_id


def _resolved_schedule(protocol: BaseProtocol, case: ReplayCase) -> tuple[str, ...]:
    snapshot = protocol.config_snapshot()
    warmup = tuple(snapshot["warmup_round_codecs"])
    routine = tuple(snapshot["routine_round_codecs"])
    return tuple(
        warmup[round_id]
        if round_id < len(warmup)
        else routine[(round_id - len(warmup)) % len(routine)]
        for round_id in range(case.start_round, case.start_round + len(case.round_codecs))
    )


def default_protocol_factory(
    case: ReplayCase, state_manager: StateDictManager, cache_dir: Path, cache_overwrite: bool
) -> _WZProtocol:
    """Create the declared protocol and align the case schedule to its absolute start round."""
    protocol = create_protocol(
        case.protocol, state_manager.get_slices(), cache_dir, cache_overwrite
    )
    assert isinstance(protocol, _WZProtocol)
    assert case.start_round >= len(protocol.warmup_round_codecs), (
        "The default sparse factory can place schedules only after protocol warmup; use "
        "protocol_factory for a warmup case."
    )
    phase = (case.start_round - len(protocol.warmup_round_codecs)) % len(case.round_codecs)
    protocol.routine_round_codecs = (
        case.round_codecs[-phase:] + case.round_codecs[:-phase] if phase else case.round_codecs
    )
    protocol._round_codec_classes = protocol._validate_round_codecs()
    return protocol


def _load_factory(identity: str | None) -> ProtocolFactory:
    if identity is None:
        return default_protocol_factory
    module_name, separator, qualified_name = identity.partition(":")
    assert separator and module_name and qualified_name, (
        f"protocol_factory must be 'module:qualified_name'; got {identity!r}."
    )
    value: object = import_module(module_name)
    for part in qualified_name.split("."):
        value = getattr(value, part)
    assert callable(value), f"Protocol factory is not callable: {identity}."
    return cast(ProtocolFactory, value)


def _validate_records(path: Path, case: ReplayCase) -> pd.DataFrame:
    assert path.is_file(), f"Missing completed case records: {path}."
    records = pd.read_csv(path)
    expected = [
        (round_id, client_id, codec)
        for round_id, codec in enumerate(case.round_codecs, start=case.start_round)
        for client_id in case.client_ids
    ]
    actual = list(records[["round_id", "client_id", "round_name_full"]].itertuples(index=False, name=None))
    assert actual == expected, f"Case records do not match {case.codec_id}/{case.case_id}: {path}."
    return records


def _completion_payload(plan: ReplayPlan, case: ReplayCase) -> dict[str, Any]:
    return {
        "case": case.model_dump(mode="json"),
        "context_source": plan.context_source,
        "replace_selected_with_raw": plan.replace_selected_with_raw,
        "seed": plan.seed,
    }


def completed_case_paths(plan: ReplayPlan) -> tuple[tuple[ReplayCase, Path], ...]:
    """Return validated records for every case carrying its matching completion marker."""
    completed: list[tuple[ReplayCase, Path]] = []
    for case in plan.cases:
        case_dir = _case_dir(plan, case)
        completion = case_dir / "complete.json"
        if not completion.is_file():
            continue
        assert json.loads(completion.read_text()) == _completion_payload(plan, case), (
            f"Completed case configuration differs: {completion}."
        )
        records_path = case_dir / "compression_records.csv"
        _validate_records(records_path, case)
        completed.append((case, records_path))
    return tuple(completed)


def run_case(plan: ReplayPlan, case: ReplayCase, protocol_factory: ProtocolFactory) -> Path:
    """Replay one placement while advancing every unselected client's surrounding context."""
    case_dir = _case_dir(plan, case)
    records_path = case_dir / "compression_records.csv"
    completion_path = case_dir / "complete.json"
    expected_completion = _completion_payload(plan, case)
    if case_dir.exists():
        assert completion_path.is_file(), (
            f"Incomplete case directory exists; remove it explicitly before rerunning: {case_dir}."
        )
        assert json.loads(completion_path.read_text()) == expected_completion
        _validate_records(records_path, case)
        return records_path

    for package in plan.codec_packages:
        import_module(package)
    set_global_seed(plan.seed)
    cfg = load_replay_config(plan.fl_config, case.protocol, f"{plan.name}_{case.codec_id}_{case.case_id}",
                             case_dir, plan.seed)
    assert (cfg.dataset_name, cfg.model_name) == (plan.dataset_name, plan.model_name)
    assert max(case.client_ids) < cfg.num_clients
    state_manager = build_state_manager(cfg)
    protocol = protocol_factory(
        case, state_manager, plan.cache_root / case.codec_id, plan.cache_overwrite
    )
    assert protocol.protocol_name_full == case.protocol
    assert _resolved_schedule(protocol, case) == case.round_codecs, (
        f"Factory schedule for {case.codec_id}/{case.case_id} does not match {case.round_codecs}."
    )

    client_ids = range(cfg.num_clients)
    seed_history(protocol, plan.context_source, case.start_round, client_ids, plan.context, plan.context)
    case_dir.mkdir(parents=True)
    write_fl_config_snapshot(cfg, case_dir)
    (case_dir / "protocol_config.json").write_text(json.dumps(protocol.config_snapshot(), indent=2))

    selected = set(case.client_ids)
    history_dtypes = {history.dtype for history in protocol.histories}
    assert len(history_dtypes) == 1
    history_dtype = next(iter(history_dtypes))
    for round_id in range(case.start_round, case.start_round + len(case.round_codecs)):
        for client_id in client_ids:
            if client_id in selected:
                delta = load_replay_tensor(plan.deltas, round_id, client_id)
                assert delta.numel() == state_manager.param_count
                simulate_compression(protocol, delta, client_id, round_id, state_manager, case_dir, {}, (), ())
                if plan.replace_selected_with_raw:
                    replace_latest_reconstruction(protocol, delta, round_id, client_id)
                continue

            context = load_replay_tensor(plan.context, round_id, client_id)
            record = CompressionRecord(round_id, client_id, f"context_{plan.context_source.value}")
            for history in protocol.histories:
                history.commit(context.to(dtype=history_dtype), record, Access.TEMPORAL_TOO)

    _validate_records(records_path, case)
    _atomic_write(
        completion_path,
        lambda temporary: temporary.write_text(json.dumps(expected_completion, indent=2)),
    )
    return records_path


def aggregate_completed(plan: ReplayPlan) -> dict[str, Path]:
    """Aggregate completed placements by codec, retaining absolute and relative positions."""
    grouped: defaultdict[str, list[tuple[ReplayCase, Path]]] = defaultdict(list)
    for case, path in completed_case_paths(plan):
        grouped[case.codec_id].append((case, path))

    outputs: dict[str, Path] = {}
    for codec_id, cases in grouped.items():
        frames = []
        for case, path in cases:
            records = pd.read_csv(path)
            records.insert(0, "relative_round", records["round_id"] - case.start_round)
            records.insert(0, "case_id", case.case_id)
            records.insert(0, "codec_id", codec_id)
            frames.append(records)
        output = plan.output_root / "records" / codec_id / "compression_records.csv"
        _atomic_write(output, lambda temporary, frames=frames: pd.concat(frames).to_csv(temporary, index=False))
        outputs[codec_id] = output
    return outputs


def _write_manifest(plan: ReplayPlan, factory: ProtocolFactory) -> None:
    plan.output_root.mkdir(parents=True, exist_ok=True)
    path = plan.output_root / "test_config.json"
    payload = {
        "schema_version": 2,
        "protocol_factory": f"{factory.__module__}:{factory.__qualname__}",
        "plan": plan.model_dump(mode="json"),
    }
    if path.exists():
        assert json.loads(path.read_text()) == payload, f"Existing replay plan differs: {path}."
        return
    assert not any(plan.output_root.iterdir()), f"Output root is non-empty without a manifest: {plan.output_root}."
    _atomic_write(path, lambda temporary: temporary.write_text(json.dumps(payload, indent=2)))


def _pin_gpu(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def run_sparse_plan(plan: ReplayPlan, *, refresh_outputs: bool = True) -> tuple[Path, ...]:
    """Run all placements in isolated GPU workers and refresh aggregates after each completion."""
    assert plan.fl_config.is_file() and plan.deltas.is_dir() and plan.context.is_dir()
    configured = json.loads(plan.fl_config.read_text())
    assert (configured.get("dataset_name"), configured.get("model_name")) == (
        plan.dataset_name, plan.model_name
    )
    assert max(plan.gpu_ids) < torch.cuda.device_count(), (
        f"Requested GPUs {plan.gpu_ids}, but torch reports {torch.cuda.device_count()}."
    )
    factory = _load_factory(plan.protocol_factory)
    _write_manifest(plan, factory)
    results: dict[tuple[str, str], Path] = {}
    context = multiprocessing.get_context("spawn")
    with ExitStack() as stack:
        executors = {
            gpu_id: stack.enter_context(ProcessPoolExecutor(
                max_workers=plan.workers_per_gpu,
                mp_context=context,
                initializer=_pin_gpu,
                initargs=(gpu_id,),
                max_tasks_per_child=1,
            ))
            for gpu_id in plan.gpu_ids
        }
        futures: dict[Future[Path], ReplayCase] = {
            executors[plan.gpu_ids[index % len(plan.gpu_ids)]].submit(run_case, plan, case, factory): case
            for index, case in enumerate(plan.cases)
        }
        for future in as_completed(futures):
            case = futures[future]
            results[(case.codec_id, case.case_id)] = future.result()
            if refresh_outputs:
                aggregate_completed(plan)
                from Experiments.Protocol_test.plot import plot_sparse_results
                plot_sparse_results(plan.output_root, plan.comparisons)

    return tuple(results[(case.codec_id, case.case_id)] for case in plan.cases)


def parse_args() -> argparse.Namespace:
    """Parse a sparse-plan JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--no-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load and execute one sparse replay plan."""
    args = parse_args()
    run_sparse_plan(load_plan(args.plan), refresh_outputs=not args.no_refresh)


if __name__ == "__main__":
    main()
