"""Compare short protocol windows on one deterministic, uncompressed FL trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import subprocess
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from Experiments.Protocol_test import records
from FL_code.FL_core.cache import content_hash
from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol, simulate_compression
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, parse_configurable_name, set_global_seed
from FL_code.cancer_protocol.cancer_protocol import _WZProtocol
from FL_code.run_fl import FLConfig
from temp_nsplit import nsplit_quantize

# Bump whenever this module's execution semantics change in a way that could produce different
# records for an unchanged plan/trace, so a completed case cannot be silently reused as if it
# still matched.
_EXPERIMENT_SEMANTICS_VERSION = 9
_CODEC_SEED_POLICY = "content_hash(seed,start_round,round_id,client_id)[:8]/v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ReplayCase(BaseModel):
    """A protocol window and the clients whose compression records the runner measures."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    codec_id: str
    protocol: str = "wz_cancer"
    start_round: int = Field(ge=0)
    initial_si: Literal["raw", "nsplit"] = "raw"
    client_ids: tuple[int, ...]
    round_codecs: tuple[str, ...]

    @field_validator("case_id", "codec_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """Keep case output beneath its experiment directory."""
        assert value not in {"", ".", ".."} and Path(value).name == value
        return value

    @model_validator(mode="after")
    def validate_window(self) -> ReplayCase:
        """Require an ordered client subset and an explicit train/freeze protocol window."""
        assert self.client_ids and tuple(sorted(set(self.client_ids))) == self.client_ids
        assert min(self.client_ids) >= 0
        assert len(self.round_codecs) >= 2, "round_codecs must contain at least two rounds."
        assert self.round_codecs[0] != "F", "The first schedule token must train a codec, not 'F'."
        assert "F" in self.round_codecs, "round_codecs must contain at least one frozen round."
        return self


class ReplayPlan(BaseModel):
    """One raw FL trace shared by independent sparse protocol cases and their outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    description: str | None = None
    fl_config: Path
    trace_device: str = "cuda:0"
    dataset_revision: str
    output_root: Path
    nsplit_bins: int = Field(default=2, ge=2, le=256)
    cases: tuple[ReplayCase, ...]
    comparisons: tuple[tuple[str, str], ...] = ()
    seed: int = Field(default=43, ge=0)
    gpu_ids: tuple[int, ...] = ()
    max_parallel_cases: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_cases(self) -> ReplayPlan:
        """Require unique case identities, valid comparisons, and the plan's seed in every case."""
        assert self.name and self.cases
        identities = [(case.codec_id, case.case_id) for case in self.cases]
        assert len(set(identities)) == len(identities), "Cases must have unique (codec_id, case_id) identities."
        assert len(set(self.gpu_ids)) == len(self.gpu_ids) and all(gpu >= 0 for gpu in self.gpu_ids)
        assert self.max_parallel_cases <= max(1, len(self.gpu_ids)), (
            "max_parallel_cases cannot exceed the number of configured GPUs."
        )
        codecs = {case.codec_id for case in self.cases}
        for candidate, reference in self.comparisons:
            assert candidate != reference and {candidate, reference} <= codecs, "Invalid comparison."
        for case in self.cases:
            for token in case.round_codecs:
                if token == "F":
                    continue
                options = parse_configurable_name(token, "round codec").options or {}
                assert options.get("training_seed") == self.seed, (
                    f"Case {case.codec_id}/{case.case_id} training token must carry "
                    f"training_seed={self.seed}; got {options.get('training_seed')!r}."
                )
        return self


def load_plan(path: Path) -> ReplayPlan:
    """Read one plan, resolving its paths relative to the JSON file."""
    payload = json.loads(path.read_text())
    for field in ("fl_config", "output_root"):
        payload[field] = (path.parent / payload[field]).resolve()
    return ReplayPlan.model_validate(payload)


def load_fl_config(path: Path) -> FLConfig:
    """Read the identity FL setup, with its dataset path relative to the config file."""
    payload = json.loads(path.read_text())
    payload["data_folder"] = (path.parent / payload.get("data_folder", "data")).resolve()
    cfg = FLConfig.model_validate(payload)
    assert cfg.protocol == "identity", "Sparse experiments require an identity FL trace."
    return cfg


def case_protocol(plan: ReplayPlan, case: ReplayCase, manager: StateDictManager | None = None) -> _WZProtocol:
    """Build the case using the protocol's schedule and codec configuration APIs."""
    protocol = create_protocol(
        case.protocol,
        None if manager is None else manager.get_slices(),
        cache_dir=None,
    )
    assert isinstance(protocol, _WZProtocol), "Sparse history experiments require a WZ protocol."
    protocol.set_schedule(case.start_round, case.round_codecs)
    return protocol


def _case_dir(plan: ReplayPlan, case: ReplayCase) -> Path:
    return plan.output_root / "cases" / case.codec_id / case.case_id


def _codec_seed(seed: int, start_round: int, round_id: int, client_id: int) -> int:
    """Derive a per-client-round seed so one fit cannot change another's stochastic training."""
    return int(content_hash((seed, start_round, round_id, client_id), {})[:8], 16)


def _case_metadata(case: ReplayCase, protocol: _WZProtocol) -> dict[str, Any]:
    """Return the schedule contract shared by case and experiment identities."""
    return {
        "protocol": case.protocol,
        "start_round": case.start_round,
        "initial_si": case.initial_si,
        "client_ids": list(case.client_ids),
        "round_codecs": list(case.round_codecs),
        "warm_retrain_repeats": getattr(protocol, "warm_retrain_repeats", None),
    }


def _case_identity(plan: ReplayPlan, case: ReplayCase, trace_identity: Mapping[str, Any],
                    protocol: _WZProtocol) -> dict[str, Any]:
    """Build the part of `complete.json` that decides whether a case can be reused."""
    protocol_snapshot = json.loads(json.dumps(protocol.config_snapshot()))
    return {
        "experiment_semantics_version": _EXPERIMENT_SEMANTICS_VERSION,
        "trace": trace_identity,
        "seed": plan.seed,
        "codec_seed_policy": _CODEC_SEED_POLICY,
        "raw_update_cache": "disabled",
        "case": _case_metadata(case, protocol),
        "protocol_snapshot": protocol_snapshot,
        "history_representation": protocol_snapshot["history"],
        "oracle_assumptions": {
            "initial_history_source": (
                "raw_updates" if case.initial_si == "raw" else f"nsplit{plan.nsplit_bins}_updates"),
            "history_replacement": "decoded_updates_after_start",
            "client_order": "normal",
            "frozen_si_refresh": protocol.refresh_frozen_si,
            "frozen_prior_refit": protocol.refit_priors_on_frozen,
            "server_history": "decoder_visible_reconstructions",
        },
    }


def _plan_identity(plan: ReplayPlan, trace_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Build the semantic identity of the whole experiment; paths are provenance, not identity."""
    return {
        "experiment_semantics_version": _EXPERIMENT_SEMANTICS_VERSION,
        "name": plan.name,
        "seed": plan.seed,
        "dataset_revision": plan.dataset_revision,
        "nsplit_bins": plan.nsplit_bins,
        "raw_update_cache": "disabled",
        "trace": trace_identity,
        "cases": [
            {"case_id": case.case_id, "codec_id": case.codec_id}
            | _case_metadata(case, case_protocol(plan, case))
            for case in plan.cases
        ],
        "comparisons": [list(pair) for pair in plan.comparisons],
    }


def _git_provenance() -> dict[str, Any]:
    """Fingerprint uncommitted source once per run: `dirty=true` alone is not enough provenance."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=_REPO_ROOT, check=True,
                               capture_output=True, text=True).stdout

    watched = ("FL_code", "Experiments/Protocol_test")
    head = git("rev-parse", "HEAD").strip()
    tracked_diff = git("diff", "HEAD", "--", *watched)
    changed_tracked = sorted(line for line in git("diff", "HEAD", "--name-only", "--", *watched).splitlines() if line)
    untracked = sorted(line for line in git("ls-files", "--others", "--exclude-standard", "--", *watched).splitlines()
                        if line.endswith(".py"))
    digest = hashlib.sha256(tracked_diff.encode())
    for relative_path in untracked:
        digest.update((_REPO_ROOT / relative_path).read_bytes())
    return {
        "git_head": head, "dirty_source_fingerprint": digest.hexdigest(),
        "dirty_source_paths": changed_tracked + untracked,
    }


def _provenance(plan: ReplayPlan, plan_path: Path) -> dict[str, Any]:
    """Build the explanatory (never reuse-deciding) half of `complete.json`/`test_config.json`."""
    return {
        **_git_provenance(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(gpu) for gpu in plan.gpu_ids] if torch.cuda.is_available() else [],
        "plan_path": str(plan_path),
        "output_root": str(plan.output_root),
        "gpu_ids": list(plan.gpu_ids),
        "max_parallel_cases": plan.max_parallel_cases,
        "raw_update_cache": "disabled",
        "server_checkpoint_scope": "run",
        "plan": json.loads(plan_path.read_text()),
    }


def run_case(
    plan: ReplayPlan, case: ReplayCase, provenance: Mapping[str, Any], checkpoint_dir: Path,
) -> Path:
    """Evaluate selected clients while decoded and unselected raw updates advance history."""
    trace = Trace(
        checkpoint_dir, load_fl_config(plan.fl_config), dataset_revision=plan.dataset_revision,
        device=torch.device(plan.trace_device),
    )
    cfg = trace.cfg
    trace_identity = trace.identity()
    manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
    protocol = case_protocol(plan, case, manager)
    depth = protocol.max_per_client_recons_history
    assert depth is not None and case.start_round >= depth > 0, (
        f"Case {case.case_id} needs {depth} preceding raw rounds.")
    assert max(case.client_ids) < cfg.num_clients
    case_dir = _case_dir(plan, case)
    records_path = case_dir / "compression_records.csv"
    identity = _case_identity(plan, case, trace_identity, protocol)
    completion = case_dir / "complete.json"
    pre_run_identity = case_dir / "pre_run_identity.json"
    if pre_run_identity.exists():
        stored = json.loads(pre_run_identity.read_text())
        mismatched = records.identity_mismatches(identity, stored["identity"])
        assert not mismatched, f"Pre-run case identity differs at {case_dir}: {mismatched}."
    if completion.exists():
        records.read_completed_case(case_dir, expected_identity=identity)
        return records_path
    if case_dir.exists():
        assert pre_run_identity.is_file(), (
            f"Existing case output lacks pre_run_identity.json, so recovery is unsafe: {case_dir}."
        )
        records.validate_case_records(records_path, identity["case"])
        temporary = completion.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": identity, "provenance": provenance}, indent=2))
        temporary.replace(completion)
        return records_path
    case_dir.mkdir(parents=True)
    temporary = pre_run_identity.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity}, indent=2))
    temporary.replace(pre_run_identity)

    checkpoint_rounds = {candidate.start_round for candidate in plan.cases}
    history_name = "raw_history" if case.initial_si == "raw" else f"nsplit{plan.nsplit_bins}_history"
    for round_id, deltas in trace.stream(
        case.start_round - depth, case.start_round - 1, checkpoint_rounds=checkpoint_rounds,
    ):
        for client_id, delta in enumerate(deltas):
            delta = nsplit_quantize(delta, plan.nsplit_bins) if case.initial_si == "nsplit" else delta
            for history in protocol.histories:
                history.commit(delta, CompressionRecord(round_id, client_id, history_name), Access.TEMPORAL_TOO)

    last_round = case.start_round + len(case.round_codecs) - 1
    for round_id, deltas in trace.stream(
        case.start_round, last_round, checkpoint_rounds=checkpoint_rounds,
    ):
        for client_id, delta in enumerate(deltas):
            assert delta.numel() == manager.param_count
            if client_id in case.client_ids:
                set_global_seed(_codec_seed(plan.seed, case.start_round, round_id, client_id))
                simulate_compression(protocol, delta, client_id, round_id, manager, case_dir, {}, (), ())
            else:
                for history in protocol.histories:
                    history.commit(delta, CompressionRecord(round_id, client_id, "raw_history"), Access.TEMPORAL_TOO)

    records.validate_case_records(records_path, identity["case"])
    temporary = completion.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "provenance": provenance}, indent=2))
    temporary.replace(completion)
    return records_path


def _run_gpu_case(
    plan: ReplayPlan, gpu_id: int, case: ReplayCase, provenance: Mapping[str, Any], checkpoint_dir: Path,
) -> Path:
    """Run one case in a disposable GPU worker so allocator growth cannot cross case boundaries."""
    torch.cuda.set_device(gpu_id)
    return run_case(plan, case, provenance, checkpoint_dir)


def _run_gpu_cases(
    plan: ReplayPlan, provenance: Mapping[str, Any], checkpoint_dir: Path,
) -> tuple[Path, ...]:
    """Run memory-bounded batches whose cases each get a disposable GPU worker."""
    assert plan.gpu_ids
    context = multiprocessing.get_context("spawn")
    results: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=plan.max_parallel_cases,
        mp_context=context,
        max_tasks_per_child=1,
    ) as pool:
        cases = tuple(sorted(plan.cases, key=lambda case: case.start_round))
        for start in range(0, len(cases), plan.max_parallel_cases):
            batch = cases[start : start + plan.max_parallel_cases]
            futures = [
                pool.submit(
                    _run_gpu_case,
                    plan,
                    plan.gpu_ids[(start + offset) % len(plan.gpu_ids)],
                    case,
                    provenance,
                    checkpoint_dir,
                )
                for offset, case in enumerate(batch)
            ]
            results.extend(future.result() for future in futures)
    return tuple(results)


def run_sparse_plan(plan: ReplayPlan, plan_path: Path) -> tuple[Path, ...]:
    """Stream every case and share only run-scoped start-point checkpoints between them."""
    cfg = load_fl_config(plan.fl_config)
    validation_protocols = [case_protocol(plan, case) for case in plan.cases]
    for case, protocol in zip(plan.cases, validation_protocols, strict=True):
        depth = protocol.max_per_client_recons_history
        assert depth is not None and case.start_round >= depth > 0, (
            f"Case {case.case_id} needs {depth} preceding raw rounds.")
        assert max(case.client_ids) < cfg.num_clients
    trace_identity = Trace(
        plan.output_root / ".start-point-models", cfg, dataset_revision=plan.dataset_revision,
        device=torch.device(plan.trace_device),
    ).identity()
    provenance = _provenance(plan, plan_path)

    identity = _plan_identity(plan, trace_identity)
    manifest_path = plan.output_root / "test_config.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        mismatched = records.identity_mismatches(identity, stored["identity"])
        assert not mismatched, f"Experiment identity differs from expected at {manifest_path}: {mismatched}."
    else:
        plan.output_root.mkdir(parents=True, exist_ok=True)
        assert not any(plan.output_root.iterdir()), f"Output root contains untracked experiment results: {plan.output_root}."
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": identity, "provenance": provenance}, indent=2))
        temporary.replace(manifest_path)

    with TemporaryDirectory(prefix=".start-point-models-", dir=plan.output_root) as scratch:
        checkpoint_dir = Path(scratch)
        return (
            _run_gpu_cases(plan, provenance, checkpoint_dir)
            if plan.gpu_ids
            else tuple(
                run_case(plan, case, provenance, checkpoint_dir)
                for case in sorted(plan.cases, key=lambda item: item.start_round)
            )
        )


def main() -> None:
    """Run the sparse experiment declared in a JSON plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    run_sparse_plan(load_plan(args.plan), args.plan)


if __name__ == "__main__":
    main()
