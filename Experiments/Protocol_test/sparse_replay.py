"""Compare short protocol windows on one deterministic, uncompressed FL trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from Experiments.Protocol_test import records
from FL_code.FL_core.explicit_cache import CacheEntry, TorchCodec, atomic_output, content_hash
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
_EXPERIMENT_SEMANTICS_VERSION = 10
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
    protocol = create_protocol(case.protocol, None if manager is None else manager.get_slices())
    assert isinstance(protocol, _WZProtocol), "Sparse history experiments require a WZ protocol."
    protocol.set_schedule(case.start_round, case.round_codecs)
    return protocol


def _case_dir(plan: ReplayPlan, case: ReplayCase) -> Path:
    return plan.output_root / "cases" / case.codec_id / case.case_id


def _codec_seed(seed: int, start_round: int, round_id: int, client_id: int) -> int:
    """Derive a per-client-round seed so one fit cannot change another's stochastic training."""
    return int(content_hash((seed, start_round, round_id, client_id))[:8], 16)


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
        "plan": json.loads(plan_path.read_text()),
    }


def _checkpoint_rounds(windows: Sequence[tuple[int, int]]) -> list[int]:
    """Pick, for `(first, last)` trace windows run in this order, the round whose initial state each saves.

    Only one continuation state is kept per trajectory, so a case leaves it where the next case's
    window begins, or right after its own last round when the next window begins beyond that.
    """
    return [min(last + 1, next_first) for (_, last), (next_first, _) in pairwise(windows)] + [windows[-1][1] + 1]


def run_case(plan: ReplayPlan, case: ReplayCase, provenance: Mapping[str, Any], checkpoint_round: int) -> Path:
    """Evaluate selected clients while decoded and unselected raw updates advance history."""
    trace = Trace(
        load_fl_config(plan.fl_config), dataset_revision=plan.dataset_revision,
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
        with atomic_output(completion) as temporary:
            temporary.write_text(json.dumps({"identity": identity, "provenance": provenance}, indent=2))
        return records_path
    case_dir.mkdir(parents=True)
    with atomic_output(pre_run_identity) as temporary:
        temporary.write_text(json.dumps({"identity": identity}, indent=2))

    history_name = "raw_history" if case.initial_si == "raw" else f"nsplit{plan.nsplit_bins}_history"
    last_round = case.start_round + len(case.round_codecs) - 1
    # Keyed only on the trajectory, so each save replaces the previous case's continuation state.
    checkpoint = CacheEntry(_REPO_ROOT / "_/data/trace_states", trace_identity, TorchCodec())
    for round_id, deltas in trace.stream(case.start_round - depth, last_round, checkpoint, checkpoint_round):
        for client_id, delta in enumerate(deltas):
            if round_id < case.start_round:
                delta = nsplit_quantize(delta, plan.nsplit_bins) if case.initial_si == "nsplit" else delta
                for history in protocol.histories:
                    history.commit(
                        delta, CompressionRecord(round_id, client_id, history_name), Access.TEMPORAL_TOO)
                continue
            assert delta.numel() == manager.param_count
            if client_id in case.client_ids:
                set_global_seed(_codec_seed(plan.seed, case.start_round, round_id, client_id))
                simulate_compression(protocol, delta, client_id, round_id, manager, case_dir, {}, (), ())
            else:
                for history in protocol.histories:
                    history.commit(delta, CompressionRecord(round_id, client_id, "raw_history"), Access.TEMPORAL_TOO)

    records.validate_case_records(records_path, identity["case"])
    with atomic_output(completion) as temporary:
        temporary.write_text(json.dumps({"identity": identity, "provenance": provenance}, indent=2))
    return records_path


def _run_gpu_case(
    plan: ReplayPlan, gpu_id: int, case: ReplayCase, provenance: Mapping[str, Any], checkpoint_round: int,
) -> Path:
    """Run one case in a disposable GPU worker so allocator growth cannot cross case boundaries."""
    torch.cuda.set_device(gpu_id)
    return run_case(plan, case, provenance, checkpoint_round)


def _run_gpu_cases(
    plan: ReplayPlan, provenance: Mapping[str, Any], checkpoint_rounds: Mapping[ReplayCase, int],
) -> tuple[Path, ...]:
    """Run memory-bounded batches, in `checkpoint_rounds` order, each case in a disposable GPU worker."""
    assert plan.gpu_ids
    context = multiprocessing.get_context("spawn")
    results: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=plan.max_parallel_cases,
        mp_context=context,
        max_tasks_per_child=1,
    ) as pool:
        cases = tuple(checkpoint_rounds)
        for start in range(0, len(cases), plan.max_parallel_cases):
            batch = cases[start : start + plan.max_parallel_cases]
            futures = [
                pool.submit(
                    _run_gpu_case,
                    plan,
                    plan.gpu_ids[(start + offset) % len(plan.gpu_ids)],
                    case,
                    provenance,
                    checkpoint_rounds[case],
                )
                for offset, case in enumerate(batch)
            ]
            results.extend(future.result() for future in futures)
    return tuple(results)


def run_sparse_plan(plan: ReplayPlan, plan_path: Path) -> tuple[Path, ...]:
    """Run cases in trace-window order, each resuming the FL trajectory from the last saved state."""
    cfg = load_fl_config(plan.fl_config)
    validation_protocols = [case_protocol(plan, case) for case in plan.cases]
    windows: dict[ReplayCase, tuple[int, int]] = {}
    for case, protocol in zip(plan.cases, validation_protocols, strict=True):
        depth = protocol.max_per_client_recons_history
        assert depth is not None and case.start_round >= depth > 0, (
            f"Case {case.case_id} needs {depth} preceding raw rounds.")
        assert max(case.client_ids) < cfg.num_clients
        windows[case] = (case.start_round - depth, case.start_round + len(case.round_codecs) - 1)
    # Execution order: the earliest trace window first, so each case can resume from its predecessor.
    ordered = sorted(windows, key=lambda case: windows[case])
    checkpoint_rounds = dict(zip(ordered, _checkpoint_rounds([windows[case] for case in ordered]), strict=True))
    trace_identity = Trace(
        cfg, dataset_revision=plan.dataset_revision,
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
        with atomic_output(manifest_path) as temporary:
            temporary.write_text(json.dumps({"identity": identity, "provenance": provenance}, indent=2))

    return (
        _run_gpu_cases(plan, provenance, checkpoint_rounds)
        if plan.gpu_ids
        else tuple(run_case(plan, case, provenance, round_id) for case, round_id in checkpoint_rounds.items())
    )


def main() -> None:
    """Run the sparse experiment declared in a JSON plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    run_sparse_plan(load_plan(args.plan), args.plan)


if __name__ == "__main__":
    main()
