"""Reconstruct the sampled study's starting model and measure its frozen three-plane prefix."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from Experiments.Protocol_test.live_plot import watch
from Experiments.Protocol_test.sparse_replay import load_fl_config
from FL_code.FL_core.codec import CompressionRecord, record_reconstruction_metrics
from FL_code.FL_core.explicit_cache import CacheEntry, TorchCodec, atomic_output, content_hash
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, payload_size_mib, set_global_seed
from FL_code.cancer_protocol.bin_coder.coded_rounds import OracleRansRoundCodec
from FL_code.cancer_protocol.cancer_protocol import WZCancerProtocol
from FL_code.cancer_protocol.round_codecs import Oracle_RoundCodec
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant


class FrozenStartingPlan(BaseModel):
    """Static-gradient reconstruction settings read by the worker and detached supervisor."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    output_root: Path
    fl_config: Path
    dataset_revision: str
    trace_device: str
    fit_device: str
    seed: int = 43
    target_round: int = 5
    target_client: int = 0
    baseline_client: int = 1
    quantizer: WZcfgQuant = WZcfgQuant(bins_per_plane=2, num_planes=9, training_log_every_epochs=10)

    @property
    def cases(self) -> tuple[str, ...]:
        """Expose the single measured F case to the supervisor's health reporting."""
        return ("F",)


def load_plan(path: Path) -> FrozenStartingPlan:
    """Read one named reconstruction plan with paths relative to its JSON file."""
    plan = FrozenStartingPlan.model_validate_json(path.read_text())
    assert plan.target_client != plan.baseline_client
    assert plan.quantizer.num_planes == 9 and plan.quantizer.bins_per_plane == 2
    assert not plan.quantizer.warm_training and plan.quantizer.norm_slices is None
    return plan.model_copy(update={key: (path.parent / getattr(plan, key)).resolve()
                                   for key in ("output_root", "fl_config")})


def collect_problem(trace: Trace, plan: FrozenStartingPlan) -> dict[str, Any]:
    """Replay raw history through the protocol ledger and capture round-start target, baseline and SI."""
    manager = StateDictManager(initialize_model(trace.cfg, torch.device("cpu")))
    slices = tuple(manager.get_slices())
    protocol = WZCancerProtocol(sd_slices=slices)
    reader = Oracle_RoundCodec(dict(bins_per_plane=2, num_planes=3, sd_slices=slices), "O")
    depth = protocol.max_per_client_recons_history
    assert depth is not None and plan.target_round >= depth
    for round_id, deltas in trace.stream(plan.target_round - depth, plan.target_round):
        print(f"TRACE round {round_id}", flush=True)
        if round_id == plan.target_round:
            _, si = protocol.training_problem(reader, round_id, plan.target_client)
            return dict(target=deltas[plan.target_client], baseline=deltas[plan.baseline_client],
                        si=tuple(si), slices=slices)
        for client_id, delta in enumerate(deltas):
            protocol.after_decode(delta, CompressionRecord(round_id, client_id, "raw_history"), reader)
    raise AssertionError(f"Trace did not produce target round {plan.target_round}.")


def fit_starting_model(codec: Oracle_RoundCodec, target: torch.Tensor, seed: int) -> dict[str, Any]:
    """Cold-fit client 1 and retain joint model weights before any separate entropy-prior fit."""
    set_global_seed(seed)
    started = perf_counter()
    codec.pre_encode_training_steps(target)
    assert codec.joint_quantizer_state is not None
    return dict(state=codec.joint_quantizer_state, seconds=perf_counter() - started,
                stage_rates=codec.quantizer.quantizer_prior_stage_rates,
                training_provenance=codec.quantizer.training_provenance)


def evaluate_frozen(plan: FrozenStartingPlan, problem: dict[str, Any], fitted: dict[str, Any]) -> dict[str, Any]:
    """Measure F with the common model's three-plane prefix, checking that all weights stay fixed."""
    codec = OracleRansRoundCodec(plan.quantizer.model_dump(exclude={"norm_slices"}) | {
        "num_planes": 3, "sd_slices": problem["slices"], "rans_lanes": 2048}, "OR")
    codec.build_quantizer(None, list(problem["si"]))
    state = {key: fitted["state"][key] for key in codec.quantizer.coding_model.state_dict()}
    codec.quantizer.coding_model.load_state_dict(state)
    codec.quantizer.side_info_list_used = list(problem["si"])
    codec.quantizer.vector_size = problem["target"].numel()
    codec.quantizer.quantizer_prior_stage_rates = tuple(fitted["stage_rates"][:3])
    codec.quantizer.training_provenance = fitted["training_provenance"]
    codec._quantizer_prior_model = codec.quantizer.coding_model.copy_for_prior_retraining()
    codec.round_name_full = "F"
    before = content_hash(codec.quantizer.coding_model.state_dict())
    record = codec.create_r_record(plan.target_round, plan.target_client)
    record.model_size = problem["target"].numel()
    record.pre_encode_training_seconds = 0.0
    started = perf_counter()
    payload = codec.encode(problem["target"], record)
    record.encode_seconds = perf_counter() - started
    started = perf_counter()
    reconstruction = codec.decode(payload, record)
    record.decode_seconds = perf_counter() - started
    record.total_seconds = record.encode_seconds + record.decode_seconds
    record.final_bytes = payload_size_mib(payload)
    record.update_record_fields(**record_reconstruction_metrics(problem["target"], reconstruction))
    assert before == content_hash(codec.quantizer.coding_model.state_dict()), "F changed starting weights."
    assert record.symbol_error_rate == 0.0
    directory = plan.output_root / "cases/F/target"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "compression_records.csv").unlink(missing_ok=True)
    record.append_record_to_csv(directory)
    result = dict(record=record.to_dict(), frozen_state_hash=before, weights_unchanged=True,
                  quantizer_fits=0, prior_fits=0, baseline_seed=plan.seed)
    with atomic_output(directory / "complete.json") as temporary:
        temporary.write_text(json.dumps(result, indent=2))
    return result


def run(plan: FrozenStartingPlan) -> None:
    """Reconstruct one seed-43 model and publish the frozen target measurements with provenance."""
    root = Path(__file__).resolve().parents[2]
    sources = content_hash({str(path.relative_to(root)): path.read_bytes()
                           for directory in (root / "FL_code", root / "Experiments/Protocol_test")
                           for path in sorted(directory.rglob("*.py"))})
    trace = Trace(load_fl_config(plan.fl_config), dataset_revision=plan.dataset_revision,
                  device=torch.device(plan.trace_device))
    plan.output_root.mkdir(parents=True, exist_ok=True)
    problem = CacheEntry(plan.output_root / "problem", {
        "trace": trace.identity(), "round": plan.target_round,
        "clients": (plan.target_client, plan.baseline_client), "history": 5,
        "si": "raw_server_round_start_v1", "sources": sources,
    }).resolve(lambda: collect_problem(trace, plan))
    print(f"PROBLEM N={problem['target'].numel()} SI={len(problem['si'])}", flush=True)
    torch.cuda.set_device(torch.device(plan.fit_device))
    set_global_seed(plan.seed)
    options = plan.quantizer.model_dump(exclude={"norm_slices"}) | {"sd_slices": problem["slices"]}
    baseline = Oracle_RoundCodec(options, "O")
    baseline.build_quantizer(None, list(problem["si"]))
    runtime = plan.quantizer.numerical_policy.runtime(torch.device(plan.fit_device),
                                                     mixed_precision=plan.quantizer.mixed_precision)
    manifest = dict(plan=plan.model_dump(mode="json"), trace=trace.identity(), sources=sources,
                    numerical_runtime=runtime, target_hash=content_hash(problem["target"]),
                    si_hash=content_hash(tuple(content_hash(value.materialize()) for value in problem["si"])),
                    historical_weights_reproduced=False)
    with atomic_output(plan.output_root / "test_config.json") as temporary:
        temporary.write_text(json.dumps(manifest, indent=2))
    fitted = CacheEntry(plan.output_root / "baseline", {
        "target": content_hash(problem["baseline"]), "si": manifest["si_hash"],
        "slices": [(part.start, part.stop) for part in problem["slices"]],
        "config": plan.quantizer.model_dump(mode="json"), "sources": sources,
        "seed": plan.seed, "runtime": runtime,
    }, TorchCodec()).resolve(lambda: fit_starting_model(baseline, problem["baseline"], plan.seed))
    print(f"BASELINE READY fit_seconds={fitted['seconds']:.1f}", flush=True)
    if baseline.quantizer.side_info_list_used is not None:
        baseline.release_round_resources()
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    result = evaluate_frozen(plan, problem, fitted)
    print(f"F COMPLETE {json.dumps(result)}", flush=True)


def render(output_root: Path, destination: Path) -> None:
    """Render available F stage measurements for the supervisor's disposable live plot processes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    completion = output_root / "cases/F/target/complete.json"
    if not completion.exists():
        (destination / "scope.txt").write_text("Reconstructing starting model; F measurement pending.\n")
        return
    record = json.loads(completion.read_text())["record"]
    figure, axes = plt.subplots()
    axes.plot(np.cumsum(record["coder_stage_rates"]), 20 * np.log10(np.array(record["stage_wmapes"]) / 100), "o-")
    axes.set(xlabel="Frozen full-SI rANS (bits/float)", ylabel="WMAPE (dB)",
             title="F: reconstructed seed-43 starting model, no target adaptation")
    figure.tight_layout()
    figure.savefig(destination / "frozen_rd.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Run a reconstruction plan or serve the detached supervisor's live plotter interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.runner_pid is not None:
        assert args.control_dir is not None
        watch(args.path, args.runner_pid, args.control_dir, plot_module=__spec__.name,
              source_paths=(Path(__file__),))
    elif args.out is not None:
        render(args.path, args.out)
    else:
        run(load_plan(args.path))


if __name__ == "__main__":
    main()
