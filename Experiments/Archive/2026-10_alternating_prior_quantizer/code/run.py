"""Alternate quantizer/prior fits on real five-client FL deltas and fixed decoder history."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep
from typing import Any, Literal

import numpy as np
import torch

from Experiments.Protocol_test.detached_replay import _meminfo_mib
from Experiments.Protocol_test.sparse_replay import load_fl_config
from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.explicit_cache import CacheEntry, atomic_output, content_hash
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import PreprocessMetadata, StateDictManager, set_global_seed
from FL_code.cancer_protocol.history import CompressedReconstructionHistory
from FL_code.cancer_protocol.prior_code import INFERENCE_BATCH, PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer, SideInfoMatrix, WZcfgQuant, WZQuantizerCancer, wz_model_training_loop

from .plan import Plan, load_plan


_REPO_ROOT = Path(__file__).resolve().parents[4]


def data_entry(plan: Plan) -> CacheEntry[tuple[CompressedReconstructionHistory, tuple[torch.Tensor, ...]]]:
    """Declare the compressed-history/target cache with its complete trajectory and storage identity."""
    cfg = load_fl_config(plan.fl_config)
    assert cfg.num_clients == len(plan.cases), "The FL trace must train exactly five clients."
    trace = Trace(cfg, dataset_revision=plan.dataset_revision, device=torch.device(plan.trace_device))
    manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
    history = CompressedReconstructionHistory(2, manager.get_slices())
    match = {
        "trace": trace.identity(), "history_rounds": [plan.target_round - 2, plan.target_round - 1],
        "target_round": plan.target_round, "clients": plan.cases, "history": history.config_snapshot(),
        "source": content_hash((_REPO_ROOT / "FL_code/cancer_protocol/history.py").read_bytes()),
    }
    return CacheEntry(_REPO_ROOT / "_/data/alternating_training", match)


def prepare_data(plan: Plan) -> None:
    """Cache two preceding rounds in the existing compressed ledger and retain raw target deltas."""
    def compute() -> tuple[CompressedReconstructionHistory, tuple[torch.Tensor, ...]]:
        cfg = load_fl_config(plan.fl_config)
        trace = Trace(cfg, dataset_revision=plan.dataset_revision, device=torch.device(plan.trace_device))
        manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
        history = CompressedReconstructionHistory(2, manager.get_slices())
        targets: tuple[torch.Tensor, ...] = ()
        for round_id, deltas in trace.stream(plan.target_round - 2, plan.target_round):
            print(f"Trace round {round_id}: {len(deltas)} clients", flush=True)
            if round_id == plan.target_round:
                targets = deltas
                continue
            for client_id, delta in enumerate(deltas):
                history.commit(delta, CompressionRecord(round_id, client_id, "raw_seed"), Access.SERVER_ONLY)
        history.frozen = True
        return history, targets

    # The expensive computation and its semantic entry are explicit at this call site.
    entry = data_entry(plan)
    entry.resolve(compute)
    print(f"Prepared compressed history and targets: {entry.path}", flush=True)


def wait_for_resources(plan: Plan) -> None:
    """Queue the single worker until host RAM and both requested GPUs have enough free capacity."""
    gpu_ids = {plan.gpu_id, int(plan.trace_device.split(":")[1])}
    while True:
        rows = subprocess.run(
            ("nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"),
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        status = {int(fields[0]): (int(fields[1]), int(fields[2]))
                  for row in rows if (fields := row.split(","))}
        available = _meminfo_mib()["MemAvailable"]
        ready = available >= plan.min_available_ram_gib * 1024 and all(
            status[gpu][0] >= plan.min_free_gpu_gib * 1024 and status[gpu][1] <= 10 for gpu in gpu_ids)
        print(f"Resource gate: ready={ready}, RAM available={available} MiB, GPUs={status}", flush=True)
        if ready:
            return
        sleep(60)


def fine_tune(
    quantizer: WZQuantizerCancer, x: torch.Tensor, side_info: SideInfoMatrix,
    phase: Literal["quantizer", "joint"],
) -> float:
    """Continue the existing RD objective, optionally freezing every conditional-prior parameter."""
    model = quantizer.coding_model
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if phase == "quantizer" and name.startswith("conditional")}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in frozen)
    try:
        loss = wz_model_training_loop(
            quantizer.compute_loss, model, (parameter for parameter in model.parameters() if parameter.requires_grad),
            (x,), side_info, quantizer.c_cfg.warm_retraining_config(),
            batch_size=quantizer.train_batch_size, label=f"Alternating {phase}",
            metrics_fn=quantizer.training_metrics, wmspe_denom=x.float().square().mean().item() / 2 + 1e-8,
            perturb_generator=torch.Generator(device="cuda").manual_seed(torch.initial_seed()),
        )
    finally:
        model.requires_grad_(True)
    assert np.isfinite(loss), f"Non-finite loss after {phase} fine-tuning."
    assert all(torch.equal(dict(model.named_parameters())[name].cpu(), value.cpu()) for name, value in frozen.items()), (
        "Frozen prior changed during quantizer fine-tuning.")
    return loss


def measure(
    quantizer: WZQuantizerCancer, target: torch.Tensor, x: torch.Tensor, side_info: SideInfoMatrix,
    metadata: PreprocessMetadata, client_id: int, phase_index: int, phase: str, seconds: float,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Measure full-vector hard-symbol rates and raw-unit stage distortions with fixed decoder SI."""
    bins, codes = quantizer.encode_prepared(x, INFERENCE_BATCH)
    prior = quantizer.prior_calculator.compute_prior_from_network(quantizer.coding_model, bins, side_info)
    rates = PriorCalculator.compute_plane_rates(prior, bins, quantizer.num_planes)
    del prior
    marginal = PriorCalculator.compute_marginal_prior(bins, quantizer.bins_per_plane, quantizer.num_planes)
    marginal_rates = PriorCalculator.compute_plane_rates(marginal, bins, quantizer.num_planes)
    del marginal
    reconstructions = quantizer.decode_stages_prepared(bins, side_info, INFERENCE_BATCH)
    records: list[dict[str, Any]] = []
    for stage, normalized in enumerate(reconstructions, 1):
        reconstruction = quantizer.postprocess(normalized, metadata)
        wmape = float((reconstruction - target).abs().mean() / target.abs().mean() * 100)
        mse = float((reconstruction - target).square().mean())
        assert np.isfinite([wmape, mse, *rates[:stage]]).all(), (
            f"Client {client_id}, phase {phase_index} {phase}, stage {stage}: non-finite RD measurement.")
        symbols = int(bins[stage - 1].unique().numel())
        records.append({
            "client_id": client_id, "phase_index": phase_index, "phase": phase, "stage": stage,
            "prior_bits": sum(rates[:stage]), "marginal_bits": sum(marginal_rates[:stage]),
            "wmape_percent": wmape, "mse": mse, "symbols": symbols, "seconds": seconds,
            "flag": "dead" if symbols < 2 else "blown" if wmape >= 100 else "",
        })
    print(f"Client {client_id}, phase {phase_index} {phase}: "
          f"rate={[round(row['prior_bits'], 5) for row in records]}, "
          f"WMAPE={[round(row['wmape_percent'], 5) for row in records]}", flush=True)
    return records, bins, codes


def run(plan_path: Path) -> None:
    """Run one continuation trajectory per client, publishing every phase before proceeding."""
    plan = load_plan(plan_path)
    cfg = load_fl_config(plan.fl_config)
    manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
    c_cfg = WZcfgQuant(norm_slices=manager.get_slices(), **plan.quantizer)
    identity = {
        "plan": plan.model_dump(mode="json"), "data": data_entry(plan).identity,
        "quantizer": c_cfg.model_dump(mode="json", exclude={"norm_slices"}),
        "training_source": content_hash(tuple(path.read_bytes() for path in (
            Path(__file__), _REPO_ROOT / "FL_code/cancer_protocol/wz_quantizer.py",
            _REPO_ROOT / "FL_code/cancer_protocol/prior_code.py", _REPO_ROOT / "FL_code/cancer_protocol/brent_wz_models.py",
        ))),
    }
    plan.output_root.mkdir(parents=True, exist_ok=True)
    config_path = plan.output_root / "test_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text())["identity"] == identity, f"Identity mismatch: {config_path}."
    else:
        with atomic_output(config_path) as temporary:
            temporary.write_text(json.dumps({"identity": identity, "diagnostic": "single trained target, fixed historical full SI"}, indent=2))
    wait_for_resources(plan)
    if not data_entry(plan).path.exists():
        # FL's dataset and optimizer allocations die with this disposable preparation process.
        subprocess.run((sys.executable, "-u", "-m", __spec__.name, str(plan_path), "--prepare-data"), check=True)
    history, targets = data_entry(plan).load()
    side_information = [entry.value for entries in history.view(Access.SERVER_ONLY, 0).values() for entry in entries]
    assert len(side_information) == 10
    for client_id in plan.cases:
        case_dir = plan.output_root / "cases/alternating" / f"client{client_id}"
        if (case_dir / "complete.json").exists():
            continue
        torch.cuda.set_device(plan.gpu_id)
        set_global_seed(plan.seed + plan.target_round * len(plan.cases) + client_id)
        quantizer = DedupedWZQuantizerCancer(c_cfg, len(side_information))
        quantizer.set_side_information(side_information)
        target = targets[client_id]
        x, metadata = quantizer.preprocess_x(target)
        si = quantizer.side_info_tensor(metadata)
        case_dir.mkdir(parents=True, exist_ok=True)
        all_records: list[dict[str, Any]] = []
        bins, codes = None, None
        for phase_index, phase in enumerate(plan.phases):
            checkpoint = case_dir / f"phase{phase_index}.pt"
            phase_records = case_dir / f"phase{phase_index}.json"
            if phase_records.exists():
                quantizer.coding_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                all_records.extend(json.loads(phase_records.read_text()))
                bins, codes = None, None
                continue
            # Phase-local seeds make an interrupted continuation match the uninterrupted run.
            set_global_seed(plan.seed + plan.target_round * len(plan.cases) + client_id + phase_index * 1_000)
            started = perf_counter()
            if phase == "initial_joint":
                quantizer._fit_prepared(x, si)
            elif phase == "prior":
                if bins is None:
                    bins, codes = quantizer.encode_prepared(x, INFERENCE_BATCH)
                fixed = {name: value.detach().clone() for name, value in quantizer.coding_model.state_dict().items()
                         if not name.startswith("conditional")}
                PriorCalculator.fit_prior_model(
                    bins, codes, si, c_cfg, quantizer.coding_model, (None,), c_cfg.prior_train_repeats,
                    batch_size=c_cfg.train_batch_size,
                )
                assert all(torch.equal(quantizer.coding_model.state_dict()[name], value) for name, value in fixed.items()), (
                    f"Client {client_id}: quantizer changed during prior fitting.")
            else:
                fine_tune(quantizer, x, si, phase)
            records, bins, codes = measure(
                quantizer, target, x, si, metadata, client_id, phase_index, phase, perf_counter() - started)
            with atomic_output(checkpoint) as temporary:
                torch.save(quantizer.coding_model.state_dict(), temporary)
            with atomic_output(phase_records) as temporary:
                temporary.write_text(json.dumps(records, indent=2, allow_nan=False))
            all_records.extend(records)
            with atomic_output(case_dir / "compression_records.csv") as temporary:
                with temporary.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
                    writer.writeheader()
                    writer.writerows(all_records)
        with atomic_output(case_dir / "complete.json") as temporary:
            # Regenerate the CSV after recovery even if every saved phase was already measured.
            with atomic_output(case_dir / "compression_records.csv") as csv_temporary:
                with csv_temporary.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
                    writer.writeheader()
                    writer.writerows(all_records)
            temporary.write_text(json.dumps({"identity": identity, "client_id": client_id}, indent=2))
        del quantizer, x, si, bins, codes
        gc.collect()
        torch.cuda.empty_cache()
    with atomic_output(plan.output_root / "complete.json") as temporary:
        temporary.write_text(json.dumps({"identity": identity}, indent=2))


def main() -> None:
    """Run the diagnostic or its isolated, explicitly cached FL data preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--prepare-data", action="store_true")
    args = parser.parse_args()
    if args.prepare_data:
        prepare_data(load_plan(args.plan.resolve()))
    else:
        run(args.plan.resolve())


if __name__ == "__main__":
    main()
