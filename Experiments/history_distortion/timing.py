"""Time all-client history admissions and profile the quantizer's normalization estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import pandas as pd
import torch

from Experiments.history_distortion.run import RoundNormalizedHistory, load_fl_config
from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, normalization_params
from FL_code.cancer_protocol import history as history_module


def main() -> None:
    """Record matched all-client timings with separate fitting, transform, packing, and retrieval costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fl-config", type=Path, default=Path("Experiments/Protocol_test/plans/svhn_30c_fl_config_seed44.json"))
    parser.add_argument("--reference", type=Path, default=Path("Experiments/history_distortion/results/svhn_seed44_round_shared_20261005"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), f"Use a fresh output directory: {args.output}"
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    reference = json.loads((args.reference / "metadata.json").read_text())
    cfg = load_fl_config(args.fl_config)
    trace = Trace(cfg, dataset_revision=reference["trace"]["dataset_revision"], device=torch.device(args.device))
    assert trace.identity() == reference["trace"], "Timing and distortion experiments must use the same trace."
    slices = StateDictManager(initialize_model(cfg, torch.device("cpu"))).get_slices()
    histories = {
        "current_5round": history_module.CompressedReconstructionHistory(1, slices),
        "round_shared": RoundNormalizedHistory(cfg.num_clients, slices),
    }
    normalization_seconds = 0.0
    normalize = history_module._normalize_history

    def timed_normalize(
        value: torch.Tensor, ranges: tuple[tuple[int, int], ...], supplied_factors: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal normalization_seconds
        started = perf_counter()
        result = normalize(value, ranges, supplied_factors)
        normalization_seconds = perf_counter() - started
        return result

    rows: list[dict[str, Any]] = []
    (args.output / "metadata.json").write_text(json.dumps({
        "trace": trace.identity(), "first_round": reference["first_round"], "last_round": reference["last_round"],
        "clients": list(range(cfg.num_clients)), "torch_cpu_threads": torch.get_num_threads(),
        "timing": "single-run wall latency, excludes FL training and profiler; packing phase includes ledger overhead",
        "current_normalization_phase": "includes parallel factor fitting on refresh; separate fit latency is unavailable",
        "fit_profile": "CPU operator profile of first-client layer fitting, separate from measured admission latency",
    }, indent=2) + "\n")
    with patch.object(history_module, "_normalize_history", timed_normalize):
        for round_id, deltas in trace.stream(reference["first_round"], reference["last_round"]):
            if round_id == reference["first_round"]:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    for layer in slices:
                        normalization_params(deltas[0][layer])
                operations = pd.DataFrame([{
                    "operator": event.key, "count": event.count,
                    "self_cpu_ms": event.self_cpu_time_total / 1000, "total_cpu_ms": event.cpu_time_total / 1000,
                } for event in profile.key_averages()]).sort_values("self_cpu_ms", ascending=False)
                operations["self_cpu_pct"] = 100 * operations.self_cpu_ms / operations.self_cpu_ms.sum()
                operations.to_csv(args.output / "fit_profile.csv", index=False)
            for client_id, source in enumerate(deltas):
                for method, history in histories.items():
                    normalization_seconds = 0.0
                    started = perf_counter()
                    history.commit(source, CompressionRecord(round_id, client_id, "timing"), Access.TEMPORAL_TOO)
                    admission_seconds = perf_counter() - started
                    shared = isinstance(history, RoundNormalizedHistory)
                    fit_seconds = history.fit_seconds if shared else 0.0
                    refreshed = fit_seconds > 0 if shared else (round_id - reference["first_round"]) % history_module._NORMALIZATION_REFRESH_ROUNDS == 0
                    payload = history.view(Access.TEMPORAL_TOO, client_id)[client_id][-1].value
                    started = perf_counter()
                    payload.materialize()
                    rows.append({
                        "round": round_id, "client": client_id, "method": method, "refreshed": refreshed,
                        "admission_seconds": admission_seconds, "fit_seconds": fit_seconds if shared or not refreshed else None,
                        "normalization_phase_seconds": normalization_seconds,
                        "packing_and_ledger_seconds": admission_seconds - normalization_seconds - fit_seconds,
                        "retrieval_seconds": perf_counter() - started,
                    })
            pd.DataFrame(rows).to_csv(args.output / "admissions.csv", index=False)
            print(f"Timed all {cfg.num_clients} clients in round {round_id}", flush=True)
    frame = pd.DataFrame(rows)
    frame.groupby(["method", "round"]).sum(numeric_only=True, min_count=1).drop(columns="client").to_csv(args.output / "rounds.csv")
    print(frame.groupby(["method", "refreshed"])[["admission_seconds", "normalization_phase_seconds", "packing_and_ledger_seconds", "retrieval_seconds"]].mean().to_string())


if __name__ == "__main__":
    main()
