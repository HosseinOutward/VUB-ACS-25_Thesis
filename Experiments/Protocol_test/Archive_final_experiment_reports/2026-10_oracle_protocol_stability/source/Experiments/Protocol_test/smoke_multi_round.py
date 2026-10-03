"""Check one full-size OMR/TMR client on the real trace before launching multi-hour plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource

import torch

from Experiments.Protocol_test.sparse_replay import load_fl_config, load_plan
from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol, simulate_compression
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, set_global_seed


def main() -> None:
    """Replay the required raw prefix and verify one full-default, gated rANS client fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    torch.cuda.set_device(plan.gpu_ids[0])
    cfg = load_fl_config(plan.fl_config)
    manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
    protocol = create_protocol(plan.cases[0].protocol, manager.get_slices())
    token = plan.cases[0].codec_id
    start = plan.cases[0].start_round
    protocol.set_schedule(start, (plan.cases[0].round_codecs[0], "F"))
    output = plan.output_root.parent / "smoke"
    assert not output.exists(), f"Smoke output already exists: {output}"
    trace = Trace(cfg, dataset_revision=plan.dataset_revision, device=torch.device(plan.trace_device))
    for round_id, deltas in trace.stream(0, start):
        print(f"Trace round {round_id}; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB", flush=True)
        for client_id, raw in enumerate(deltas):
            protocol.remember_target(raw, round_id, client_id)
            if round_id < start:
                protocol._recons_history.commit(
                    raw, CompressionRecord(round_id, client_id, "I"), Access.TEMPORAL_TOO,
                )
            else:
                set_global_seed(43)
                reconstructed = simulate_compression(protocol, raw, client_id, round_id, manager, output, {}, (), ())
                assert torch.isfinite(reconstructed).all()
                print(json.dumps({
                    "codec": token, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
                    "gpu_peak_mib": torch.cuda.max_memory_allocated() // 1024 ** 2,
                }), flush=True)
                return


if __name__ == "__main__":
    main()
