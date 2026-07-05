"""Simple script to run FL workflow and trigger debug dumps when prior > marginal."""
from pathlib import Path

import numpy as np
import torch
from FL_code.cancer_protocol import CancerCodec, CancerConfig


def run_workflow():
    """Run FL workflow to trigger prior > marginal cases."""
    print("="*70)
    print("RUNNING FL WORKFLOW TO GENERATE DEBUG DUMPS")
    print("="*70)

    num_clients = 2
    c_cfg = CancerConfig()
    c_cfg.training_progress_bar = True

    # Use M for first round since no SI available yet
    c_cfg.warmup_phase = (('M', 8, 3), ('T', 8, 3), ('R', 4, 3), ('R', 4, 3), ('R', 4, 3))

    codec = CancerCodec(c_cfg, quantizer_kwargs={'norm_slices': [slice(0, None)]})

    n = 300_000
    base_gradient = torch.randn(n) * 0.1

    print(f"\nnum_clients={num_clients}, vector_size={n}")
    print(f"warmup_phase: {c_cfg.warmup_phase}")
    print(f"routine_phase: {c_cfg.routine_phase}")
    print("\nRunning rounds...")

    for round_id in range(12):
        for client_id in range(num_clients):
            delta = base_gradient + torch.randn(n) * 0.05

            record = codec.create_record(round_id, client_id)
            record.model_size = n

            payload = codec.encode(delta, record)
            _ = codec.decode(payload, record)

            diff = record.prior_rate - record.marginal_rate
            status = "✓" if diff < 0.1 else "❌"

            print(f"R{round_id:2d}C{client_id} [{record.phase[0]}|{record.round_type:2s}|{record.bins_per_plane}bpp] "
                  f"prior={record.prior_rate:.3f} marg={record.marginal_rate:.3f} diff={diff:+.3f} {status}")

        base_gradient = base_gradient + torch.randn(n) * 0.01

    # Check for dumps
    dump_dir = Path(__file__).parent / "debug_dumps"
    dumps = list(dump_dir.glob("dump_*.pt")) if dump_dir.exists() else []
    print(f"\n{'='*70}")
    print(f"Generated {len(dumps)} debug dump(s) in {dump_dir}")
    if dumps:
        print("Run: python inspect_debug_dump.py --list")
        print("Or:  python inspect_debug_dump.py -i")


if __name__ == "__main__":
    run_workflow()
