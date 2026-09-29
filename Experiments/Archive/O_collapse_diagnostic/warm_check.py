"""Check warm-started training under the current defaults against cold training of the same clients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from Experiments.O_collapse_diagnostic.run_diagnostic import round_problems, train_attempt
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant


def main() -> None:
    """Build the protocol's five-parent cold pool, then warm-train later clients from two parents each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--si-set", required=True, choices=("own5", "own5_same_round", "server_prev", "server_actual"))
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--pool-clients", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--warm-clients", type=int, nargs="+", default=[10, 29])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "warm_results")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    problems, slices = round_problems(args.start_round, [*args.pool_clients, *args.warm_clients])
    # Protocol defaults, including its best-of-attempts cold fit for the pool parents.
    cold = WZcfgQuant(bins_per_plane=2, num_planes=3, norm_slices=slices, training_log_every_epochs=5)
    warm = cold.model_copy(update={"warm_training": True})
    (args.out / "curves").mkdir(parents=True, exist_ok=True)

    def record(stem: str, client_id: int, phase: str, seed: int, result: dict[str, object], curve: list) -> None:
        pd.DataFrame(curve).to_csv(args.out / "curves" / f"{stem}.csv", index=False)
        row = {"client_id": client_id, "si_set": args.si_set, "phase": phase, "seed": seed, **result}
        (args.out / f"{stem}.json").write_text(json.dumps(row))
        print(json.dumps(row), flush=True)

    pool = []
    for client_id in args.pool_clients:
        target, si_sets = problems[client_id]
        result, curve, joint_state = train_attempt(target, si_sets[args.si_set], cold, seed=0)
        record(f"pool_client{client_id}_{args.si_set}", client_id, "cold_pool", 0, result, curve)
        pool.append(joint_state)

    # The protocol's balanced rotating window: each warm call takes the next two slots.
    for call, client_id in enumerate(args.warm_clients):
        slots = [(2 * call + offset) % len(pool) for offset in range(2)]
        target, si_sets = problems[client_id]
        for seed in args.seeds:
            result, curve, _ = train_attempt(
                target, si_sets[args.si_set], warm, seed, [pool[slot] for slot in slots])
            record(f"warm_client{client_id}_{args.si_set}_seed{seed}", client_id, "warm", seed,
                   result | {"parent_slots": slots}, curve)


if __name__ == "__main__":
    main()
