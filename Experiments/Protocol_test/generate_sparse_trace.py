"""Generate a compact multi-client identity trace without spawning one process per client."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from FL_code.FL_core.dataset import (
    DATASET_NAMES, GROUP_FIELD, create_dataloader, precompute_dataset_to_shared,
)
from FL_code.FL_core.models import BACKBONE_NAMES, initialize_model
from FL_code.FL_core.utils import StateDictManager, set_global_seed, write_fl_config_snapshot
from FL_code.run_fl import FLConfig


def parse_args() -> argparse.Namespace:
    """Parse the dataset-specific trace settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--model", choices=BACKBONE_NAMES, default="resnet18")
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--non-iid", action="store_true",
                        help="Natural per-sample partition where the dataset carries one, else a Dirichlet label skew.")
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--save-from", type=int, default=0,
                        help="First round to persist; earlier rounds still train but are not written.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Train clients sequentially and persist the raw updates needed by sparse replay."""
    args = parse_args()
    assert args.clients > 0 and args.rounds > 0 and args.image_size > 0
    assert 0 <= args.save_from < args.rounds
    assert not args.output.exists(), f"Trace output already exists: {args.output}."
    cfg = FLConfig(
        protocol="identity",
        run_name=f"{args.model}_{args.dataset.lower()}{args.image_size}_{args.clients}c_trace",
        model_name=args.model,
        non_iid_partition=args.non_iid,
        dataset_name=args.dataset,
        image_size=args.image_size,
        num_clients=args.clients,
        num_loader_workers=0,
        data_folder=Path("FL_code/data"),
        rounds=args.rounds,
        local_epochs=1,
        batch_size=args.batch_size,
        lr=args.lr,
        client_eval_every_n_rounds=args.rounds,
        recalibrate_bn=False,
        records_dir=args.output / "unused_records",
        debug_data_folder=args.output.parent,
        debug_save_deltas=args.output.name,
    )
    set_global_seed(cfg.seed)
    # A dataset carrying a per-sample client id must have its pool capped here for both arms, or
    # the IID twin draws from every group and holds far more data per client than the natural split.
    features, labels, groups = precompute_dataset_to_shared(
        cfg.dataset_name, cfg.data_folder, "train", torch.float32, None, cfg.seed, cfg.image_size,
        args.clients if args.dataset.upper() in GROUP_FIELD else None,
    )
    cfg.num_classes = int(labels.max()) + 1
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None
    model = initialize_model(cfg, device)
    manager = StateDictManager(model)
    loaders = [
        create_dataloader(
            features, labels, cfg, device, True, client_id, cfg.num_clients, groups
        )
        for client_id in range(cfg.num_clients)
    ]
    scalers = [torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision) for _ in loaders]
    buffers = {
        client_id: {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
        for client_id in range(cfg.num_clients)
    }

    args.output.mkdir(parents=True)
    write_fl_config_snapshot(cfg, args.output)
    for round_id in range(cfg.rounds):
        server_state = manager.clone_trainable(model.state_dict())
        aggregate = torch.zeros(manager.param_count, dtype=torch.float32)
        total_samples = sum(len(loader.dataset) for loader in loaders)
        for client_id, (loader, scaler) in enumerate(zip(loaders, scalers, strict=True)):
            manager.load_trainable_state(model, server_state)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[client_id][name].to(value.device))
            optimizer = model.configure_optimizer()
            for _ in range(cfg.local_epochs):
                model.train_epoch(loader, optimizer, scaler)
            buffers[client_id] = {
                name: value.detach().cpu().clone() for name, value in model.named_buffers()
            }
            delta = manager.flatten(manager.compute_delta(
                manager.clone_trainable(model.state_dict()), server_state
            )).cpu().contiguous()
            # A replay reads only the rounds around its target, so earlier ones need to advance
            # the model without costing disk -- an 11M-parameter trace is 1.3 GiB per round.
            if round_id >= args.save_from:
                temporary = args.output / f".round_{round_id}_client_{client_id}.tmp"
                torch.save(delta, temporary)
                temporary.replace(args.output / f"round_{round_id}_client_{client_id}.pt")
            aggregate.add_(delta, alpha=len(loader.dataset) / total_samples)
            print(
                f"{cfg.dataset_name} round {round_id}/{cfg.rounds - 1}, "
                f"client {client_id}/{cfg.num_clients - 1}",
                flush=True,
            )
        manager.load_trainable_state(model, server_state)
        manager.apply_delta_inplace(model, manager.unflatten(aggregate))


if __name__ == "__main__":
    main()
