from __future__ import annotations

import argparse
import os
from datetime import timedelta
from typing import Any

from FL_code.FL_core.dataset import _force_class_coverage
from FL_code.FL_core.utils import _assert_class_coverage_before_spawn, _prepare_records_dir, assert_debug_fl_config_matches, write_fl_config_snapshot
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import start_processes
from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator

from FL_code.FL_core.codec import parse_and_validate_protocol


class FLConfig(BaseModel):
    """Federated learning configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    protocol: str
    run_name: str | None = None

    model_name: str = "resnet18"  # resnet18, resnet50, resnet56
    dataset_name: str = "SVHN"    # SVHN, CIFAR10, SYNTHETIC
    num_classes: int | None = None

    debug_mode: bool = False

    num_clients: int = 5
    num_loader_workers: int = 3
    data_folder: Path = Path("data")
    rounds: int = 60
    local_epochs: int = 5
    batch_size: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 43

    recalibrate_bn: bool = True
    bn_recalib_batches: int = 100
    client_eval_every_n_rounds: int = 1 # client evaluation on test data frequency in rounds

    # Memory and performance optimizations
    channels_last: bool = True
    fused_optimizer: bool = True
    mixed_precision: bool = True  # AMP

    training_progress_bar: bool = False
    records_dir: Path = Path("records")
    dataset_fraction: float | None = None  # Fraction of dataset to use or None for full

    master_addr: str = "localhost"
    master_port: str = "29500"

    debug_save_train_data: bool = False
    debug_data_folder: Path = Path('data/_debug_saved_checckpoints')
    debug_save_deltas: str = 'delta_vec_data'
    debug_load_from_saved_data: bool = False
    debug_continue_from_saved_data: bool = False
    debug_continue_then_save: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent changing the configured protocol after configuration creation."""
        assert not (
            name == "protocol" and "protocol" in self.__dict__ and self.__dict__["protocol"] != value
        ), "FLConfig.protocol is immutable after construction."
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def validate_explicit_configuration(self) -> FLConfig:
        """Reject contradictory options that would otherwise change behavior implicitly."""
        parse_and_validate_protocol(self.protocol)
        if self.debug_mode:
            object.__setattr__(self, "num_clients", 3)
            object.__setattr__(self, "local_epochs", 1)
            if self.dataset_fraction is None:
                object.__setattr__(self, "dataset_fraction", 0.1)
        assert not (
            self.debug_save_train_data and self.debug_load_from_saved_data
        ), "debug_save_train_data and debug_load_from_saved_data cannot both be enabled."
        assert not (
            self.debug_continue_from_saved_data or self.debug_continue_then_save
        ), ("debug_continue_from_saved_data and debug_continue_then_save were removed because they "
            "change behavior when saved debug files are missing.")
        assert self.dataset_fraction is None or 0.0 < self.dataset_fraction < 1.0, (
            "dataset_fraction must be None or a value in (0.0, 1.0).")
        assert self.client_eval_every_n_rounds > 0, "client_eval_every_n_rounds must be positive."
        return self


def _worker(
    rank: int,
    world_size: int,
    cfg: FLConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> None:
    """Distributed worker with access to shared data tensors."""
    import sys
    import traceback as tb

    role = "Server" if rank == 0 else f"Client {rank - 1}"
    if cfg.debug_mode:
        assert cfg.num_clients == 3 and cfg.local_epochs == 1 and cfg.dataset_fraction is not None

    # Set up process group
    dist.init_process_group(backend="gloo", init_method='env://',
            timeout=timedelta(minutes=75), world_size=world_size, rank=rank)

    try:
        if rank == 0:
            from FL_code.FL_core.server import run_federated_server
            run_federated_server(cfg, rank, world_size, X_test, y_test)
        else:
            from FL_code.FL_core.client import run_federated_client
            run_federated_client(cfg, rank, world_size, X_train, y_train, X_test, y_test)
    except Exception:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[{role}] EXCEPTION OCCURRED", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        # Print formatted traceback with role prefix
        exc_type, exc_value, exc_tb = sys.exc_info()
        formatted_tb = tb.format_exception(exc_type, exc_value, exc_tb)

        for line in formatted_tb:
            for subline in line.rstrip().split('\n'):
                print(f"[{role}] {subline}", file=sys.stderr)

        print(f"{'='*70}\n", file=sys.stderr)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=str, required=True)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--model", type=str, default=FLConfig.model_fields["model_name"].default,
                    choices=['resnet18', 'resnet50', 'resnet56'])
    ap.add_argument("--dataset", type=str, default=FLConfig.model_fields["dataset_name"].default,
                    choices=['SVHN', 'CIFAR10', 'SYNTHETIC'])
    ap.add_argument("--master-port", type=str, default=FLConfig.model_fields["master_port"].default)
    ap.add_argument("--debug-mode", "--debug_mode", action="store_true", default=False)
    ap.add_argument("--rounds", type=int, default=FLConfig.model_fields["rounds"].default)
    ap.add_argument("--num-clients", type=int, default=FLConfig.model_fields["num_clients"].default)
    ap.add_argument("--local-epochs", type=int, default=FLConfig.model_fields["local_epochs"].default)
    ap.add_argument("--batch-size", type=int, default=FLConfig.model_fields["batch_size"].default)
    ap.add_argument("--dataset-fraction", type=float, default=FLConfig.model_fields["dataset_fraction"].default)
    ap.add_argument("--records-dir", type=Path, default=FLConfig.model_fields["records_dir"].default)
    ap.add_argument("--data-folder", type=Path, default=FLConfig.model_fields["data_folder"].default)
    ap.add_argument("--debug-save-train-data", action="store_true", default=False)
    ap.add_argument("--debug-load-from-saved-data", action="store_true", default=False)
    ap.add_argument("--debug-data-folder", type=Path, default=FLConfig.model_fields["debug_data_folder"].default)
    ap.add_argument("--debug-save-deltas", type=str, default=FLConfig.model_fields["debug_save_deltas"].default)
    args = ap.parse_args()

    cfg = FLConfig(
        protocol=args.protocol,
        run_name=args.run_name,
        model_name=args.model,
        dataset_name=args.dataset,
        master_port=args.master_port,
        debug_mode=args.debug_mode,
        rounds=args.rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        dataset_fraction=args.dataset_fraction,
        records_dir=args.records_dir,
        data_folder=args.data_folder,
        debug_save_train_data=args.debug_save_train_data,
        debug_load_from_saved_data=args.debug_load_from_saved_data,
        debug_data_folder=args.debug_data_folder,
        debug_save_deltas=args.debug_save_deltas,
    )

    if cfg.debug_mode:
        print('**************************************************')
        print('******************  DEBUG MODE  ******************')

    _prepare_records_dir(cfg)

    # Set environment variables for address and port
    os.environ['MASTER_ADDR'] = cfg.master_addr
    os.environ['MASTER_PORT'] = cfg.master_port

    # Precompute dataset and store in shared memory
    if cfg.dataset_fraction is not None:
        print(f"[Debug] Using {cfg.dataset_fraction*100:.1f}% of dataset for quick testing.")

    from FL_code.FL_core.dataset import precompute_dataset_to_shared
    X_train, y_train = precompute_dataset_to_shared(cfg.dataset_name, cfg.data_folder,
                                                    "train", torch.float32, cfg.dataset_fraction, cfg.seed)
    X_test, y_test = precompute_dataset_to_shared(cfg.dataset_name, cfg.data_folder,
                                                  "test", torch.float32, cfg.dataset_fraction, cfg.seed)

    # Union of both splits so a fraction dropping a class from one split is still detected.
    cfg.num_classes = int(torch.cat([y_train, y_test]).unique().numel())
    if cfg.dataset_fraction is not None:
        _force_class_coverage(y_train, y_test, cfg)
    _assert_class_coverage_before_spawn(y_train, y_test, cfg)

    debug_delta_dir = cfg.debug_data_folder / cfg.debug_save_deltas
    if cfg.debug_save_train_data:
        assert not debug_delta_dir.exists() or not any(debug_delta_dir.iterdir()), (
            f"debug_save_train_data refuses to write into non-empty {debug_delta_dir}; "
            f"remove stale saved deltas first."
        )
        write_fl_config_snapshot(cfg, debug_delta_dir)
    if cfg.debug_load_from_saved_data:
        assert_debug_fl_config_matches(cfg, debug_delta_dir)

    print(f'[MAIN] {cfg.protocol}')

    # Start distributed processes with spawn semantics.
    start_processes(
        _worker,
        args=(cfg.num_clients + 1, cfg, X_train, y_train, X_test, y_test),
        nprocs=cfg.num_clients + 1,
        join=True,
        start_method="spawn",
    )
