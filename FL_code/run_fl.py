from __future__ import annotations

import argparse
import os
from datetime import timedelta
from typing import Any

from FL_code.utils import _prepare_records_dir, assert_debug_fl_config_matches, write_fl_config_snapshot
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator

from FL_code.codec import parse_and_validate_codec_name


class FLConfig(BaseModel):
    """Federated learning configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    codec: str
    run_name: str | None = None

    model_name: str = "resnet18"  # resnet18, resnet50, resnet56
    dataset_name: str = "SVHN"    # SVHN, CIFAR10
    num_classes: int | None = None

    debug_mode: bool = False

    num_clients: int = 5
    num_loader_workers: int = 3
    data_folder: Path = Path("data")
    rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 500
    single_batch_accum_grad_steps: int = 1 # chop down the above batch into smaller pieces and combine grads.
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 43

    recalibrate_bn: bool = True
    bn_recalib_batches: int = 100

    # Memory and performance optimizations
    channels_last: bool = True
    cudnn_benchmark: bool = True
    fused_optimizer: bool = True
    tf32: bool = True  # TF32 on Ampere+ GPUs
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
        """Prevent changing the codec name after configuration creation."""
        assert not (
            name == "codec" and "codec" in self.__dict__ and self.__dict__["codec"] != value
        ), "FLConfig.codec is immutable after construction."
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def validate_explicit_configuration(self) -> FLConfig:
        """Reject contradictory options that would otherwise change behavior implicitly."""
        parse_and_validate_codec_name(self.codec)
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

    # Set up process group
    dist.init_process_group(backend="nccl", init_method='env://',
            timeout=timedelta(minutes=75), world_size=world_size, rank=rank)

    role = "Server" if rank == 0 else f"Client {rank - 1}"

    try:
        if rank == 0:
            from FL_code.server import run_federated_server
            run_federated_server(cfg, rank, world_size, X_test, y_test)
        else:
            from FL_code.client import run_federated_client
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
    ap.add_argument("--codec", type=str, required=True)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--model", type=str, default=FLConfig.model_fields["model_name"].default,
                    choices=['resnet18', 'resnet50', 'resnet56'])
    ap.add_argument("--dataset", type=str, default=FLConfig.model_fields["dataset_name"].default,
                    choices=['SVHN', 'CIFAR10'])
    ap.add_argument("--master-port", type=str, default=FLConfig.model_fields["master_port"].default)
    ap.add_argument("--debug-mode", "--debug_mode", action="store_true", default=False)
    args = ap.parse_args()

    cfg = FLConfig(
        codec=args.codec,
        run_name=args.run_name,
        model_name=args.model,
        dataset_name=args.dataset,
        master_port=args.master_port,
        debug_mode=args.debug_mode,
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

    from FL_code.dataset import precompute_dataset_to_shared
    X_train, y_train = precompute_dataset_to_shared(cfg.dataset_name, cfg.data_folder,
                                                    "train", torch.float32, cfg.dataset_fraction)
    X_test, y_test = precompute_dataset_to_shared(cfg.dataset_name, cfg.data_folder,
                                                  "test", torch.float32, cfg.dataset_fraction)

    cfg.num_classes = torch.unique(y_train).numel()

    debug_delta_dir = cfg.debug_data_folder / cfg.debug_save_deltas
    if cfg.debug_save_train_data:
        write_fl_config_snapshot(cfg, debug_delta_dir)
    if cfg.debug_load_from_saved_data:
        assert_debug_fl_config_matches(cfg, debug_delta_dir)

    print(f'[MAIN] {cfg.codec}')

    # Spawn distributed processes
    mp.spawn(
        _worker,
        args=(cfg.num_clients + 1, cfg, X_train, y_train, X_test, y_test),
        nprocs=cfg.num_clients + 1,
        join=True
    )
