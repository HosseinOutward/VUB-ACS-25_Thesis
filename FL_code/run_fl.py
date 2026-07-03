from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator


class FLConfig(BaseModel):
    """Federated learning configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Codec to use: identity, basic, ?_split_codec (2,3,...), debug_CancerWithBoundCalc
    # non_wz_learned, cancer (_w_outlier, _basic_norm, _binary), temporal_only
    codec: str = "cancer_w_outlier"
    run_name: str | None = None

    model_name: str = "resnet18"  # resnet18, resnet50, resnet56
    dataset_name: str = "SVHN"    # SVHN, CIFAR10

    debug_mode: bool = False
    num_clients: int = 5
    num_loader_workers: int = 2
    num_classes: int = 10
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
    compile_mode: str | bool = False  # linux only; False for no compiling

    training_progress_bar: bool = False
    records_dir: Path | None = Path("records")  # Directory to save records, None to disable
    dataset_fraction: float | None = None  # Fraction of dataset to use or None for full

    backend: str = "gloo" # "gloo" only, "nccl" for GPU/Linux and doesn't support cpu
    master_addr: str = "localhost"
    master_port: str = "29500"

    debug_save_train_data: bool = False
    debug_data_folder: Path = Path('experiments/debuging/debugging_data')
    debug_save_deltas: str = 'delta_vec_data'
    debug_load_from_saved_data: bool = False
    debug_continue_from_saved_data: bool = False
    debug_continue_then_save: bool = False

    @model_validator(mode="after")
    def normalize_legacy_codec_options(self) -> FLConfig:
        """Normalize legacy compound codec names to the codec selector."""
        codec_name = self.codec.lower()
        if self.run_name is None:
            self.run_name = codec_name

        if codec_name.endswith("_split_codec") and codec_name.split("_", 1)[0].isdigit():
            self.codec = "split"
            if self.debug_mode:
                self.num_clients = 3
                self.local_epochs = 1
                if self.dataset_fraction is None:
                    self.dataset_fraction = 0.1
            return self

        legacy_markers = (
            "_basic_norm", "_w_outlier", "_binary", "_mid_rate", "_mid",
            "_saved_state", "_load_state", "_continue_then_save", "_continue",
        )

        if "_saved_state" in codec_name:
            self.debug_save_train_data = True
        if "_load_state" in codec_name:
            self.debug_load_from_saved_data = True
        if "_continue_then_save" in codec_name:
            self.debug_continue_from_saved_data = True
            self.debug_continue_then_save = True
        elif "_continue" in codec_name:
            self.debug_continue_from_saved_data = True

        for marker in legacy_markers:
            codec_name = codec_name.replace(marker, "")

        if codec_name == "debug_cancerwithboundcalc":
            self.codec = "debug_CancerWithBoundCalc"
        else:
            self.codec = codec_name

        if self.debug_mode:
            self.num_clients = 3
            self.local_epochs = 1
            if self.dataset_fraction is None:
                self.dataset_fraction = 0.1

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
    dist.init_process_group(backend=cfg.backend, init_method='env://',
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


def _prepare_records_dir(cfg: FLConfig) -> None:
    """Create the numbered run directory and save configuration snapshots."""
    if cfg.records_dir is None:
        return

    if cfg.run_name is not None:
        run_name = cfg.run_name
    else:
        run_name = cfg.codec.lower()

    records_root = Path(cfg.records_dir)
    records_root.mkdir(exist_ok=True, parents=True)
    existing_names = {path.name for path in records_root.iterdir()}
    run_num = 1
    while any(name == f"run{run_num}" or name.startswith(f"run{run_num}_") for name in existing_names):
        run_num += 1

    run_dir = records_root / f"run{run_num}_{run_name}"
    run_dir.mkdir()
    cfg.records_dir = run_dir

    with (run_dir / "fl_config.json").open('w') as f:
        json.dump(cfg.model_dump(mode="json"), f, indent=2)

    if cfg.codec.lower() in {
        "cancer",
        "temporal_only",
        "retrain_only",
        "non_wz_learned_worker",
        "non_wz_learned_server",
        "debug_cancerwithboundcalc",
    }:
        assert cfg.codec.lower() in 'cancer'
        from FL_code.cancer_protocol import build_cancer_config_for_fl
        with (run_dir / "codec_config.json").open('w') as f:
            json.dump(build_cancer_config_for_fl(cfg).model_dump(mode="json"), f, indent=2)


if __name__ == "__main__":
    default_cfg = FLConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", type=str, default=default_cfg.run_name or default_cfg.codec)
    ap.add_argument("--model", type=str, default=default_cfg.model_name,
                    choices=['resnet18', 'resnet50', 'resnet56'])
    ap.add_argument("--dataset", type=str, default=default_cfg.dataset_name,
                    choices=['SVHN', 'CIFAR10'])
    ap.add_argument("--master-port", type=str, default=default_cfg.master_port)
    ap.add_argument("--debug-mode", "--debug_mode", action="store_true", default=default_cfg.debug_mode)
    args = ap.parse_args()

    cfg = FLConfig(
        codec=args.codec,
        model_name=args.model,
        dataset_name=args.dataset,
        master_port=args.master_port,
        debug_mode=args.debug_mode,
    )

    if cfg.debug_mode:
        print('**************************************************')
        print('**********  DEBUG MODE - fast inference **********')

    assert not (cfg.debug_save_train_data and cfg.debug_load_from_saved_data), (
        "Cannot save and load debug data simultaneously."
    )

    _prepare_records_dir(cfg)

    # Set environment variables for address and port
    os.environ['MASTER_ADDR'] = cfg.master_addr
    os.environ['MASTER_PORT'] = cfg.master_port

    # Precompute dataset and store in shared memory
    if cfg.dataset_fraction:
        assert 0.0 < cfg.dataset_fraction < 1.0, "dataset_fraction must be in (0.0, 1.0)"
        print(f"[Debug] Using {cfg.dataset_fraction*100:.1f}% of dataset for quick testing.")

    from FL_code.dataset import precompute_dataset_to_shared
    X_train, y_train = precompute_dataset_to_shared(cfg.dataset_name, cfg.data_folder,
                                                    "train", torch.float32, cfg.dataset_fraction)
    X_test, y_test = precompute_dataset_to_shared(cfg.dataset_name, cfg.data_folder,
                                                  "test", torch.float32, cfg.dataset_fraction)

    cfg.num_classes = torch.unique(y_train).numel()

    print(f'[MAIN] {cfg.codec}')

    # Spawn distributed processes
    mp.spawn(
        _worker,
        args=(cfg.num_clients + 1, cfg, X_train, y_train, X_test, y_test),
        nprocs=cfg.num_clients + 1,
        join=True
    )
