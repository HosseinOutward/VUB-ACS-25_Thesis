import os
from dataclasses import dataclass
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


@dataclass
class FLConfig:
    """Federated learning configuration."""
    codec: str = "cancer_only_normalize" # "identity", "basic", "cancer", "cancer_raw", "cancer_only_normalize"

    num_clients: int = 5
    num_loader_workers: int = 2
    num_classes: int = 10
    data_folder: str = "data"
    dataset_name: str = "SVHN"
    rounds: int = 120
    local_epochs: int = 10
    batch_size: int = 5000 # 500 for every 10GB
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
    records_dir: str | None = f"records"  # Directory to save records, None to disable
    dataset_fraction: float = None  # Fraction of dataset to use or None for full dataset

    backend: str = "gloo" # "gloo" only, "nccl" for GPU/Linux and doesn't support cpu
    master_addr: str = "localhost"
    master_port: str = "29500"


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
    dist.init_process_group(backend=cfg.backend, init_method='env://', world_size=world_size, rank=rank)

    role = "Server" if rank == 0 else f"Client {rank - 1}"

    try:
        if rank == 0:
            from server import run_federated_server
            run_federated_server(cfg, rank, world_size, X_test, y_test)
        else:
            from client import run_federated_client
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
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--codec", type=str, default=def_cfg.codec, choices=["identity"])
    # args = ap.parse_args()
    # cfg = FLConfig(
    #     codec=args.codec,
    # )

    # Initialize configuration
    cfg = FLConfig()

    # Auto-create run folder with incremented number
    if cfg.records_dir:
        from pathlib import Path
        import json
        base_dir = Path(cfg.records_dir)
        base_dir.mkdir(exist_ok=True, parents=True)

        # Find next run number
        run_num = 1
        while (base_dir / f"run{run_num}").exists():
            run_num += 1

        run_dir = base_dir / f"run{run_num}"
        run_dir.mkdir()
        cfg.records_dir = str(run_dir)

        # Save FL config
        fl_config_dict = {k: v for k, v in cfg.__dict__.items()}
        with open(run_dir / "fl_config.json", 'w') as f:
            json.dump(fl_config_dict, f, indent=2)

        # Save codec config if cancer codec
        if "cancer" in cfg.codec.lower():
            from cancer_protocol import CancerConfig
            c_cfg = CancerConfig()
            codec_config_dict = {k: v for k, v in c_cfg.__dict__.items()}
            with open(run_dir / "codec_config.json", 'w') as f:
                json.dump(codec_config_dict, f, indent=2, default=str)

    # Set environment variables for address and port
    os.environ['MASTER_ADDR'] = cfg.master_addr
    os.environ['MASTER_PORT'] = cfg.master_port

    # Precompute dataset and store in shared memory
    from dataset import precompute_svhn_to_shared
    if cfg.dataset_fraction:
        assert 0.0 < cfg.dataset_fraction < 1.0, "dataset_fraction must be in (0.0, 1.0)"
        print(f"[Debug] Using {cfg.dataset_fraction*100:.1f}% of dataset for quick testing.")
    X_train, y_train = precompute_svhn_to_shared(cfg.data_folder, "train", torch.float32, cfg.dataset_fraction)
    X_test, y_test = precompute_svhn_to_shared(cfg.data_folder, "test", torch.float32, cfg.dataset_fraction)

    cfg.num_classes = torch.unique(y_train).numel()

    # Spawn distributed processes
    mp.spawn(
        _worker,
        args=(cfg.num_clients + 1, cfg, X_train, y_train, X_test, y_test),
        nprocs=cfg.num_clients + 1,
        join=True
    )

