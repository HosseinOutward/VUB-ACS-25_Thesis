import argparse
import os
from dataclasses import dataclass
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path

_DEBUG_FLAG = False # True
if _DEBUG_FLAG:
    print('**************************************************')
    print('******************  DEBUG MODE  ******************')

@dataclass
class FLConfig:
    """Federated learning configuration."""
    # Codec to use: identity, basic, ?_split_codec (2,3,...), debug_CancerWithBoundCalc
    # non_wz_learned, cancer (_w_outlier, _basic_norm, _binary), temporal_only
    codec: str = "cancer_w_outlier"
    model_name: str = "resnet18"  # resnet18, resnet50, resnet56
    dataset_name: str = "SVHN"    # SVHN, CIFAR10

    num_clients: int = 5 if not _DEBUG_FLAG else 3
    num_loader_workers: int = 2
    num_classes: int = 10
    data_folder: str = "data"
    rounds: int = 50
    local_epochs: int = 5 if not _DEBUG_FLAG else 1
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
    records_dir: str | None = f"records"  # Directory to save records, None to disable
    dataset_fraction: float = None if not _DEBUG_FLAG else 0.1  # Fraction of dataset to use or None for full

    backend: str = "gloo" # "gloo" only, "nccl" for GPU/Linux and doesn't support cpu
    master_addr: str = "localhost"
    master_port: str = "29500"

    debug_folder:Path | bool = Path('experiments/debuging/debugging_data') # false to disable all debug saving
    debug_save_deltas:str = 'delta_vec_data'


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
    cfg = FLConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", type=str, default=cfg.codec,)
    ap.add_argument("--model", type=str, default=cfg.model_name,
                    choices=['resnet18', 'resnet50', 'resnet56'])
    ap.add_argument("--dataset", type=str, default=cfg.dataset_name,
                    choices=['SVHN', 'CIFAR10'])
    ap.add_argument("--master-port", type=str, default=cfg.master_port)
    args = ap.parse_args()
    cfg = FLConfig(
        codec=args.codec,
        model_name=args.model,
        dataset_name=args.dataset,
        master_port=args.master_port
    )

    # Auto-create run folder with incremented number
    if cfg.records_dir:
        from pathlib import Path
        import json
        base_dir = Path(cfg.records_dir)
        base_dir.mkdir(exist_ok=True, parents=True)

        # Find next run number
        run_num = 1
        file_list = os.listdir(base_dir)
        while any([f"run{run_num}" in f for f in file_list]):
            run_num += 1

        run_dir = base_dir / f"run{run_num}_{cfg.codec}"
        run_dir.mkdir()
        cfg.records_dir = str(run_dir)

        # Save FL config
        fl_config_dict = {k: v for k, v in cfg.__dict__.items()}
        with open(run_dir / "fl_config.json", 'w') as f:
            def json_default(obj):
                if isinstance(obj, Path): return str(obj)
                raise TypeError(f"{obj.__class__.__name__} is not JSON serializable")
            json.dump(fl_config_dict, f, indent=2, default=json_default)

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
    from dataset import precompute_dataset_to_shared
    if cfg.dataset_fraction:
        assert 0.0 < cfg.dataset_fraction < 1.0, "dataset_fraction must be in (0.0, 1.0)"
        print(f"[Debug] Using {cfg.dataset_fraction*100:.1f}% of dataset for quick testing.")
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

