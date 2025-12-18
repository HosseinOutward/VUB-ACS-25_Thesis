import os
from dataclasses import dataclass
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


@dataclass
class FLConfig:
    """Federated learning configuration."""
    codec: str = "basic"

    num_clients: int = 2
    single_process: bool = False  # Debug mode: run only server in single process
    num_loader_workers: int = 8
    num_classes: int = 10
    data_folder: str = "../data"
    dataset_name: str = "SVHN"
    rounds: int = 80
    local_epochs: int = 20
    batch_size: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 43

    recalibrate_bn: bool = True
    bn_recalib_batches: int = 100
    channels_last: bool = True
    tf32: bool = True # Enable TF32 on Ampere+ GPUs
    use_compile: bool = False # linux only, requires torch>=2.0

    backend: str = "gloo" # "gloo" or "nccl" for GPU/Linux
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
    dist.init_process_group(backend=cfg.backend, init_method='env://', world_size=world_size, rank=rank)

    try:
        if rank == 0:
            from server import run_federated_server
            run_federated_server(cfg, rank, world_size, X_test, y_test)
        else:
            from client import run_federated_client
            run_federated_client(cfg, rank, world_size, X_train, y_train)
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"[Rank {rank}] EXCEPTION OCCURRED:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}\n")
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

    # Set environment variables for address and port
    os.environ['MASTER_ADDR'] = cfg.master_addr
    os.environ['MASTER_PORT'] = cfg.master_port

    # Precompute dataset and store in shared memory
    from dataset import precompute_svhn_to_shared
    X_train, y_train = precompute_svhn_to_shared(cfg.data_folder, "train", torch.float32)
    X_test, y_test = precompute_svhn_to_shared(cfg.data_folder, "test", torch.float32)

    cfg.num_classes = torch.unique(y_train).numel()

    # Run in single-process debug mode or spawn distributed processes
    if cfg.single_process:
        # Debug mode: run server only in single process
        print("[DEBUG] Running in single-process mode")
        dist.init_process_group(backend=cfg.backend, init_method='env://', world_size=1, rank=0)
        from server import run_federated_server
        run_federated_server(cfg, rank=0, world_size=1, X_test=X_test, y_test=y_test)
        dist.destroy_process_group()
    else:
        mp.spawn(
            _worker,
            args=(cfg.num_clients + 1, cfg, X_train, y_train, X_test, y_test),
            nprocs=cfg.num_clients + 1,
            join=True
        )
