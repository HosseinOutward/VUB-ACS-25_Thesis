from __future__ import annotations

import torch
import torch.distributed as dist

from run_fl import FLConfig
from utils import set_global_seed, evaluate, recalibrate_batchnorm, get_device, StateDictManager
from models import initialize_model
from dataset import create_dataloader
from codec import create_codec, simulate_compression


def run_federated_server(
    cfg: FLConfig,
    rank: int,
    world_size: int,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> None:
    """Server coordinates FL training and gradient aggregation."""

    assert rank == 0, "Server must have rank 0"

    set_global_seed(cfg.seed)
    num_clients = world_size - 1

    device = get_device()

    model = initialize_model(cfg, device)

    test_loader = create_dataloader(X_test, y_test, cfg, device, is_train=False)

    # Initialize state dict manager to handle parameter structure
    sd_manager = StateDictManager(model)

    # Initialize codec
    codec = create_codec(cfg.codec)

    print(f"[Server] Starting FL with {num_clients} clients, {round(sd_manager.param_count/1e6,1)}M trainable params")
    print(f"[Server] Using codec: {codec.__class__.__name__}")


    for rnd_i in range(cfg.rounds+1):
        # ---- recalibrate then evaluate global model ----
        if cfg.recalibrate_bn:
            recalibrate_batchnorm(model, test_loader, device, cfg.bn_recalib_batches)

        metrics = evaluate(model, test_loader, device)

        ending_round = (rnd_i == cfg.rounds)
        temp = f'Start of Round {rnd_i:03d}' if not ending_round else 'Final'
        print(f"[Server] {temp} - Loss: {metrics['loss']:.4f}, Acc: {metrics['acc']:.4f}")

        if ending_round:
            print("[Server] Training complete.")
            break

        # ---- send global model params and (round for validation) to clients ----
        vectorized_params = sd_manager.flatten(model.state_dict()).contiguous().cpu()
        dist.broadcast(vectorized_params, src=0)
        dist.broadcast(torch.tensor([rnd_i], dtype=torch.long), src=0)

        # ---- receive compressed gradients from clients ----
        grads_list = []
        sample_counts = []

        for client_rank in range(0, num_clients):
            # Receive client id and round id
            temp = torch.zeros(2, dtype=torch.long)
            dist.recv(temp, src=client_rank+1)
            rcvd_client_id, rcvd_round_id = temp.tolist()

            assert rcvd_round_id == rnd_i, f"Round mismatch: expected {rnd_i}, got {rcvd_round_id}"
            assert rcvd_client_id == client_rank, f"Client ID mismatch: expected {client_rank}, got {rcvd_client_id}"

            # -- wait for training completion --
            # Receive number of samples used for training
            n_samples_tensor = torch.zeros(1, dtype=torch.long)
            dist.recv(n_samples_tensor, src=client_rank+1)
            sample_counts.append(n_samples_tensor.item())

            # Receive gradients from clients (as vector)
            rcvd_grads_vec = torch.zeros(sd_manager.param_count, dtype=torch.float32)
            dist.recv(rcvd_grads_vec, src=client_rank+1)

            # Simulate compression with eval metrics
            reconstructed_vec = rcvd_grads_vec #simulate_compression(
            #     codec, rcvd_grads_vec, rcvd_client_id, rnd_i, eval_metrics=metrics)

            # Unflatten back to dict for aggregation
            grads_dict = sd_manager.unflatten(reconstructed_vec)
            grads_list.append(grads_dict)

        _aggregate_and_update(model, grads_list, sample_counts, sd_manager)

    dist.broadcast(torch.tensor([-1], dtype=torch.long), src=0)


def _aggregate_and_update(model, grads_list, sample_counts, sd_manager: StateDictManager) -> None:
    """FedAvg aggregation weighted by sample counts."""
    total_samples = sum(sample_counts)

    # Weighted average of gradients
    aggregated_delta = {}
    for key in sd_manager.keys:
        weighted_sum = torch.zeros_like(grads_list[0][key])
        for grad_dict, n_samples in zip(grads_list, sample_counts):
            weight = n_samples / total_samples
            weighted_sum += grad_dict[key] * weight
        aggregated_delta[key] = weighted_sum

    # Apply aggregated delta to model
    sd_manager.apply_delta_inplace(model.state_dict(), aggregated_delta)
