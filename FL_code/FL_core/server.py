from __future__ import annotations

import torch
import torch.distributed as dist

from FL_code.run_fl import FLConfig
from .utils import set_global_seed, evaluate, recalibrate_batchnorm, setup_fl_worker, format_metrics, StateDictManager
from .codec import simulate_compression
from .codec import create_codec


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

    model, device, test_loader, _, sd_manager = setup_fl_worker(
        cfg, "Server", device_id=0,
        X_train=None, y_train=None, X_test=X_test, y_test=y_test
    )

    if cfg.debug_save_train_data:
        delta_data_path = cfg.debug_data_folder / cfg.debug_save_deltas / f'_initial_model.pt'
        delta_data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), delta_data_path)
    if cfg.debug_load_from_saved_data:
        delta_data_path = cfg.debug_data_folder / cfg.debug_save_deltas / f'_initial_model.pt'
        assert not cfg.debug_save_train_data
        loaded_state_dict = torch.load(delta_data_path)
        model.load_state_dict(loaded_state_dict)

    codec = create_codec(cfg.codec, sd_manager)

    print(f"[Server] Starting FL with {num_clients} clients, {round(sd_manager.param_count/1e6,1)}M trainable params")
    print(f"[Server] Using codec: {codec.__class__.__name__}")

    for rnd_i in range(cfg.rounds + 1):
        # ---- Recalibrate then evaluate global model ----
        if cfg.recalibrate_bn:
            recalibrate_batchnorm(model, test_loader, cfg.bn_recalib_batches)

        metrics = evaluate(model, test_loader)

        # --- Print metrics and check for termination ----
        ending_round = (rnd_i == cfg.rounds)
        label = 'Final' if ending_round else f'Start of Round {rnd_i:03d}'
        print(f"[Server] {label} - {format_metrics(metrics)}")
        if ending_round:
            print("[Server] Training complete.")
            break

        # ---- Send global model params and round number to clients ----
        dist.broadcast(torch.tensor([rnd_i], dtype=torch.long), src=0)
        vectorized_params = sd_manager.flatten(model.state_dict()).contiguous().cpu()
        dist.broadcast(vectorized_params, src=0)

        # ---- Receive deltas from clients ----
        grads_list = []
        sample_counts = []

        for client_rank in range(num_clients):
            # Receive client id and round id
            meta = torch.zeros(2, dtype=torch.long)
            dist.recv(meta, src=client_rank + 1)
            rcvd_client_id, rcvd_round_id = meta.tolist()

            assert rcvd_round_id == rnd_i, f"Round mismatch: expected {rnd_i}, got {rcvd_round_id}"
            assert rcvd_client_id == client_rank, f"Client ID mismatch: expected {client_rank}, got {rcvd_client_id}"

            # Receive number of samples
            n_samples = torch.zeros(1, dtype=torch.long)
            dist.recv(n_samples, src=client_rank + 1)
            sample_counts.append(n_samples.item())

            # Receive delta vector
            delta_vec = torch.zeros(sd_manager.param_count, dtype=torch.float32)
            dist.recv(delta_vec, src=client_rank + 1)

            # Receive worker eval metrics (train + test, dynamically sized)
            num_metrics = len(list(metrics.keys()))
            worker_metrics_vec = torch.zeros(num_metrics * 2, dtype=torch.float32)
            dist.recv(worker_metrics_vec, src=client_rank + 1)

            # ***************** Simulate compression and reconstruct *****************
            # recon_delta_vec = delta_vec 
            recon_delta_vec = simulate_compression(
               codec, delta_vec, rcvd_client_id, rnd_i,
               model_size=sd_manager.param_count,
               save_dir=cfg.records_dir, server_eval_metrics=metrics,
               worker_eval_metrics=worker_metrics_vec.tolist(), metric_keys=list(metrics.keys()))

            grads_list.append(sd_manager.unflatten(recon_delta_vec))

        _aggregate_and_update(model, grads_list, sample_counts, sd_manager)

    # Signal clients to terminate
    dist.broadcast(torch.tensor([-1], dtype=torch.long), src=0)


def _aggregate_and_update(model, grads_list, sample_counts, sd_manager: StateDictManager) -> None:
    """FedAvg aggregation weighted by sample counts."""
    total_samples = sum(sample_counts)
    weights = [n / total_samples for n in sample_counts]

    # Weighted average of gradients (in-place on first gradient)
    aggregated_delta = grads_list[0]
    for key in sd_manager.keys:
        aggregated_delta[key].mul_(weights[0])
        for grad_dict, weight in zip(grads_list[1:], weights[1:]):
            aggregated_delta[key].add_(grad_dict[key], alpha=weight)

    sd_manager.apply_delta_inplace(model, aggregated_delta)
