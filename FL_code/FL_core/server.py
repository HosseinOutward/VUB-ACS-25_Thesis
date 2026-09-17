from __future__ import annotations

import torch
import torch.distributed as dist

from FL_code.run_fl import FLConfig
from .utils import set_global_seed, evaluate, recalibrate_batchnorm, setup_fl_worker, format_metrics, StateDictManager
from .codec import simulate_compression, create_protocol


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

    model, _, test_loader, _, sd_manager = setup_fl_worker(
        cfg, "Server", device_id=0,
        X_train=None, y_train=None, X_test=X_test, y_test=y_test
    )

    protocol = create_protocol(
        cfg.protocol,
        sd_manager.get_slices(),
        cache_dir=cfg.protocol_cache_dir,
        cache_overwrite=cfg.protocol_cache_overwrite,
    )

    print(f"[Server] Starting FL with {num_clients} clients, {round(sd_manager.param_count/1e6,1)}M trainable params")
    print(f"[Server] Using protocol: {protocol.__class__.__name__}")

    # FedAvgM as a plain torch optimizer over the aggregated delta (see aggregate_and_update).
    server_opt = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.server_lr, momentum=cfg.server_momentum, nesterov=cfg.server_nesterov)

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
               protocol, delta_vec, rcvd_client_id, rnd_i,
               sd_manager=sd_manager,
               save_dir=cfg.records_dir, server_eval_metrics=metrics,
               worker_eval_metrics=worker_metrics_vec.tolist(), metric_keys=list(metrics.keys()))
            if cfg.debug_save_recons is not None:
                recon_data_path = cfg.debug_data_folder / cfg.debug_save_recons
                recon_data_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    recon_delta_vec.cpu(),
                    recon_data_path / f'round_{rnd_i}_client_{rcvd_client_id}.pt',
                )

            grads_list.append(sd_manager.unflatten(recon_delta_vec))

        aggregate_and_update(model, grads_list, sample_counts, sd_manager, server_opt)

    # Signal clients to terminate
    dist.broadcast(torch.tensor([-1], dtype=torch.long), src=0)


def aggregate_and_update(
    model: torch.nn.Module,
    grads_list: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
    sd_manager: StateDictManager,
    optimizer: torch.optim.Optimizer,
) -> None:
    """FedAvg aggregation weighted by sample counts, applied through the server optimizer."""
    total_samples = sum(sample_counts)
    weights = [n / total_samples for n in sample_counts]

    # Reconstruction histories can retain these dictionaries, so aggregation must not alter them.
    aggregated_delta = {key: grads_list[0][key].clone() for key in sd_manager.keys}
    for key in sd_manager.keys:
        aggregated_delta[key].mul_(weights[0])
        for grad_dict, weight in zip(grads_list[1:], weights[1:]):
            aggregated_delta[key].add_(grad_dict[key], alpha=weight)

    # Optimizers descend, and the aggregated delta is the direction to move toward, so it enters
    # negated. Assigning replaces last round's grad outright, which is why there is no zero_grad.
    parameters = dict(model.named_parameters())
    for key in sd_manager.keys:
        target = parameters[key]
        target.grad = aggregated_delta[key].to(device=target.device, dtype=target.dtype).neg()
    optimizer.step()
