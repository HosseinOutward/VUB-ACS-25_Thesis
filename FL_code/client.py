from __future__ import annotations

from collections import defaultdict
import torch
import torch.distributed as dist

from run_fl import FLConfig
from utils import set_global_seed, recalibrate_batchnorm, evaluate, setup_fl_worker, format_metrics


def run_federated_client(
    cfg: FLConfig,
    rank: int,
    world_size: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> None:
    """Client performs local training and gradient compression."""
    client_id = rank - 1
    set_global_seed(cfg.seed + 1000 * rank)

    model, device, test_loader, train_loader, sd_manager = setup_fl_worker(
        cfg, f"Client {client_id}", device_id=client_id,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        client_id=client_id, num_clients=world_size - 1
    )

    optimizer = model.configure_optimizer(device)
    use_amp = cfg.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    curr_rnd_i = 0
    while True:
        print(f"[Client {client_id}] Starting round {curr_rnd_i}")

        # ---- Receive updated global model from server ----
        # Get current round number
        srvr_rnd = torch.zeros(1, dtype=torch.long)
        dist.broadcast(srvr_rnd, src=0)
        srvr_rnd = srvr_rnd.item()

        # Check if training is complete
        if srvr_rnd == -1:
            assert curr_rnd_i == cfg.rounds, f"Client {client_id} received termination signal but at round {curr_rnd_i}"
            print(f"[Client {client_id}] Training complete. Shutting down.")
            break

        # Get model state dict
        vec_srvr_sd = torch.zeros(sd_manager.param_count, dtype=torch.float32, device='cpu')
        dist.broadcast(vec_srvr_sd, src=0)
        model.load_state_dict(sd_manager.unflatten(vec_srvr_sd), strict=False)

        # Reset optimizer state after loading new parameters from server (optional)
        # optimizer.state = defaultdict(dict)

        if cfg.recalibrate_bn:
            recalibrate_batchnorm(model, train_loader, device, cfg.bn_recalib_batches)

        assert srvr_rnd == curr_rnd_i, f"Round mismatch: expected {curr_rnd_i}, got {srvr_rnd}"

        # ---- Train the local model ----
        print(f"[Client {client_id}] Starting local training for {cfg.local_epochs} epoch(s)")
        pre_train_state = sd_manager.clone_trainable(model.state_dict())

        for _ in range(cfg.local_epochs):
            model.train_epoch(train_loader, optimizer, device, scaler, use_amp, cfg)

        post_train_state = sd_manager.clone_trainable(model.state_dict())

        # ---- Evaluate local model post-training ----
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        print(f"[Client {client_id}] Post-training - {format_metrics(test_metrics, 'Test')} "
              f"| {format_metrics(train_metrics, 'Train')}")

        # ---- Send model delta to server ----
        delta = sd_manager.compute_delta(post_train_state, pre_train_state)
        delta_vec = sd_manager.flatten(delta).cpu().contiguous()

        # Flatten worker eval metrics (train first, then test)
        metric_keys = list(train_metrics.keys())
        worker_metrics_list = [train_metrics[k] for k in metric_keys] + [test_metrics[k] for k in metric_keys]
        worker_metrics_vec = torch.tensor(worker_metrics_list, dtype=torch.float32)

        dist.send(torch.tensor([client_id, curr_rnd_i], dtype=torch.long), dst=0)
        dist.send(torch.tensor([len(train_loader.dataset)], dtype=torch.long), dst=0)
        dist.send(delta_vec, dst=0)
        dist.send(worker_metrics_vec, dst=0)

        print(f"[Client {client_id}] Broadcast complete for round {srvr_rnd}")
        curr_rnd_i += 1
