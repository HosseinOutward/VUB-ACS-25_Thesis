from __future__ import annotations

import torch
import torch.distributed as dist

from FL_code.run_fl import FLConfig
from FL_code.utils import (
    EVAL_METRIC_KEYS, set_global_seed, recalibrate_batchnorm, evaluate, setup_fl_worker, format_metrics)


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

    optimizer = model.configure_optimizer()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)

    # Reused receive buffers; dist.broadcast overwrites them in place each round.
    round_signal = torch.zeros(1, dtype=torch.long)
    vec_srvr_sd = torch.zeros(sd_manager.param_count, dtype=torch.float32, device='cpu')

    curr_rnd_i = 0
    while True:
        # ---- Receive updated global model from server ----
        dist.broadcast(round_signal, src=0)
        srvr_rnd = round_signal.item()

        # Check if training is complete
        if srvr_rnd == -1:
            assert curr_rnd_i == cfg.rounds, f"Client {client_id} received termination signal but at round {curr_rnd_i}"
            print(f"[Client {client_id}] Training complete. Shutting down.")
            break

        assert srvr_rnd == curr_rnd_i, f"Round mismatch: expected {curr_rnd_i}, got {srvr_rnd}"
        print(f"[Client {client_id}] Starting round {curr_rnd_i}")

        # Get model state dict
        dist.broadcast(vec_srvr_sd, src=0)
        sd_manager.load_trainable_state(model, sd_manager.unflatten(vec_srvr_sd))

        # Reset optimizer state after loading new parameters from server (optional)
        # optimizer.state = defaultdict(dict)

        # ---- Train the local model ----
        print(f"[Client {client_id}] Starting local training for {cfg.local_epochs} epoch(s)")
        pre_train_state = sd_manager.clone_trainable(model.state_dict())

        if not cfg.debug_load_from_saved_data:
            for _ in range(cfg.local_epochs):
                model.train_epoch(train_loader, optimizer, scaler)
        else:
            assert not cfg.debug_save_train_data
            delta_data_path = cfg.debug_data_folder / cfg.debug_save_deltas
            delta_data_path = delta_data_path / f'round_{curr_rnd_i}_client_{client_id}.pt'
            print(f"[Client {client_id}] Debug mode: Skipping actual training and using pre-trained model state")
            loaded_state_dict = sd_manager.unflatten(-torch.load(delta_data_path))
            loaded_state_dict = sd_manager.compute_delta(pre_train_state, loaded_state_dict)
            sd_manager.load_trainable_state(model, loaded_state_dict)
            if cfg.recalibrate_bn:
                recalibrate_batchnorm(model, train_loader, cfg.bn_recalib_batches)

        post_train_state = sd_manager.clone_trainable(model.state_dict())

        # ---- Evaluate local model post-training (skipped rounds report NaN) ----
        run_local_eval = (curr_rnd_i % cfg.client_eval_every_n_rounds == 0) or (curr_rnd_i == cfg.rounds - 1)
        if run_local_eval:
            train_metrics = evaluate(model, train_loader)
            test_metrics = evaluate(model, test_loader)
            print(f"[Client {client_id}] Post-training - {format_metrics(test_metrics, 'Test')} "
                  f"| {format_metrics(train_metrics, 'Train')}")
        else:
            train_metrics = dict.fromkeys(EVAL_METRIC_KEYS, float("nan"))
            test_metrics = dict.fromkeys(EVAL_METRIC_KEYS, float("nan"))

        # ---- Send model delta to server ----
        delta = sd_manager.compute_delta(post_train_state, pre_train_state)
        delta_vec = sd_manager.flatten(delta).cpu().contiguous()

        # Flatten worker eval metrics (train first, then test)
        worker_metrics_list = [train_metrics[k] for k in EVAL_METRIC_KEYS] + \
                              [test_metrics[k] for k in EVAL_METRIC_KEYS]
        worker_metrics_vec = torch.tensor(worker_metrics_list, dtype=torch.float32)

        dist.send(torch.tensor([client_id, curr_rnd_i], dtype=torch.long), dst=0)
        dist.send(torch.tensor([len(train_loader.dataset)], dtype=torch.long), dst=0)
        dist.send(delta_vec, dst=0)
        dist.send(worker_metrics_vec, dst=0)

        print(f"[Client {client_id}] Broadcast complete for round {srvr_rnd}")

        if cfg.debug_save_train_data:
            print(f"[Client {client_id}] Debug mode: Saving delta vector for round {curr_rnd_i}")
            delta_data_path = cfg.debug_data_folder / cfg.debug_save_deltas
            delta_data_path = delta_data_path / f'round_{curr_rnd_i}_client_{client_id}.pt'
            torch.save(delta_vec, delta_data_path)

        curr_rnd_i += 1
