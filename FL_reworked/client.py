from __future__ import annotations

import torch
import torch.distributed as dist

from run_fl import FLConfig
from utils import set_global_seed, get_device, recalibrate_batchnorm, StateDictManager, evaluate
from models import initialize_model
from dataset import create_dataloader


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
    set_global_seed(cfg.seed + 1000 * rank)
    client_id = rank - 1

    device = get_device(client_id)
    print(f"[Client {client_id}] Device: {device}")

    model = initialize_model(cfg, device)

    train_loader = create_dataloader(X_train, y_train, cfg, device, is_train=True,
                               client_id=client_id, num_clients=world_size - 1)

    # Create test loader for evaluation
    test_loader = create_dataloader(X_test, y_test, cfg, device, is_train=False)

    # Initialize state dict manager to handle parameter structure
    sd_manager = StateDictManager(model)

    optimizer = model.configure_optimizer(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    curr_rnd_i = 0
    while True:
        print(f"[Client {client_id}] Starting round {curr_rnd_i}")

        # ---- Receive updated global model from server (and get current round number) ----
        # getting server state dict
        vec_srvr_sd = torch.zeros(sd_manager.param_count, dtype=torch.float32, device='cpu')
        dist.broadcast(vec_srvr_sd, src=0)
        srvr_sd = sd_manager.unflatten(vec_srvr_sd)

        model.load_state_dict(srvr_sd, strict=False)

        if cfg.recalibrate_bn:
            recalibrate_batchnorm(model, train_loader, device, cfg.bn_recalib_batches)

        # Get current round number
        srvr_rnd = torch.zeros(1, dtype=torch.long)
        dist.broadcast(srvr_rnd, src=0)
        srvr_rnd = srvr_rnd.item()

        # Check if training is complete
        if srvr_rnd == -1:
            assert curr_rnd_i == cfg.rounds, f"Client {client_id} received termination signal but at round {curr_rnd_i}"
            print(f"[Client {client_id}] Training complete. Shutting down.")
            break

        assert srvr_rnd == curr_rnd_i, f"Round mismatch: expected {curr_rnd_i}, got {srvr_rnd}"

        # ---- train the local model ----
        print(f"[Client {client_id}] Starting local training for {cfg.local_epochs} epoch(s)")
        pre_train_state = sd_manager.clone_trainable(model.state_dict())

        for _ in range(cfg.local_epochs):
            model.train_epoch(train_loader, optimizer, device, scaler, use_amp, cfg)

        post_train_state = sd_manager.clone_trainable(model.state_dict())

        # ---- evaluate local model post-training ----
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        print(f"[Client {client_id}] Post-training - Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, AUC: {train_metrics['auc']:.4f} | Test Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}")

        # ---- send model delta to server ----
        delta = sd_manager.compute_delta(post_train_state, pre_train_state)
        delta_vec = sd_manager.flatten(delta).cpu().contiguous()
        n_samples = len(train_loader.dataset)

        # Send client_id and round_id
        dist.send(torch.tensor([client_id, curr_rnd_i], dtype=torch.long), dst=0)
        # Send number of samples
        dist.send(torch.tensor([n_samples], dtype=torch.long), dst=0)
        # Send delta
        dist.send(delta_vec, dst=0)

        print(f"[Client {client_id}] Broadcast complete for round {srvr_rnd}")

        # ---- wrap up ----
        curr_rnd_i+=1
