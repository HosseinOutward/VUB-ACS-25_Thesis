from __future__ import annotations
import random
import sys
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

from models import FLModelTemplate, initialize_model
from dataset import create_dataloader


def create_training_progress_bar(
    iterable_or_total: Any,
    desc: str,
    disable: bool = False,
    leave: bool = False,
    position: int = 0
) -> tqdm:
    common_config = {
        'disable': disable,
        'desc': desc,
        'leave': leave,
        'position': position
    }

    # try:
    #     ipython = get_ipython()  # type: ignore
    # except NameError:
    common_config['file'] = sys.stderr
    common_config['bar_format'] = \
        '{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]{postfix}'

    return tqdm(total=iterable_or_total, **common_config)


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_id: int = 0) -> torch.device:
    """Get GPU device with optimizations enabled."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        return device
    return torch.device("cpu")


def setup_fl_worker(
    cfg: Any,
    role: str,
    device_id: int,
    X_train: torch.Tensor | None,
    y_train: torch.Tensor | None,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    client_id: int | None = None,
    num_clients: int | None = None
) -> tuple[FLModelTemplate, torch.device, DataLoader, DataLoader | None, StateDictManager]:
    """
    Common setup for FL server/client workers.

    Returns:
        model, device, test_loader, train_loader (None for server), sd_manager
    """
    device = get_device(device_id)
    print(f"[{role}] Device: {device}")

    model = initialize_model(cfg, device)
    test_loader = create_dataloader(X_test, y_test, cfg, device, is_train=False)

    train_loader = None
    if X_train is not None and client_id is not None:
        train_loader = create_dataloader(
            X_train, y_train, cfg, device, is_train=True,
            client_id=client_id, num_clients=num_clients
        )

    sd_manager = StateDictManager(model)

    return model, device, test_loader, train_loader, sd_manager


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    """Format metrics dict into a readable string."""
    p = f"{prefix} " if prefix else ""
    return f"{p}Loss: {metrics['loss']:.4f}, Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}"


@torch.no_grad()
def recalibrate_batchnorm(model: FLModelTemplate, loader: DataLoader, max_batches: int = 50) -> None:
    """Recalibrate BatchNorm running statistics (critical for FL)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()

    model.train()
    for i, (x, _) in enumerate(loader):
        x: torch.Tensor
        if i >= max_batches:
            break
        x = x.to(model.device)
        if x.ndim == 4 and model.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        model(x)


def evaluate(model: FLModelTemplate, loader: DataLoader) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(model.device)
            y = y.to(model.device)

            use_channels_last = next(model.parameters()).is_contiguous(memory_format=torch.channels_last)
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            logits = model(x)
            total_loss += loss_fn(logits, y).item() * x.size(0)

            # Get predictions and probabilities
            probs_P = torch.softmax(logits, dim=1)
            y_preds = logits.argmax(dim=1)

            # Move to CPU and store
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs_P.cpu().numpy())
            all_preds.append(y_preds.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    # -- metrics --
    # Calculate metrics using scikit-learn
    avg_loss = total_loss / len(y_true)
    accuracy = accuracy_score(y_true, y_pred)

    # F1 score (macro average for multiclass)
    num_classes = y_probs.shape[1]
    f1 = f1_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary', zero_division=0)

    # AUC calculation
    if num_classes == 2:
        auc = roc_auc_score(y_true, y_probs[:, 1])
    else:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')

    return {
        "loss": avg_loss,
        "acc": accuracy,
        "f1": f1,
        "auc": auc
    }


class StateDictManager:
    def __init__(self, model: nn.Module) -> None:
        self.keys: list[str] = []
        self.shapes: list[torch.Size] = []
        self.numels: list[int] = []

        # Extract trainable parameters metadata
        for key, param in model.named_parameters():
            if param.requires_grad:
                self.keys.append(key)
                self.shapes.append(param.size())
                self.numels.append(param.numel())

        self.param_count = sum(self.numels)

    def flatten(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        flat_list = []
        for key in self.keys:
            param = state_dict[key]
            flat_list.append(param.cpu().detach().reshape(-1))
        return torch.cat(flat_list, out=None)

    def unflatten(self, flat_vector: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        offset = 0

        for key, shape, numel in zip(self.keys, self.shapes, self.numels):
            param_flat = flat_vector[offset:offset + numel]
            state_dict[key] = param_flat.view(shape)
            offset += numel

        return state_dict

    def get_slices(self) -> list[slice]:
        slices = []
        offset = 0
        for numel in self.numels:
            slices.append(slice(offset, offset + numel))
            offset += numel
        return slices

    def clone_trainable(self, state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((k, state_dict[k].cpu().detach().clone()) for k in self.keys)

    def compute_delta(
        self,
        new_state: dict[str, torch.Tensor],
        old_state: dict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((k, new_state[k] - old_state[k]) for k in self.keys)

    def apply_delta_inplace(
        self,
        state_dict: dict[str, torch.Tensor],
        delta: dict[str, torch.Tensor]
    ) -> None:
        for key in self.keys:
            state_dict[key].add_(delta[key].to(state_dict[key].device))
