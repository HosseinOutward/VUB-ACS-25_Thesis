from __future__ import annotations
import random
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from models import FLModelTemplate, initialize_model
from dataset import create_dataloader


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
    cfg,
    role: str,
    device_id: int,
    X_train: Optional[torch.Tensor],
    y_train: Optional[torch.Tensor],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    client_id: Optional[int] = None,
    num_clients: Optional[int] = None
) -> Tuple[FLModelTemplate, torch.device, DataLoader, Optional[DataLoader], StateDictManager]:
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


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dict into a readable string."""
    p = f"{prefix} " if prefix else ""
    return f"{p}Loss: {metrics['loss']:.4f}, Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}"


@torch.no_grad()
def recalibrate_batchnorm(model: FLModelTemplate, loader: DataLoader, device: torch.device, max_batches: int = 50) -> None:
    """Recalibrate BatchNorm running statistics (critical for FL)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()

    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        if x.ndim == 4 and next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)
        model(x)


@torch.no_grad()
def evaluate(model: FLModelTemplate, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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
    def __init__(self, model: nn.Module):
        self.keys: List[str] = []
        self.shapes: List[torch.Size] = []
        self.numels: List[int] = []

        # Extract trainable parameters metadata
        for key, param in model.named_parameters():
            if param.requires_grad:
                self.keys.append(key)
                self.shapes.append(param.size())
                self.numels.append(param.numel())

        self.param_count = sum(self.numels)

    def flatten(self, state_dict: dict) -> torch.Tensor:
        flat_list = []
        for key in self.keys:
            param = state_dict[key]
            flat_list.append(param.detach().reshape(-1))
        return torch.cat(flat_list, out=None)

    def unflatten(self, flat_vector: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        state_dict = OrderedDict()
        offset = 0

        for key, shape, numel in zip(self.keys, self.shapes, self.numels):
            param_flat = flat_vector[offset:offset + numel]
            state_dict[key] = param_flat.view(shape)
            offset += numel

        return state_dict

    def clone_trainable(self, state_dict: dict) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((k, state_dict[k].detach().clone()) for k in self.keys)

    def compute_delta(self, new_state: dict, old_state: dict) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((k, new_state[k] - old_state[k]) for k in self.keys)

    def apply_delta_inplace(self, state_dict: dict, delta: dict) -> None:
        for key in self.keys:
            state_dict[key].add_(delta[key].to(state_dict[key].device))
