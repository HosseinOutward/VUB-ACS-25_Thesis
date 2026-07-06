from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import optim as optim, nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

from FL_code.other_codes.resnet56 import ResNet56CIFAR
from FL_code.run_fl import FLConfig


class FLModelTemplate(nn.Module, ABC):
    """Minimalist base class for federated learning models."""

    def __init__(self, cfg: FLConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

    @abstractmethod
    def configure_optimizer(self) -> optim.Optimizer:
        """Configure and return the optimizer for this model."""
        ...

    @abstractmethod
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Execute a single training step on one batch."""
        ...

    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, scaler: torch.amp.GradScaler) -> None:
        """Train for one complete epoch."""
        self.train()

        for batch in dataloader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.cfg.mixed_precision):
                loss = self.training_step(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


class ResNetFLModel(FLModelTemplate):
    """Unified ResNet model for federated learning."""

    def __init__(self, cfg: FLConfig, device: torch.device, model: nn.Module) -> None:
        super().__init__(cfg, device)
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizer(self) -> optim.Optimizer:
        return optim.SGD(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            fused=self.cfg.fused_optimizer
        )

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = batch[0].to(self.device, non_blocking=True)
        y = batch[1].to(self.device, non_blocking=True)
        if self.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss


def _create_backbone(name: str, num_classes: int) -> nn.Module:
    """Create model backbone by name."""
    if name == 'resnet18':
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == 'resnet50':
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == 'resnet56':
        m = ResNet56CIFAR(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Available: resnet18, resnet50, resnet56")
    return m


def initialize_model(cfg: FLConfig, device: torch.device) -> FLModelTemplate:
    """Initialize model with optimizations."""
    model_name = cfg.model_name.lower()
    backbone = _create_backbone(model_name, cfg.num_classes).to(device)
    model = ResNetFLModel(cfg, device, backbone)

    if device.type == "cuda":
        if cfg.channels_last:
            for module in model.modules():
                if not isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    continue
                for param in module.parameters():
                    if param.ndim == 4:
                        param.data = param.data.contiguous(memory_format=torch.channels_last)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return model
