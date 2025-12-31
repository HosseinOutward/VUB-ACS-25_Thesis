from __future__ import annotations
from typing import Tuple
from abc import ABC, abstractmethod

import torch
from torch import optim as optim, nn as nn
from torchvision.models import resnet18


class FLModelTemplate(nn.Module, ABC):
    """Minimalist base class for federated learning models."""

    @abstractmethod
    def configure_optimizer(self, device: torch.device) -> optim.Optimizer:
        """Configure and return the optimizer for this model."""
        ...

    @abstractmethod
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device, cfg) -> torch.Tensor:
        """Execute a single training step on one batch."""
        ...

    def train_epoch(
        self,
        dataloader,
        optimizer: optim.Optimizer,
        device: torch.device,
        scaler: torch.amp.GradScaler,
        use_amp: bool,
        cfg,
        max_grad_norm: float = None
    ) -> None:
        """Train for one complete epoch."""
        self.train()

        for batch in dataloader:
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = self.training_step(batch, device, cfg)

            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()


class Resnet18FLModelTemplate(FLModelTemplate):
    """ResNet18 model for federated learning."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, cfg.num_classes)
        self.model = backbone

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def configure_optimizer(self, device: torch.device) -> optim.Optimizer:
        """Configure optimizer."""
        fused = self.cfg.fused_optimizer
        return optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, fused=fused)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device, cfg) -> torch.Tensor:
        """Single training step."""
        x, y = batch

        x = x.to(device)
        y = y.to(device)

        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        return loss



def initialize_model(cfg, device: torch.device) -> FLModelTemplate:
    """Initialize model with optimizations."""
    model = Resnet18FLModelTemplate(cfg)
    model = model.to(device)

    if device.type == "cuda":
        if cfg.channels_last:
            for module in model.modules():
                if not isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    continue
                for param in module.parameters():
                    if param.ndim == 4:
                        param.data = param.data.contiguous(memory_format=torch.channels_last)

        if cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        if cfg.compile_mode:
            model = torch.compile(model, mode=cfg.compile_mode)

    return model
