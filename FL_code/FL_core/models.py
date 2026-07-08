from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50


if TYPE_CHECKING:
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


class ResNet56CIFAR(nn.Module):
    """ResNet-56 for CIFAR (32x32 images). 3 stages × 9 blocks each."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 9, stride=1)
        self.layer2 = self._make_layer(32, 9, stride=2)
        self.layer3 = self._make_layer(64, 9, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = [BasicBlockCIFAR(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(n - 1):
            layers.append(BasicBlockCIFAR(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.avgpool(x), 1))


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


class BasicBlockCIFAR(nn.Module):
    """Basic residual block for CIFAR ResNet."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))
