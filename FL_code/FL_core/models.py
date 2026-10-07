from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, resnet18, resnet50, vgg16
from torchvision.models.vision_transformer import VisionTransformer
from transformers import AutoConfig, AutoModel


if TYPE_CHECKING:
    from FL_code.run_fl import FLConfig


# --- Checkpoints and augmentation policy --- #
# Shared by the backbone and by the text preprocessing, which must tokenize with this checkpoint's
# own WordPiece vocabulary, at this sequence length, for the pretrained weights to mean anything.
BERT_CHECKPOINT: Final[str] = "google/bert_uncased_L-4_H-256_A-4"
DISTILBERT_CHECKPOINT: Final[str] = "distilbert-base-uncased"
BERT_SEQUENCE_LENGTH: Final[int] = 128
# Digits stop meaning the same thing under reflection, so SVHN and MNIST never get flipped.
HORIZONTAL_FLIP_SAFE: Final[frozenset[str]] = frozenset(
    {'CIFAR10', 'CIFAR100', 'STL10', 'IMAGENETTE', 'EUROSAT', 'FOOD101', 'TINYIMAGENET'})
# Every backbone `_create_backbone` can build. The single source of truth: the factory's error
# message and the run_fl CLI both read it, so neither can list a set that has drifted from reality.
BACKBONE_NAMES: Final[tuple[str, ...]] = (
    'resnet18', 'resnet18_cifar', 'resnet50', 'resnet56', 'mobilenet_v2', 'mobilenet_v2_cifar',
    'vgg16_conv',
    'vit_b2_p4', 'bert', 'bert_scratch', 'distilbert', 'distilbert_fresh', 'lstm_lm',
)


# --- Federated model wrapper --- #
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
            if self.cfg.grad_clip_norm > 0.0:
                # Unscale first, or the threshold would be compared against gradients still
                # carrying the AMP loss scale and would clip at an arbitrary multiple of itself.
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()


def augment_batch(images: torch.Tensor, allow_flip: bool) -> torch.Tensor:
    """Per-image random crop with reflection padding, plus an optional horizontal flip.

    Each dataset is precomputed into one normalized tensor for zero-copy sharing, so augmentation
    cannot live in a dataloader transform. Doing it per batch on the training device keeps that
    design and costs almost nothing; the offsets are drawn per image rather than per batch, which
    a batched torchvision transform would not do.
    """
    count, _, height, width = images.shape
    pad = max(1, height // 8)
    padded = nn.functional.pad(images, (pad, pad, pad, pad), mode="reflect")
    offsets = torch.randint(0, 2 * pad + 1, (count, 2), device=images.device)
    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
    columns = offsets[:, 1, None] + torch.arange(width, device=images.device)
    cropped = padded[
        torch.arange(count, device=images.device)[:, None, None], :,
        rows[:, :, None], columns[:, None, :],
    ].permute(0, 3, 1, 2)

    if allow_flip:
        flip = torch.rand(count, device=images.device) < 0.5
        cropped[flip] = cropped[flip].flip(-1)
    return cropped.contiguous()


class ClassificationFLModel(FLModelTemplate):
    """Cross-entropy wrapper around any backbone the federated learning loop trains.

    Classifiers give one logit row per sample; the language model gives `[batch, vocab, sequence]`
    against a `[batch, sequence]` target, which is the K-dimensional form the same loss accepts.
    Augmentation only applies to the four-dimensional image case.
    """

    def __init__(self, cfg: FLConfig, device: torch.device, model: nn.Module) -> None:
        super().__init__(cfg, device)
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizer(self) -> optim.Optimizer:
        # Plain SGD: no momentum and no weight decay. Decay would add -lr*wd*w to every client's
        # delta, and since all clients start a round from the same global weights that term is
        # identical across them - a shared, perfectly predictable component injected into the very
        # quantity this sweep asks one client's delta to predict from another's.
        return optim.SGD(self.parameters(), lr=self.cfg.lr, fused=self.cfg.fused_optimizer)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = batch[0].to(self.device, non_blocking=True)
        y = batch[1].to(self.device, non_blocking=True)
        if x.ndim == 4:
            if self.cfg.augment:
                x = augment_batch(x, self.cfg.dataset_name.upper() in HORIZONTAL_FLIP_SAFE)
            if self.cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss


# --- Backbones built here rather than taken from torchvision --- #
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


class BertClassifier(nn.Module):
    """Pretrained BERT encoder with a freshly initialized head, for single-document classification.

    Produced by `_create_backbone` for 20NEWS and consumed by the FL model. Its input is the
    `[batch, 2, sequence]` token-id / attention-mask pair that the 20NEWS preprocessing emits,
    tokenized with this same checkpoint's WordPiece vocabulary.
    """

    def __init__(
        self, num_classes: int, checkpoint: str = BERT_CHECKPOINT, pretrained: bool = True,
        reinit_encoder: bool = False,
    ) -> None:
        super().__init__()
        # Untrained, the checkpoint supplies only the architecture. That is the point of the
        # `bert_scratch` variant: every pretrained text setup here reaches its final accuracy
        # inside round 0, so none of them exercise the training phase the sweep measures.
        self.encoder = (AutoModel.from_pretrained(checkpoint) if pretrained
                        else AutoModel.from_config(AutoConfig.from_pretrained(checkpoint)))
        if reinit_encoder:
            # The middle ground between the two extremes above. Fully pretrained, DistilBERT is
            # done by round 11 and all four measurement windows land on the same converged model;
            # fully random, it would have to learn a 30,522-token embedding from 48k rows under
            # plain SGD. Keeping the embeddings and resetting only the transformer blocks leaves
            # the vocabulary free and every layer of task structure still to be learned.
            blocks = getattr(getattr(self.encoder, "transformer", None), "layer", None)
            if blocks is None:                                  # BERT names it encoder.layer
                blocks = self.encoder.encoder.layer
            blocks.apply(self.encoder._init_weights)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3 and x.size(1) == 2, (
            f"BERT expects [batch, 2, sequence] token IDs and attention masks; got {tuple(x.shape)}."
        )
        input_ids, attention_mask = x.unbind(dim=1)
        max_length = self.encoder.config.max_position_embeddings
        assert input_ids.size(1) <= max_length, (
            f"BERT sequence length {input_ids.size(1)} exceeds its {max_length}-token limit."
        )
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler, so fall back to the [CLS] hidden state, which is what its own
        # sequence-classification head uses.
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        return self.classifier(pooled)


class LSTMLanguageModel(nn.Module):
    """Two-layer word-level LSTM with a tied embedding/decoder - the standard WikiText baseline.

    Produced by `_create_backbone` for WIKITEXT2 and consumed by the FL model. Input is a
    `[batch, sequence]` block of vocabulary ids; output is `[batch, vocab, sequence]`, which is the
    layout `nn.CrossEntropyLoss` wants for a K-dimensional target, so the `[batch, sequence]` target
    of next tokens needs no reshaping at the loss.

    Tying the decoder to the embedding is what keeps this at 21M rather than 38M parameters, and is
    standard for this benchmark. It also means the shared matrix appears once in `named_parameters`,
    so the state-dict manager sees one 17M-parameter tensor rather than two.
    """

    def __init__(self, vocab_size: int, width: int = 512, layers: int = 2,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.lstm = nn.LSTM(width, width, layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(width, vocab_size)
        self.decoder.weight = self.embedding.weight
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"LSTM expects [batch, sequence] token ids; got {tuple(x.shape)}."
        hidden, _ = self.lstm(self.dropout(self.embedding(x)))
        return self.decoder(self.dropout(hidden)).transpose(1, 2)


# --- Model construction --- #
def _create_backbone(name: str, num_classes: int, image_size: int = 32) -> nn.Module:
    """Create model backbone by name. Only the ViT depends on the input resolution."""
    if name == 'resnet18':
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == 'resnet18_cifar':
        m = resnet18(weights=None)
        # CIFAR stem: a 3x3 stride-1 convolution and no max-pool, so 32x32 inputs reach layer1 at
        # full resolution instead of the 4x downsample the ImageNet stem applies.
        m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == 'resnet50':
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == 'resnet56':
        m = ResNet56CIFAR(num_classes)
    elif name == 'vgg16_conv':
        # VGG16's convolutional stack with its 119M fully-connected head replaced by a single
        # linear layer, which is the only part of VGG16 that fits a small-image budget.
        m = nn.Sequential(
            vgg16(weights=None).features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )
    elif name == 'vit_b2_p4':
        # The first two blocks of ViT-B at its full width, with a patch size small enough that a
        # 32x32 image still yields 64 tokens.
        m = VisionTransformer(
            image_size=image_size, patch_size=4, num_layers=2, num_heads=12,
            hidden_dim=768, mlp_dim=3_072, num_classes=num_classes,
        )
    elif name == 'mobilenet_v2':
        # Stock stem, for the 64px sets. ResNet-18 and ResNet-50 keep theirs there too, so the three
        # models see the same feature-map sizes and the comparison is of capacity, not of stride.
        m = mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif name == 'mobilenet_v2_cifar':
        m = mobilenet_v2(weights=None)
        # CIFAR stem: the stock network halves the resolution four times before the features do any
        # work, which leaves a 32x32 input at 1x1. Drop the first two stride-2 steps.
        m.features[0][0] = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        m.features[2].conv[1][0].stride = (1, 1)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif name == 'distilbert':
        m = BertClassifier(num_classes, DISTILBERT_CHECKPOINT)
    elif name == 'distilbert_fresh':
        m = BertClassifier(num_classes, DISTILBERT_CHECKPOINT, reinit_encoder=True)
    elif name == 'bert':
        m = BertClassifier(num_classes)
    elif name == 'bert_scratch':
        m = BertClassifier(num_classes, BERT_CHECKPOINT, pretrained=False)
    elif name == 'lstm_lm':
        # `num_classes` is the vocabulary here: the model predicts the next token, so its output
        # cardinality and its input cardinality are the same thing.
        m = LSTMLanguageModel(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Available: {', '.join(BACKBONE_NAMES)}")
    return m


def initialize_model(cfg: FLConfig, device: torch.device) -> FLModelTemplate:
    """Build the configured backbone on `device` without changing backend precision settings."""
    assert cfg.num_classes is not None, "cfg.num_classes must be set before model initialization."
    model_name = cfg.model_name.lower()
    backbone = _create_backbone(model_name, cfg.num_classes, cfg.image_size).to(device)
    model = ClassificationFLModel(cfg, device, backbone)

    if device.type == "cuda":
        if cfg.channels_last:
            for module in model.modules():
                if not isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    continue
                for param in module.parameters():
                    if param.ndim == 4:
                        param.data = param.data.contiguous(memory_format=torch.channels_last)

    return model
