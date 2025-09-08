import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy
from torchmetrics.classification import AUROC
from torchvision import models
from components.FL_sim import FederatedModelWrapper
from components.other_utilities.resnet20 import ResNet20


class ResNetPLModel(FederatedModelWrapper):
    def __init__(self, num_classes, resnet_version='resnet50', lr=0.01):
        super(ResNetPLModel, self).__init__()

        self.lr = lr
        self.resnet_version = resnet_version
        self.num_classes = num_classes

        # 1) backbone ----------------------------------------------------------
        if resnet_version == 'resnet20':
            backbone = ResNet20(num_classes=num_classes)
        else:
            backbone = dict(
                resnet50=models.resnet50,
                resnet18=models.resnet18,
            )[resnet_version](weights=None)

            # 2) replace the classification head for torchvision models
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)

        self.model = backbone

        # 3) loss & metric -----------------------------------------------------
        self.loss_fn = nn.CrossEntropyLoss()
        self.aucroc = AUROC(num_classes=num_classes, average="weighted", task="multiclass")
        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass")

    def forward(self, x):
        return self.model(x)

    def get_loss_etc(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        auc = self.aucroc(torch.softmax(logits.detach(), dim=1), y)
        acc = self.accuracy(logits.detach(), y)

        etc = (auc, acc,)
        return loss, etc

    def _log_metrics(self, loss, etc, stage: str):
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_auc', etc[0], on_step=True, on_epoch=True, prog_bar=True)

    def clone(self, copy=None):
        assert copy is None
        copy = self.__class__(num_classes=self.num_classes, lr=self.lr, resnet_version=self.resnet_version)
        return super(ResNetPLModel, self).clone(copy=copy)

