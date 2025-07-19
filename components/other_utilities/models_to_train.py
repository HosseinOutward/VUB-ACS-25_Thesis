import torch
import torch.nn as nn
from torchmetrics.classification import AUROC
from torchvision import models
from components.FL_sim import FederatedModelWrapper


class ResNetPLModel(FederatedModelWrapper):
    def __init__(self, num_classes, resnet_version='resnet50', lr=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.resnet_version = resnet_version
        self.num_classes = num_classes

        # 1) backbone ----------------------------------------------------------
        backbone = dict(resnet50=models.resnet50,
                        resnet18=models.resnet18)[resnet_version](weights=None)

        # 2) replace the classification head ----------------------------------
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.model = backbone

        # 3) loss & metric -----------------------------------------------------
        self.loss_fn = nn.CrossEntropyLoss()
        self.aucroc = AUROC(num_classes=num_classes, average="weighted", task="multiclass")

    def forward(self, x):
        return self.model(x)

    def get_loss_etc(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        auc = self.aucroc(torch.softmax(logits.detach(), dim=1), y)

        etc = (auc,)
        return loss, etc

    def clone(self, copy=None):
        new_model = ResNetPLModel(num_classes=self.num_classes, lr=self.lr, resnet_version=self.resnet_version)
        return super(ResNetPLModel, self).clone(copy=new_model)