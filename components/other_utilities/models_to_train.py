import torch
import torch.nn as nn
from torchmetrics.classification import AUROC
from torchvision import models
from components.FL_sim import FederatedModelWrapper


class ResNetPLModel(FederatedModelWrapper):
    def __init__(self, num_classes, resnet_version='resnet50', lr=0.01,
                 logging_disabled=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.resnet_version = resnet_version
        self.num_classes = num_classes
        self.logging_disabled = logging_disabled

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

    def step_with_custom_logs(self, stage: str, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        auc = None
        if stage is None or not self.logging_disabled:
            auc = self.aucroc(torch.softmax(logits.detach(), dim=1), y)

        if not self.logging_disabled:
            self.log(f"{stage}_loss", loss, on_step=True, prog_bar=True, logger=True)
            self.log(f"{stage}_auc", auc,
                     on_step=True, prog_bar=True, logger=True)

        return loss, auc

    def training_step(self, batch, batch_idx):
        super(ResNetPLModel, self).training_step(batch, batch_idx)

        loss, auc = self.step_with_custom_logs('train', batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.logging_disabled:
            return 0
        loss, auc = self.step_with_custom_logs('valid', batch, batch_idx)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.lr, momentum=0, weight_decay=1e-4)
        return optimizer

    def clone(self, copy=None):
        new_model = ResNetPLModel(num_classes=self.num_classes, lr=self.lr,
                                  resnet_version=self.resnet_version,
                                  logging_disabled=self.logging_disabled)

        return super(ResNetPLModel, self).clone(copy=new_model)
