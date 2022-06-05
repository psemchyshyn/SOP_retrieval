from pytorch_lightning import LightningModule
import torchvision.models as models
import torch
import torch.nn as nn
import torchmetrics
from pytorch_metric_learning.losses import ArcFaceLoss


class FineTunedResnetArc(LightningModule):
    def __init__(self, config):
        super(FineTunedResnetArc, self).__init__()
        self.classes = config["model"]["num_classes"]
        self.loss_fn = ArcFaceLoss(self.classes, self.classes)
        self.model =  models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.classes)
        self.lr = config["model"]["lr"]

        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(), torchmetrics.F1Score(), torchmetrics.Precision(), torchmetrics.Recall()])
        self.train_metrics = metrics.clone("train")
        self.val_metrics = metrics.clone("val")
        self.test_metrics = metrics.clone("test")

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, dataset_idx):
        return self.step(batch, dataset_idx, "train")

    def validation_step(self, batch, dataset_idx):
        return self.step(batch, dataset_idx, "val")

    def test_step(self, batch, dataset_idx):
        return self.step(batch, dataset_idx, "test")

    def step(self, batch, dataset_idx, type):
        data, class_id, super_class_id = batch
        output = self(data)
        loss = self.loss_fn(output, super_class_id)

        if type == "train":
            self.log_dict({"train_loss": loss, **self.train_metrics(output, super_class_id)})
        elif type == "val":
            self.log_dict({"val_loss": loss, **self.val_metrics(output, super_class_id)})
        else:
            self.log_dict({"test_loss": loss, **self.test_metrics(output, super_class_id)})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
