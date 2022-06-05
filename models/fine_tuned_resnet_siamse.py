from pytorch_lightning import LightningModule
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                    (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive

class FineTunedResnetSiamse(LightningModule):
    def __init__(self, config):
        super(FineTunedResnetSiamse, self).__init__()
        self.classes = config["model"]["num_classes"]
        self.loss_fn = ContrastiveLoss()
        self.model =  models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 64)
        self.lr = config["model"]["lr"]


    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, dataset_idx):
        return self.step(batch, dataset_idx, "train")

    def validation_step(self, batch, dataset_idx):
        return self.step(batch, dataset_idx, "val")

    def test_step(self, batch, dataset_idx):
        return self.step(batch, dataset_idx, "test")

    def step(self, batch, dataset_idx, type):
        data1, data2, same_class = batch
        output1 = self(data1)
        output2 = self(data2)
        loss = self.loss_fn(output1, output2, same_class)
        self.log(f"{type}_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
