import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import EbayDataset, EbayDatasetSiamse
from utils import train_valid_split, get_basic_augmentations


class EbayDataModule(pl.LightningDataModule):
    def __init__(self, config_data):
        super().__init__()

        transforms = get_basic_augmentations()
        if not config_data["train_data"]["contrastive"]:
            self.train_ds, self.val_ds = train_valid_split(EbayDataset(config_data["train_data"], transforms))
        else:
            self.train_ds, self.val_ds = train_valid_split(EbayDatasetSiamse(config_data["train_data"], transforms))

        if not config_data["test_data"]["contrastive"]:
            self.test_ds = EbayDataset(config_data["test_data"], transforms)
        else:
            self.test_ds = EbayDatasetSiamse(config_data["test_data"], transforms)
        self.batch_size = config_data["model"]["batch_size"]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, pin_memory=True)
