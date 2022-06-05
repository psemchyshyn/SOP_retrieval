import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models import FineTunedResnetCE, FineTunedResnetArc, FineTunedResnetSiamse
from datamodule import EbayDataModule
from utils import parse_config


if __name__ == "__main__":
    config = parse_config("./config.yaml")
    dm = EbayDataModule(config)
    model = FineTunedResnetSiamse(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints_contrastive",
        filename="model_contrastive-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    estopping_callback = EarlyStopping(monitor="val_loss", patience=3)

    trainer = Trainer(deterministic=True,
                    progress_bar_refresh_rate=20,
                    max_epochs=50,
                    callbacks=[checkpoint_callback, estopping_callback])

    trainer.fit(model, dm)
