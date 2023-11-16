from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.nn import functional as F


class LeNet5(LightningModule):
    def __init__(self):
        """
        LeNet-5 model architechture

        Article: Gradient-based learning applied to document recognition
        DOI: https://doi.org/10.1109/5.726791
        Input: (batch,1,28,28)
        Output: (batch,10)
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x).view(x.shape[0], -1))

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        return {"loss": loss, "preds": pred}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        return {"loss": loss, "preds": pred}

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        return {"preds": pred}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        pred = self(x)
        return {"input": x, "target": y, "preds": pred}


def cli_main():
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
    from pytorchlab.models.lenet import LeNet5
    from pytorchlab.datamodules.vision import FashionMNISTDataModule, MNISTDataModule
    from pytorchlab.callbacks.loss import LossCallback
    from pytorchlab.callbacks.classify import ClassifyCallback,ClassifyPredictCallback

    logger = TensorBoardLogger(save_dir="logs", name="lenet5")
    callbacks = [
        ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
        ),
        LossCallback(),
        ClassifyCallback(task="multiclass", num_classes=10),
        ClassifyPredictCallback(task="multiclass"),
    ]
    trainer_defaults = {
        "devices": [0],
        "max_epochs": 10,
        "logger": logger,
        "callbacks": callbacks,
    }
    cli = LightningCLI(
        seed_everything_default=1234,
        trainer_defaults=trainer_defaults,
        model_class=LeNet5,
        datamodule_class=MNISTDataModule,
    )


if __name__ == "__main__":
    cli_main()
