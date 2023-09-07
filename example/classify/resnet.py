from argparse import ArgumentParser
from typing import Any, Callable

import torch
from lightning.pytorch import LightningModule, Trainer
from torch import nn
from torchmetrics import Accuracy
from torchvision.models import resnet

from pytorchlab import MNISTDataModule


class ResNetModule(LightningModule):
    def __init__(
        self,
        name="resnet18",
        in_channels: int = 1,
        num_classes: int = 1000,
        lr=0.001,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        resnet_cls: Callable[..., resnet.ResNet] = getattr(resnet, name, None)
        assert (
            resnet_cls is not None
        ), f"Cannot find {name} in torchvision.models.resnet."
        self.net = resnet_cls(num_classes=num_classes)
        feature: nn.Conv2d = self.net.conv1
        self.net.conv1 = nn.Conv2d(
            in_channels, feature.out_channels, feature.kernel_size, feature.stride
        )
        self.acc_metric = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x) -> Any:
        return self.net(x)

    @property
    def criterion(self):
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return x, y, pred

    def training_step(self, batch, batch_idx):
        _, y, pred = self._common_step(batch, batch_idx)
        loss = self.criterion(pred, y)
        acc = self.acc_metric(pred, y)
        self.log_dict(
            {
                "loss_train": loss,
                "acc_train": acc,
            },
            prog_bar=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _, y, pred = self._common_step(batch, batch_idx)
        acc = self.acc_metric(pred, y)
        self.log_dict(
            {
                "acc_valid": acc,
            },
            sync_dist=True,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a ResNet")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()
    trainer = Trainer(devices=args.devices)
    datamodule = MNISTDataModule()
    model = ResNetModule(num_classes=10)
    trainer = Trainer(devices=args.devices, max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)
