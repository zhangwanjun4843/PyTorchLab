from typing import Any, Literal

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class LossCallback(Callback):
    def __init__(self) -> None:
        """Record loss of train/val/test stage.Json format outputs are required, like```{"loss",loss}```."""
        super().__init__()

    def record_loss(self, mode: Literal["train", "val"], pl_module, outputs):
        if not isinstance(outputs, dict):
            return
        loss = outputs.get("loss", None)
        if loss is not None:
            pl_module.log_dict(
                {f"{mode}_loss": outputs["loss"]},
                sync_dist=True,
                on_epoch=True,
                prog_bar=True,
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.record_loss("train", pl_module, outputs)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.record_loss("val", pl_module, outputs)
