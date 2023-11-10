from pathlib import Path
from typing import Any, Literal

import torch
import torchvision
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from pytorchlab.metrics.classify import ClassifyMetrics
from pytorchlab.utils.common import get_json_value


class ClassifyCallback(Callback):
    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: int | None,
    ) -> None:
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.metrics = ClassifyMetrics(task=task, num_classes=num_classes)

    def compute_on_batch(
        self,
        mode: Literal["train", "val", "test"],
        pl_module: LightningModule,
        batch,
        outputs,
    ):
        preds = get_json_value(outputs, "preds", None)
        if preds is None:
            return
        _, y = batch
        self.metrics.compute_on_batch(preds, y)

    def compute_on_epoch(self, mode: Literal["train", "val", "test"], pl_module):
        metrics, fig_ = self.metrics.compute_on_epoch()
        pl_module.log_dict(metrics, sync_dist=True)
        log_path = Path(pl_module.logger.log_dir) / "metrics"
        log_path.mkdir(exist_ok=True, parents=True)
        fig_.savefig(log_path / f"roc_epoch={pl_module.current_epoch}_{mode}.jpg")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.compute_on_batch("val", pl_module, batch, outputs)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.compute_on_epoch("val", pl_module)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.compute_on_batch("test", pl_module, batch, outputs)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.compute_on_epoch("test", pl_module)


class ClassifyPredictCallback(Callback):
    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
    ) -> None:
        super().__init__()
        self.task = task

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x = get_json_value(outputs, "input", None)
        y = get_json_value(outputs, "target", None)
        preds = get_json_value(outputs, "preds", None)
        if None in [x, y, preds]:
            return
        for index in range(x.shape[0]):
            _input = x[index]
            _target = y[index]
            _preds = preds[index]
            if self.task == "multiclass":
                _preds = torch.argmax(_preds)
            log_path = (
                Path(pl_module.logger.log_dir) / "predict" / f"{self.predict_number}"
            )
            self.predict_number += 1
            log_path.mkdir(exist_ok=True, parents=True)
            torchvision.utils.save_image(_input, log_path / "image.jpg")
            info_str = f"label={_target}\nprediction={_preds}"
            with open((log_path / "prediction.txt"), "w", encoding="utf-8") as f:
                f.write(info_str)
