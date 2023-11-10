from typing import Literal

import torch
from matplotlib.figure import Figure
from torchmetrics import (ROC, Accuracy, F1Score, Metric, Precision, Recall,
                          Specificity)


class ClassifyMetrics:
    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: int | None = None,
    ) -> None:
        """Metric some value in classify tasks."""
        self.device = torch.device("cpu")
        self.metrics_func: dict[str, Metric] = {
            "precision": Precision(task=task, num_classes=num_classes),
            "recall": Recall(task=task, num_classes=num_classes),
            "speicificity": Specificity(task=task, num_classes=num_classes),
            "f1score": F1Score(task=task, num_classes=num_classes),
            "accuracy": Accuracy(task=task, num_classes=num_classes),
            "roc": ROC(task=task, num_classes=num_classes),
        }

    def to(self, device):
        for k in self.metrics_func.keys():
            self.metrics_func[k] = self.metrics_func[k].to(device)
        self.device = device

    def compute_on_batch(self, preds: torch.Tensor, target: torch.Tensor):
        if self.device != preds.device:
            self.to(preds.device)
        for v in self.metrics_func.values():
            v.update(preds, target)

    def compute_on_epoch(self) -> tuple[dict[str, float], Figure]:
        ans = {}
        for k, v in self.metrics_func.items():
            ans.update({k: v.compute()})
        ans.pop("roc")
        fig_, _ = self.metrics_func["roc"].plot(score=True)
        self.reset()
        return ans, fig_

    def reset(self):
        for v in self.metrics_func.values():
            v.reset()
