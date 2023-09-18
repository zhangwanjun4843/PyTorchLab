from pathlib import Path
from typing import Tuple

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid, save_image


class TensorBoardGANSampler(Callback):
    def __init__(
        self,
        save_dir: str = None,
        num_samples: int = 9,
        nrow: int = 3,
        padding: int = 2,
        normalize: bool = False,
        value_range: Tuple[int, int] | None = None,
        scale_each: bool = False,
        pad_value=0,
        not_save_files: bool = False,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.value_range = value_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.save_dir = save_dir
        self.not_save_files = not_save_files

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dim = (self.num_samples, pl_module.hparams.latent_dim)
        z = torch.normal(mean=0.0, std=1.0, size=dim).to(device=pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            output = pl_module(z)
            pl_module.train()
        grid = make_grid(
            output,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.value_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        save_name = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(save_name, grid, pl_module.global_step)
        if not self.not_save_files:
            save_dir = Path(trainer.logger.log_dir) / save_name
            if self.save_dir is not None:
                save_dir = Path(self.save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            self.save_files(save_dir, grid, epoch=pl_module.current_epoch)

    def save_files(self, save_dir, grid, epoch: int):
        save_path = save_dir / f"epoch={epoch}.png"
        save_image(grid, save_path)
