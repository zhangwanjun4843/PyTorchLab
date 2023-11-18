from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid, save_image


class GANCallback(Callback):
    def __init__(
        self,
        latent_dim: int,
        nums: int = 8,
        tag: str = "gan_images",
        save_on_tensorboard: bool = True,
        save_on_directory: bool = True,
        **kwargs,
    ):
        """generate images on GAN

        Args:
            latent_dim(int): dimension of latent code
            nums (int, optional): number of images. Defaults to 8.
            tag(str, optional): tag name of directory for saving images. Default to "gan_images".
            save_on_tensorboard(bool, optional): save image on tensorboard. Default to True.
            save_on_directory(bool, optional): save image on directory. Default to True.
            **kwargs: Other arguments are documented in make_grid
        """
        self.latent_dim = latent_dim
        self.nums = nums
        self.tag = tag
        self._save_on_tensorboard = save_on_tensorboard
        self._save_on_directory = save_on_directory
        self.kwargs = kwargs

    def generate_images(
        self, pl_module: LightningModule, train: bool = True
    ) -> torch.Tensor:
        # sample noise with shape (batch, latent_dim)
        z = torch.randn(self.nums, self.latent_dim).to(pl_module.device)
        # generate images
        if train:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module(z)
                pl_module.train()
        else:
            images = pl_module(z)
        return make_grid(images, **self.kwargs)

    def save_images(
        self, trainer: Trainer, pl_module: LightningModule, images: torch.Tensor
    ) -> None:
        if self._save_on_tensorboard:
            self.save_on_tensorboard(
                pl_module.logger.experiment, images, trainer.global_step
            )
        if self._save_on_directory:
            save_path = Path(pl_module.logger.log_dir) / self.tag
            save_path.mkdir(exist_ok=True, parents=True)
            self.save_on_directory(images, trainer.current_epoch, save_path)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        images = self.generate_images(pl_module, train=True)
        self.save_images(trainer, pl_module, images)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        images = self.generate_images(pl_module, train=False)
        self.save_images(trainer, pl_module, images)

    def save_on_tensorboard(
        self, tb: SummaryWriter, images: torch.Tensor, global_step: int
    ):
        tb.add_image(
            tag=self.tag,
            img_tensor=images,
            global_step=global_step,
        )

    def save_on_directory(self, images: torch.Tensor, epoch: int, path: Path):
        save_name = path / f"epoch_{epoch}.png"
        save_image(images, save_name)
