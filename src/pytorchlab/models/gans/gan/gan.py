from typing import Any, Callable, Iterable

import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim import Optimizer

from .components import Discriminator, Generator

OptimizerCallable = Callable[[Iterable], Optimizer]
LossCallable = Callable[[Iterable], nn.Module]


class GAN(LightningModule):
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        latent_dim: int,
        channel_list_g: list[int] = [128, 256, 512, 1024],
        channel_list_d: list[int] = [512, 256],
        criterion: LossCallable = nn.BCELoss,
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
    ) -> None:
        super().__init__()
        # do not optimize model automatically
        self.automatic_optimization = False
        # init model
        self.latent_dim = latent_dim
        self.generator: nn.Module = Generator(
            latent_dim=latent_dim, out_shape=in_shape, channel_list=channel_list_g
        )
        self.discriminator: nn.Module = Discriminator(
            in_shape=in_shape, channel_list=channel_list_d
        )
        self.criterion = criterion()
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def configure_optimizers(self):
        optimizer_g = self.optimizer_g(self.generator.parameters())
        optimizer_d = self.optimizer_d(self.discriminator.parameters())
        return optimizer_g, optimizer_d

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, _ = batch
        g_loss = self.generator_step(x.size(0))
        d_loss = self.discriminator_step(x)
        self.log_dict(
            {
                "g_loss": g_loss,
                "d_loss": d_loss,
            },
            sync_dist=True,
        )

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        ...

    def generator_step(self, batch_size: int) -> torch.Tensor:
        """generate an image that discriminator regard it as groundtruth

        Args:
            batch_size (int): size for one batch

        Returns:
            torch.Tensor: generator loss
        """
        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        # Generate images
        generated_imgs: torch.Tensor = self(z)
        # ground truth result (all true)
        valid = torch.ones(batch_size, 1).to(self.device)
        g_loss: torch.Tensor = self.criterion(self.discriminator(generated_imgs), valid)
        # loss backward
        self.manual_backward(g_loss)
        # update generator optimizer
        optimizer_g.step()
        return g_loss

    def discriminator_step(self, x: torch.Tensor) -> torch.Tensor:
        """discriminate whether x is ground truth or not

        Args:
            x (torch.Tensor): image tensor

        Returns:
            torch.Tensor: discriminator loss
        """
        batch_size = x.size(0)
        # zero grad discriminator optimizer
        optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
        optimizer_d.zero_grad()
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        # Generate images
        generated_imgs: torch.Tensor = self(z)
        # ground truth/ fake result
        real = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        real_loss = self.criterion(self.discriminator(x), real)
        fake_loss = self.criterion(self.discriminator(generated_imgs), fake)
        d_loss = real_loss + fake_loss
        self.manual_backward(d_loss)
        optimizer_d.step()
        return d_loss
