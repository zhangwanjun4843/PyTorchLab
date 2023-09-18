from typing import Tuple

import numpy as np
import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        feats = int(np.prod(image_shape))
        self.image_shape = image_shape
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, feats)

    def forward(self, z: torch.Tensor):
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        img = torch.tanh(self.fc4(z))
        return img.view(img.size(0), *self.image_shape)


class Discriminator(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dim: int = 1024,
    ):
        super().__init__()
        feats = int(np.prod(image_shape))
        self.image_shape = image_shape
        self.fc1 = nn.Linear(feats, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, image: torch.Tensor):
        x = image.view(image.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return F.sigmoid(self.fc4(x))


class GANModule(LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator = self.init_generator(latent_dim, image_shape)
        self.discriminator = self.init_discriminator(image_shape)
        self.automatic_optimization = False

    def init_generator(self, latent_dim, image_shape):
        return Generator(latent_dim, image_shape)

    def init_discriminator(self, image_shape):
        return Discriminator(image_shape)

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return opt_g, opt_d

    def training_step(self, batch, batch_idx):
        x, _ = batch
        opt_g, opt_d = self.optimizers()
        z = torch.randn(x.size(0), self.hparams.latent_dim).to(device=self.device)
        real = torch.ones(x.size(0), 1).to(device=self.device)
        fake = torch.zeros(x.size(0), 1).to(device=self.device)
        # generate image
        image_fake = self(z)
        # train generator
        opt_g.zero_grad()
        loss_g = F.binary_cross_entropy(self.discriminator(image_fake), real)
        self.log_dict(
            {"g_loss": loss_g},
            on_epoch=True,
            prog_bar=True,
        )
        loss_g.backward()
        opt_g.step()
        # train discriminator
        opt_d.zero_grad()
        loss_d_real = F.binary_cross_entropy(self.discriminator(x), real)
        loss_d_fake = F.binary_cross_entropy(
            self.discriminator(image_fake.detach()), fake
        )
        loss_d = loss_d_real + loss_d_fake
        self.log_dict(
            {"d_loss": loss_d},
            on_epoch=True,
            prog_bar=True,
        )
        loss_d.backward()
        opt_d.step()
        self.log_dict(
            {
                "g_lr": opt_g.optimizer.param_groups[0]["lr"],
                "d_lr": opt_d.optimizer.param_groups[0]["lr"],
            },
            prog_bar=True,
        )
