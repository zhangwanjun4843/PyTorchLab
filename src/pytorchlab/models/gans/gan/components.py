import numpy as np
import torch
from torch import nn

from pytorchlab.models.common_modules import linear_block


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_shape: tuple[int, int, int],
        channel_list: list[int],
    ):
        """
        generate image from latent code

        Args:
            latent_dim (int): dimension of latent code
            out_shape (tuple[int,int,int]): shape of output image
            channel_list (list[int]): channels of hidden layers.
        """
        super().__init__()
        assert len(channel_list) > 0, "length of channel_list must be greater than 0"
        self.latent_dim = latent_dim

        self.out_shape = out_shape
        self.out_features = int(np.prod(out_shape))

        layers: list[nn.Module] = linear_block(
            self.latent_dim, channel_list[0], norm=False
        )
        for i in range(len(channel_list) - 1):
            layers.extend(linear_block(channel_list[i], channel_list[i + 1]))
        layers.extend([nn.Linear(channel_list[-1], self.out_features), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        x = x.view(x.size(0), *self.out_shape)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        channel_list: list[int],
    ):
        """
        discriminate whether image is generated or groundtruth

        Args:
            in_shape (tuple[int,int,int]): shape of input image
            channel_list (ChannelList, optional): channels of hidden layers. Defaults to [128,256,512,1024].
        """
        super().__init__()
        assert len(channel_list) > 0, "length of channel_list must be greater than 0"
        self.in_shape = in_shape
        self.in_features: int = int(np.prod(in_shape))

        layers = linear_block(self.in_features, channel_list[0], norm=False)
        for i in range(len(channel_list) - 1):
            layers.extend(
                linear_block(channel_list[i], channel_list[i + 1], norm=False)
            )
        layers.extend([nn.Linear(channel_list[-1], 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x
