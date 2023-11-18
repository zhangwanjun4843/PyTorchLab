import numpy as np
import torch
from torch import nn

from pytorchlab.utils.type_hint import ImageShape, IntList, ModuleList


def linear_layer(
    in_features: int,
    out_features: int,
    norm: bool = True,
) -> ModuleList:
    """
    linear layer with linear, batchnorm and leakyrelu

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        norm (bool, optional): batchnorm or not. Defaults to True.

    Returns:
        ModuleList: list of modules to realize linear layer
    """
    layers: ModuleList = [nn.Linear(in_features, out_features)]
    if norm:
        layers.append(nn.BatchNorm1d(out_features))
    layers.append(nn.ReLU(inplace=True))
    return layers


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_shape: ImageShape,
        hidden_layers: IntList,
    ):
        """
        generate image from latent code

        Args:
            latent_dim (int): dimension of latent code
            out_shape (ImageShape): shape of output image
            hidden_layers (IntList): define hidden layers.
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.out_shape = out_shape
        self.out_features = int(np.prod(out_shape))

        layers: ModuleList = linear_layer(self.latent_dim, hidden_layers[0], norm=False)
        for i in range(len(hidden_layers) - 1):
            layers.extend(linear_layer(hidden_layers[i], hidden_layers[i + 1]))
        layers.extend([nn.Linear(hidden_layers[-1], self.out_features), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        x = x.view(x.size(0), *self.out_shape)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_shape: ImageShape,
        hidden_layers: IntList,
    ):
        """
        discriminate whether image is generated or groundtruth

        Args:
            in_shape (ImageShape): shape of input image
            hidden_layers (ChannelList, optional): define hidden layers. Defaults to [128,256,512,1024].
        """
        super().__init__()
        self.in_shape = in_shape
        self.in_features: int = int(np.prod(in_shape))

        layers = linear_layer(self.in_features, hidden_layers[0], norm=False)
        for i in range(len(hidden_layers) - 1):
            layers.extend(
                linear_layer(hidden_layers[i], hidden_layers[i + 1], norm=False)
            )
        layers.extend([nn.Linear(hidden_layers[-1], 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x
