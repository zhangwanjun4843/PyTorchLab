import torch
from torch import nn

from pytorchlab.utils.type_hint import ImageShape, IntList, ModuleList


def deconv_block(
    in_features: int,
    out_features: int,
    norm: bool = True,
) -> ModuleList:
    """upsample image with scale 2

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        norm (bool, optional): batchnorm or not. Defaults to True.

    Returns:
        ModuleList: list of modules to realize linear layer
    """
    layers: ModuleList = [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_features, out_features, 3, 1, 1),
    ]
    if norm:
        layers.append(nn.BatchNorm2d(out_features))
    layers.append(nn.ReLU(inplace=True))
    return layers


def conv_block(
    in_features: int,
    out_features: int,
    norm: bool = True,
) -> ModuleList:
    """downsample image with scale 2

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        norm (bool, optional): batchnorm or not. Defaults to True.

    Returns:
        ModuleList: list of modules to realize linear layer
    """
    layers: ModuleList = [nn.Conv2d(in_features, out_features, 3, 2, 1)]
    if norm:
        layers.append(nn.BatchNorm2d(out_features))
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

        if len(out_shape) == 2:
            self.channels = 1
        else:
            self.channels = out_shape[0]
        self.height, self.width = out_shape[1:]

        self.num_blocks = len(hidden_layers)
        self.hidden_layers = hidden_layers

        self.linear = nn.Linear(
            latent_dim,
            hidden_layers[0]
            * (self.width // 2**self.num_blocks)
            * (self.height // 2**self.num_blocks),
        )

        layers: ModuleList = [
            nn.BatchNorm2d(hidden_layers[0]),
            *deconv_block(hidden_layers[0], hidden_layers[0]),
        ]
        for i in range(len(hidden_layers) - 1):
            layers.extend(deconv_block(hidden_layers[i], hidden_layers[i + 1]))
        layers.extend([nn.Conv2d(hidden_layers[-1], self.channels, 3, 1, 1), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x.view(
            x.size(0),
            self.hidden_layers[0],
            self.height // 2**self.num_blocks,
            self.width // 2**self.num_blocks,
        )
        x = self.net(x)
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
        if len(in_shape) == 2:
            self.channels = 1
        else:
            self.channels = in_shape[0]
        self.height, self.width = in_shape[1:]

        self.num_blocks = len(hidden_layers)

        layers = conv_block(self.channels, hidden_layers[0], norm=False)
        for i in range(len(hidden_layers) - 1):
            layers.extend(conv_block(hidden_layers[i], hidden_layers[i + 1]))
        self.net = nn.Sequential(*layers)

        self.linear = nn.Sequential(
            nn.Linear(
                hidden_layers[-1]
                * (self.height // 2**self.num_blocks)
                * (self.width // 2**self.num_blocks),
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
