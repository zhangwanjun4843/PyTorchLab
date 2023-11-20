import torch
from torch import nn

from pytorchlab.models.common_modules import conv2d_block, deconv2d_block


def init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_shape: tuple[int, int, int],
        channel_list: list[int],
    ):
        """
        generate image from latent code

        output H = W = 2 ^ (len(channel_list) + 1)

        Args:
            latent_dim (int): dimension of latent code
            out_channels (int): channels of output image
            channel_list (list[int]): channels of hidden layers.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.channels, self.height, self.width = out_shape
        self.channel_list = channel_list
        self.num_blocks = len(channel_list)
        assert self.num_blocks > 0, "length of channel_list must be greater than 0"

        self.height_pred = self.height // 2**self.num_blocks
        self.width_pred = self.width // 2**self.num_blocks
        assert (
            self.height_pred >= 4 and self.width_pred >= 4
        ), "channel_list is too deep"

        self.linear = nn.Linear(
            latent_dim, self.height_pred * self.width_pred * channel_list[0]
        )

        layers: list[nn.Module] = []
        for i in range(self.num_blocks - 1):
            layers.extend(deconv2d_block(channel_list[i], channel_list[i + 1], 4, 2, 1))
        layers.extend(
            deconv2d_block(
                channel_list[-1], self.channels, 4, 2, 1, activation=nn.Sigmoid()
            )
        )
        self.net = nn.Sequential(*layers)

        init_weights(self.net)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x.view(x.size(0), self.channel_list[0], self.height_pred, self.width_pred)
        x = self.net(x)
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
        self.in_shape = in_shape
        self.channels, self.height, self.width = in_shape
        self.channel_list = channel_list
        self.num_blocks = len(channel_list)
        assert self.num_blocks > 0, "length of channel_list must be greater than 0"

        self.height_pred = self.height // 2**self.num_blocks
        self.width_pred = self.width // 2**self.num_blocks
        assert (
            self.height_pred >= 4 and self.width_pred >= 4
        ), "channel_list is too deep"

        layers = conv2d_block(self.channels, channel_list[0], 4, 2, 1, norm=False)
        for i in range(len(channel_list) - 1):
            layers.extend(conv2d_block(channel_list[i], channel_list[i + 1], 4, 2, 1))
        self.net = nn.Sequential(*layers)
        init_weights(self.net)

        self.linear = nn.Sequential(
            nn.Linear(
                channel_list[-1] * self.height_pred * self.width_pred,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
