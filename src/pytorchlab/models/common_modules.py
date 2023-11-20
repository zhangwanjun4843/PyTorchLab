from torch import nn


def autopad(
    in_size: int,
    out_size: int,
    kernel_size: int,
    stride: int,
) -> int:
    """
    automatically calculate padding for convolution

    out_features = (in + 2 * padding - kernel_size) // stride + 1

    Args:
        in_size (int): input size
        out_size (int): output size
        kernel_size (int): kernel size
        stride (int): stride

    Returns:
        int: padding
    """
    return max((out_size - 1) * stride + kernel_size - in_size, 0) // 2


def linear_block(
    in_features: int,
    out_features: int,
    drop_out: float | None = None,
    norm: bool = True,
    activation: nn.Module = nn.ReLU(inplace=True),
) -> list[nn.Module]:
    """
    Linear + (Dropout1d) + (BatchNorm1d) + activation

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        drop_out (float | None, optional): drop_out rate(0.2 to 0.5 is recommand). Defaults to None.
        norm (bool, optional): batchnorm or not. Defaults to True.
        activation (nn.Module, optional): activate function. Defaults to nn.ReLU(inplace=True).

    Returns:
        list[nn.Module]: linear block
    """

    layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
    if isinstance(drop_out, float):
        layers.append(nn.Dropout1d(drop_out))
    if norm:
        layers.append(nn.BatchNorm1d(out_features))
    layers.append(activation)
    return layers


def conv2d_block(
    in_features: int,
    out_features: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    drop_out: float | None = None,
    norm: bool = True,
    activation: nn.Module = nn.LeakyReLU(0.2, inplace=True),
) -> list[nn.Module]:
    """
    Conv2d + (Dropout) + (BatchNorm) + activation

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        kernel_size (int, optional): size of kernel. Defaults to 3.
        stride (int, optional): step for kernel. Defaults to 1.
        padding (int, optional): pad width around image. Defaults to 1.
        drop_out (float | None, optional): dropout rate. Defaults to None.
        norm (bool, optional): batch normalization. Defaults to True.
        activation (nn.Module, optional): activate function. Defaults to nn.LeakyReLU(0.2,inplace=True).

    Returns:
        list[nn.Module]: conv2d block
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_features, out_features, kernel_size, stride, padding)
    ]
    if isinstance(drop_out, float):
        layers.append(nn.Dropout2d(drop_out))
    if norm:
        layers.append(nn.BatchNorm2d(out_features))
    layers.append(activation)
    return layers


def deconv2d_block(
    in_features: int,
    out_features: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    drop_out: float | None = None,
    norm: bool = True,
    activation: nn.Module = nn.ReLU(inplace=True),
) -> list[nn.Module]:
    """
    ConvTranspose2d + (Dropout2d) + (BatchNorm2d) + activation

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        kernel_size (int, optional): size of kernel. Defaults to 3.
        stride (int, optional): step for kernel. Defaults to 1.
        padding (int, optional): pad width around image. Defaults to 1.
        drop_out (float | None, optional): dropout rate. Defaults to None.
        norm (bool, optional): batch normalization. Defaults to True.
        activation (nn.Module, optional): activate function. Defaults to nn.LeakyReLU(0.2,inplace=True).

    Returns:
        list[nn.Module]: conv2d block
    """
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_features, out_features, kernel_size, stride, padding)
    ]
    if isinstance(drop_out, float):
        layers.append(nn.Dropout2d(drop_out))
    if norm:
        layers.append(nn.BatchNorm2d(out_features))
    layers.append(activation)
    return layers
