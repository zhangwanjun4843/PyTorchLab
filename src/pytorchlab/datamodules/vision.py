from typing import Any, Callable

from torch.utils.data import Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import transforms

from pytorchlab.datamodules.basic import BasicDataModule


class ImageFolderDataModule(BasicDataModule):
    def entire_train_dataset(self, transforms: Callable[..., Any]) -> Dataset:
        return ImageFolder((self.root / "train").as_posix(), transform=transforms)

    def entire_test_dataset(self, transforms: Callable[..., Any]) -> Dataset:
        return ImageFolder((self.root / "test").as_posix(), transform=transforms)

    def default_transforms(self) -> Callable[..., Any]:
        return transforms.Compose([transforms.ToTensor()])


class VisionDataModule(BasicDataModule):
    name: str
    dataset_cls: type
    dim: tuple
    norm_mean: tuple
    norm_std: tuple

    def prepare_data(self) -> None:
        """Saves files to root."""
        self.dataset_cls(self.root, train=True, download=True)
        self.dataset_cls(self.root, train=False, download=True)

    def entire_train_dataset(self, transforms) -> Dataset:
        return self.dataset_cls(self.root, train=True, transform=transforms)

    def entire_test_dataset(self, transforms) -> Dataset:
        return self.dataset_cls(self.root, train=False, transform=transforms)

    def default_transforms(self):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.norm_mean, self.norm_std)]
        )


class MNISTDataModule(VisionDataModule):
    name = "MNIST"
    dataset_cls = MNIST
    dim = (1, 28, 28)
    norm_mean = (0.1307,)
    norm_std = (0.3081,)
