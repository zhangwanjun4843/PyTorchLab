from typing import Callable

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from pytorchlab.datamodules.basic import BasicDataModule


class ImageFolderDataModule(BasicDataModule):
    def entire_train_dataset(
        self, transforms: Callable[[torch.Tensor], torch.Tensor]
    ) -> Dataset:
        return torchvision.datasets.ImageFolder(
            self.train_root.as_posix(), transform=transforms
        )

    def entire_test_dataset(
        self, transforms: Callable[[torch.Tensor], torch.Tensor]
    ) -> Dataset:
        return torchvision.datasets.ImageFolder(
            self.test_root.as_posix(), transform=transforms
        )

    def default_transforms(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose([transforms.ToTensor()])


class VisionDataModule(BasicDataModule):
    def prepare_data(self) -> None:
        """Saves files to root."""
        self.dataset_cls(self.train_root, train=True, download=True)
        self.dataset_cls(self.test_root, train=False, download=True)

    def entire_train_dataset(self, transforms) -> Dataset:
        return self.dataset_cls(self.train_root, train=True, transform=transforms)

    def entire_test_dataset(self, transforms) -> Dataset:
        return self.dataset_cls(self.test_root, train=False, transform=transforms)

    def default_transforms(self):
        return transforms.Compose([transforms.ToTensor()])


class MNISTDataModule(VisionDataModule):
    dataset_cls = torchvision.datasets.MNIST


class FashionMNISTDataModule(VisionDataModule):
    dataset_cls = torchvision.datasets.FashionMNIST


class CIFAR10DataModule(VisionDataModule):
    dataset_cls = torchvision.datasets.CIFAR10


class CIFAR100DataModule(VisionDataModule):
    dataset_cls = torchvision.datasets.CIFAR100
