from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


def get_splits(
    len_dataset: int,
    val_split: int | float,
) -> list[int]:
    """Computes split lengths for train and validation set."""
    assert isinstance(val_split, int | float), "val_split should be int or float type"
    if isinstance(val_split, int):
        train_len = len_dataset - val_split
        splits = [train_len, val_split]
    elif isinstance(val_split, float):
        assert 0 <= val_split <= 1, "val_split in float type should between 0 and 1"
        val_len = int(val_split * len_dataset)
        train_len = len_dataset - val_len
        splits = [train_len, val_len]
    return splits


def split_dataset(
    dataset: Dataset,
    val_split: int | float,
    seed: int,
    train: bool,
) -> Dataset:
    """Splits dataset into train and validation set."""
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, val_split)
    dataset_train, dataset_val = random_split(
        dataset, splits, generator=torch.Generator().manual_seed(seed)
    )
    if train:
        return dataset_train
    return dataset_val


class BasicDataModule(LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self,
        train_root: str | Path,
        test_root: str | Path,
        val_split: int | float = 0.2,
        split_seed: int = 42,
        num_workers: int = 4,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        transforms: Callable[[torch.Tensor], torch.Tensor] = None,
        train_transforms: Callable[[torch.Tensor], torch.Tensor] = None,
        val_transforms: Callable[[torch.Tensor], torch.Tensor] = None,
        test_transforms: Callable[[torch.Tensor], torch.Tensor] = None,
        pred_transforms: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        """abstract class for datamodule with train/val/test/pred dataloader

        train_dataset -> train_dataloader + val_dataloader
        test_dataset -> test_dataloader + pred_dataloader

        Args:
            train_root (str): root path for train dataset
            test_root (str): root path for test dataset
            val_split (int | float, optional): rate or length of second part of splits. Defaults to 0.2.
            num_workers (int, optional): number of workers. Defaults to 4.
            batch_size (int, optional): size for one batch. Defaults to 32.
            split_seed (int, optional): seed for random split dataset. Defaults to 42.
            shuffle (bool, optional): shuffle dataset or not. Defaults to True.
            pin_memory (bool, optional): see more details in torch.utils.data.DataLoader. Defaults to True.
            drop_last (bool, optional): see more details in torch.utils.data.DataLoader. Defaults to False.
            transforms (Callable[[torch.Tensor],torch.Tensor], optional): default transform. Defaults to None.
            train_transforms (Callable[[torch.Tensor],torch.Tensor], optional): train transform. Defaults to None.
            val_transforms (Callable[[torch.Tensor],torch.Tensor], optional): val transform. Defaults to None.
            test_transforms (Callable[[torch.Tensor],torch.Tensor], optional): test transform. Defaults to None.
            pred_transforms (Callable[[torch.Tensor],torch.Tensor], optional): pred transform. Defaults to None.
        """

        super().__init__()

        self.train_root = Path(train_root)
        self.test_root = Path(test_root)
        self.val_split = val_split
        self.split_seed = split_seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._default_transforms = transforms
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self._pred_transforms = pred_transforms

    @abstractmethod
    def entire_train_dataset(
        self, transforms: Callable[[torch.Tensor], torch.Tensor]
    ) -> Dataset:
        """entire dataset for train/val

        Args:
            transforms (Callable[[torch.Tensor],torch.Tensor]): train/val transform

        Returns:
            Dataset: train dataset
        """

    @abstractmethod
    def entire_test_dataset(
        self, transforms: Callable[[torch.Tensor], torch.Tensor]
    ) -> Dataset:
        """entire dataset for test/pred

        Args:
            transforms (Callable[[torch.Tensor],torch.Tensor]): test/pred transform

        Returns:
            Dataset: train dataset
        """

    def default_transforms(self):
        """Default transform for dataset."""
        return lambda x: x

    @property
    def transforms(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """transforms for train/val/test/pred"""
        return self._default_transforms or self.default_transforms()

    @property
    def train_transforms(self):
        """transforms for train"""
        return self._train_transforms or self.transforms

    @property
    def val_transforms(self):
        """transforms for val"""
        return self._val_transforms or self.transforms

    @property
    def test_transforms(self):
        """transforms for test"""
        return self._test_transforms or self.transforms

    @property
    def pred_transforms(self):
        """transforms for pred"""
        return self._pred_transforms or self.transforms

    def setup(self, stage: str):
        """split dataset for different stage

        Args:
            stage (Stage): 'fit', 'validate', 'test', 'predict'
        """
        if stage in ["fit", "validate"]:
            dataset_train = self.entire_train_dataset(transforms=self.train_transforms)
            dataset_val = self.entire_train_dataset(transforms=self.val_transforms)

            # Split
            self.dataset_train = split_dataset(
                dataset=dataset_train,
                val_split=self.val_split,
                seed=self.split_seed,
                train=True,
            )
            self.dataset_val = split_dataset(
                dataset=dataset_val,
                val_split=self.val_split,
                seed=self.split_seed,
                train=False,
            )

        if stage in ["test", "predict"]:
            dataset_test = self.entire_test_dataset(transforms=self.test_transforms)
            dataset_pred = self.entire_test_dataset(transforms=self.pred_transforms)
            # Split
            self.dataset_test = split_dataset(
                dataset=dataset_test,
                val_split=self.val_split,
                seed=self.split_seed,
                train=True,
            )
            self.dataset_pred = split_dataset(
                dataset=dataset_pred,
                val_split=self.val_split,
                seed=self.split_seed,
                train=False,
            )

    def train_dataloader(self):
        """train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self):
        """val dataloader."""
        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        """test dataloader."""
        return self._data_loader(self.dataset_test)

    def predict_dataloader(self):
        """prediction dataloader."""
        return self._data_loader(self.dataset_pred)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
