from abc import abstractmethod
from pathlib import Path
from typing import Callable

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from pytorchlab.utils.data import split_dataset


class BasicDataModule(LightningDataModule):
    def __init__(
        self,
        root: str = "dataset",
        val_split: int | float = 0.2,
        num_workers: int = 4,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        transforms: Callable = None,
        train_transforms: Callable = None,
        valid_transforms: Callable = None,
        test_transforms: Callable = None,
    ) -> None:
        """
        Args:
            root: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
            transforms: default transformations
            train_transforms: transformations applied to train dataset
            val_transforms: transformations applied to validation dataset
            test_transforms: transformations applied to test dataset
        """

        super().__init__()

        self.root = Path(root)
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._default_transforms = transforms
        self._train_transforms = train_transforms
        self._valid_transforms = valid_transforms
        self._test_transforms = test_transforms

    @abstractmethod
    def entire_train_dataset(self, transforms) -> Dataset:
        """Entire dataset used to train and validate.

        Returns:
            Dataset: train/val dataset
        """

    @abstractmethod
    def entire_test_dataset(self, transforms) -> Dataset:
        """Entire dataset used to test.

        Returns:
            Dataset: test dataset
        """

    def default_transforms(self):
        """Default transform for the dataset."""
        return lambda x: x

    @property
    def transforms(self):
        """transforms (or collection of transforms) applied to train/val/test dataset."""
        return self._default_transforms or self.default_transforms()

    @property
    def train_transforms(self):
        """transforms (or collection of transforms) applied to train dataset."""
        return self._train_transforms or self.transforms

    @property
    def valid_transforms(self):
        """transforms (or collection of transforms) applied to validation dataset."""
        return self._valid_transforms or self.transforms

    @property
    def test_transforms(self):
        """transforms (or collection of transforms) applied to test dataset."""
        return self._test_transforms or self.transforms

    def setup(self, stage: [str] = None):
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            dataset_train = self.entire_train_dataset(transforms=self.train_transforms)
            dataset_valid = self.entire_train_dataset(transforms=self.valid_transforms)

            # Split
            self.dataset_train = split_dataset(
                dataset=dataset_train,
                val_split=self.val_split,
                seed=self.seed,
                train=True,
            )
            self.dataset_valid = split_dataset(
                dataset=dataset_valid,
                val_split=self.val_split,
                seed=self.seed,
                train=False,
            )

        if stage == "test" or stage is None:
            self.dataset_test = self.entire_test_dataset(
                transforms=self.test_transforms
            )

    def train_dataloader(self):
        """The train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self):
        """The val dataloader."""
        return self._data_loader(self.dataset_valid)

    def test_dataloader(self):
        """The test dataloader."""
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
