from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, random_split


def get_splits(
    len_dataset: int,
    val_split: int | float,
) -> List[int]:
    """Computes split lengths for train and validation set."""
    if isinstance(val_split, int):
        train_len = len_dataset - val_split
        splits = [train_len, val_split]
    elif isinstance(val_split, float):
        val_len = int(val_split * len_dataset)
        train_len = len_dataset - val_len
        splits = [train_len, val_len]
    else:
        raise ValueError(f"Unsupported type {type(val_split)}")

    return splits


def split_dataset(
    dataset: Dataset,
    val_split: int | float,
    seed: int,
    train: bool,
) -> Dataset:
    """Splits the dataset into train and validation set."""
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, val_split)
    dataset_train, dataset_val = random_split(
        dataset, splits, generator=torch.Generator().manual_seed(seed)
    )
    if train:
        return dataset_train
    return dataset_val


def get_mean_std_value(dataloader: DataLoader):
    """Calculate mean and std of image dataloader"""
    mean_list = []
    std_list = []

    for data, _ in dataloader:
        mean_list.append(torch.mean(data, dim=[0, 2, 3]))
        std_list.append(torch.mean(data**2, dim=[0, 2, 3]))
    mean = sum(mean_list) / len(mean_list)
    std = (sum(std_list) / len(std_list) - mean**2) ** 0.5
    return [x.item() for x in mean], [x.item() for x in std]
