from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except Exception as e:  # pragma: no cover
    datasets = None
    transforms = None


@dataclass
class MNISTLoaders:
    train: DataLoader
    test: DataLoader


def _mnist_transform(binarized: bool) -> "transforms.Compose":
    assert transforms is not None, "torchvision is required to load MNIST"
    tfms = [transforms.ToTensor()]  # -> [0,1], shape (1,28,28)
    if binarized:
        tfms.append(transforms.Lambda(lambda x: (x > 0.5).float()))
    return transforms.Compose(tfms)


def get_mnist_loaders(
    batch_size: int,
    binarized: bool,
    root: str = "data",
    num_workers: int = 0,
) -> MNISTLoaders:
    """Return train/test loaders for MNIST.

    - If `binarized=True`, pixels are thresholded at 0.5 as required in Part A.
    - If `binarized=False`, returns standard MNIST in [0,1] as required in Part B.
    """
    if datasets is None or transforms is None:
        raise ImportError("torchvision is not available; please install torchvision.")

    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=_mnist_transform(binarized))
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=_mnist_transform(binarized))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return MNISTLoaders(train=train_loader, test=test_loader)


def to_minus1_1(x01: torch.Tensor) -> torch.Tensor:
    """Convert [0,1] tensor to [-1,1]."""
    return x01 * 2.0 - 1.0
