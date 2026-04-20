from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def subsample_mnist(
    data: torch.Tensor,
    targets: torch.Tensor,
    num_data: int = 2048,
    num_classes: int = 3,
) -> TensorDataset:
    """
    Keep only the first `num_classes` classes from MNIST and return the first
    `num_data` observations.

    Output images are:
      - shape: [N, 1, 28, 28]
      - dtype: float32
      - scaled to [0, 1]
    """
    mask = targets < num_classes
    x = data[mask][:num_data].unsqueeze(1).to(torch.float32) / 255.0
    y = targets[mask][:num_data]
    return TensorDataset(x, y)


def get_mnist_subset_datasets(
    num_train_data: int = 2048,
    num_test_data: int = 2048,
    num_classes: int = 3,
) -> tuple[TensorDataset, TensorDataset]:
    """
    Return train/test MNIST subsets for classes {0, 1, 2} by default.
    The project specifies non-binarized MNIST.
    """
    _ = transforms.ToTensor()  # kept for clarity; raw tensors are used below

    train_raw = datasets.MNIST(
        root="data/",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_raw = datasets.MNIST(
        root="data/",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_dataset = subsample_mnist(
        train_raw.data,
        train_raw.targets,
        num_data=num_train_data,
        num_classes=num_classes,
    )
    test_dataset = subsample_mnist(
        test_raw.data,
        test_raw.targets,
        num_data=num_test_data,
        num_classes=num_classes,
    )
    return train_dataset, test_dataset


def get_mnist_subset_loaders(
    batch_size: int = 32,
    num_train_data: int = 2048,
    num_test_data: int = 2048,
    num_classes: int = 3,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/test dataloaders.

    num_workers=0 is a safe default on macOS.
    """
    train_dataset, test_dataset = get_mnist_subset_datasets(
        num_train_data=num_train_data,
        num_test_data=num_test_data,
        num_classes=num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


def add_training_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """
    Add small Gaussian noise and clamp back to [0, 1].
    This matches the idea from the handout that mild noise can help Part B.
    """
    eps = std * torch.randn_like(x)
    return torch.clamp(x + eps, min=0.0, max=1.0)
