from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device(requested: str = "auto") -> str:
    """
    Pick a reasonable torch device.

    On an Intel Mac, MPS is usually unavailable, so this will generally fall
    back to CPU.
    """
    requested = requested.lower()

    if requested in {"cpu", "mps", "cuda"}:
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return requested

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: str) -> torch.nn.Module:
    path = Path(path)
    state = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def get_latent_means(
    model: torch.nn.Module,
    data_loader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return latent posterior means and labels from a dataloader.

    Output:
      z_means: [N, latent_dim]
      labels:  [N]
    """
    model.eval()
    all_z = []
    all_y = []

    for x, y in data_loader:
        x = x.to(device)
        z_mean = model.encode_mean(x)
        all_z.append(z_mean.cpu())
        all_y.append(y.cpu())

    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0)


@torch.no_grad()
def get_dataset_tensors_from_loader(data_loader) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for x, y in data_loader:
        xs.append(x.cpu())
        ys.append(y.cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def select_fixed_point_pairs(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_pairs: int = 25,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Select fixed test point pairs using indices into the test set.
    """
    if len(images) != len(labels):
        raise ValueError("images and labels must have same length.")
    if 2 * num_pairs > len(images):
        raise ValueError("Not enough images to form the requested number of pairs.")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(images), generator=g)

    start_idx = perm[:num_pairs]
    end_idx = perm[num_pairs:2 * num_pairs]

    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "pair_labels_start": labels[start_idx],
        "pair_labels_end": labels[end_idx],
    }


def materialize_pairs_from_latents(
    latent_means: torch.Tensor,
    pair_index_dict: dict[str, torch.Tensor],
    device: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    starts = latent_means[pair_index_dict["start_idx"]]
    ends = latent_means[pair_index_dict["end_idx"]]

    pairs = []
    for s, e in zip(starts, ends):
        pairs.append((s.to(device), e.to(device)))
    return pairs


def checkpoint_path(
    experiment_dir: str | Path,
    name: str = "model.pt",
) -> Path:
    experiment_dir = ensure_dir(experiment_dir)
    return experiment_dir / name
