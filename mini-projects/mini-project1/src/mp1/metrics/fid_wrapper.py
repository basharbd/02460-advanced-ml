from __future__ import annotations
from pathlib import Path
import torch

# Import the instructor-provided function from repo root
from fid import compute_fid  # noqa: F401

def compute_fid_mnist(
    x_real_01: torch.Tensor,
    x_gen_01: torch.Tensor,
    device: str = "cpu",
    ckpt_path: str = "mnist_classifier.pth",
) -> float:
    """Compute FID for MNIST using the provided code.

    Input tensors must be (N,1,28,28) in [0,1]. We convert to [-1,1] as required.
    """
    x_real = x_real_01 * 2.0 - 1.0
    x_gen  = x_gen_01  * 2.0 - 1.0
    return compute_fid(x_real, x_gen, device=device, classifier_ckpt=ckpt_path)
