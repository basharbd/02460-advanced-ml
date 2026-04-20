from __future__ import annotations

import torch
import torch.nn as nn


class LinearCurve(nn.Module):
    """
    Straight-line curve between start and end:
        c(t) = (1-t) * start + t * end
    """

    def __init__(self, start: torch.Tensor, end: torch.Tensor):
        super().__init__()
        if start.ndim != 1 or end.ndim != 1:
            raise ValueError("start and end must be 1D tensors of shape [latent_dim].")
        if start.shape != end.shape:
            raise ValueError("start and end must have the same shape.")

        self.latent_dim = start.numel()
        self.start = nn.Parameter(start.reshape(1, -1), requires_grad=False)
        self.end = nn.Parameter(end.reshape(1, -1), requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        elif t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("t must have shape [T] or [T, 1].")

        return (1.0 - t) * self.start + t * self.end


class PolynomialCurve(nn.Module):
    """
    Smooth curve between start and end:

        c(t) = (1-t) * start + t * end + r(t)

    where r(0)=0 and r(1)=0.

    We model r(t) as a degree-K polynomial per latent dimension:
        r_i(t) = sum_{k=1}^{K} w_{i,k} t^k
    with the last coefficient chosen so that r_i(1)=0.
    """

    def __init__(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        degree: int = 3,
        init_scale: float = 0.05,
    ):
        super().__init__()

        if start.ndim != 1 or end.ndim != 1:
            raise ValueError("start and end must be 1D tensors of shape [latent_dim].")
        if start.shape != end.shape:
            raise ValueError("start and end must have the same shape.")
        if degree < 2:
            raise ValueError("degree must be at least 2.")

        self.latent_dim = start.numel()
        self.degree = degree

        self.start = nn.Parameter(start.reshape(1, -1), requires_grad=False)
        self.end = nn.Parameter(end.reshape(1, -1), requires_grad=False)

        # Learn K-1 coefficients; derive the K-th so r(1)=0
        self.weights = nn.Parameter(
            init_scale * torch.randn(self.latent_dim, self.degree - 1)
        )

    def _remainder(self, t: torch.Tensor) -> torch.Tensor:
        last_weight = -torch.sum(self.weights, dim=1, keepdim=True)
        all_weights = torch.cat([self.weights, last_weight], dim=1)
        t_powers = torch.cat([t**k for k in range(1, self.degree + 1)], dim=1)
        return t_powers @ all_weights.T

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        elif t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("t must have shape [T] or [T, 1].")

        base = (1.0 - t) * self.start + t * self.end
        return base + self._remainder(t)


def make_time_grid(
    num_points: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return regular grid in [0, 1] of shape [num_points, 1].
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    return torch.linspace(0.0, 1.0, num_points, device=device, dtype=dtype).unsqueeze(1)
