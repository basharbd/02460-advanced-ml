from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def _decode_mean_single(model: nn.Module, z: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "decode_mean"):
        raise AttributeError("Model must implement decode_mean(z).")
    return model.decode_mean(z)


def _decode_mean_multi(model: nn.Module, z: torch.Tensor, decoder_idx: int) -> torch.Tensor:
    if not hasattr(model, "decode_mean"):
        raise AttributeError("Model must implement decode_mean(z, decoder_idx=...).")
    return model.decode_mean(z, decoder_idx=decoder_idx)


def latent_curve_points(
    curve: nn.Module,
    num_points: int,
    device: torch.device | str,
) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, num_points, device=device).unsqueeze(1)
    return curve(t)


def curve_energy_mean_decoder(
    curve: nn.Module,
    model: nn.Module,
    num_points: int = 20,
    scale_by_num_segments: bool = True,
) -> torch.Tensor:
    """
    Approximate the pull-back curve energy using decoder means only.

        E ~= N * sum_s || f(c_s) - f(c_{s-1}) ||^2
    """
    z_points = latent_curve_points(curve, num_points=num_points, device=curve.start.device)
    x_means = _decode_mean_single(model, z_points)

    deltas = x_means[1:] - x_means[:-1]
    squared_step_lengths = deltas.flatten(start_dim=1).pow(2).sum(dim=1)

    energy = squared_step_lengths.sum()
    if scale_by_num_segments:
        num_segments = num_points - 1
        energy = num_segments * energy

    return energy


def curve_energy_ensemble(
    curve: nn.Module,
    model: nn.Module,
    num_points: int = 20,
    num_mc_samples: int = 1,
    num_decoders_to_use: int | None = None,
    scale_by_num_segments: bool = True,
) -> torch.Tensor:
    """
    Approximate the model-average curve energy for an ensemble-decoder VAE.

    Uses decoder means, not random decoder samples.
    """
    if not hasattr(model, "num_decoders"):
        raise AttributeError("Model must have attribute num_decoders.")

    total_decoders = model.num_decoders
    if num_decoders_to_use is None:
        num_decoders_to_use = total_decoders
    num_decoders_to_use = min(num_decoders_to_use, total_decoders)

    if num_mc_samples < 1:
        raise ValueError("num_mc_samples must be >= 1.")

    z_points = latent_curve_points(curve, num_points=num_points, device=curve.start.device)

    decoded = []
    for d in range(num_decoders_to_use):
        decoded.append(_decode_mean_multi(model, z_points, decoder_idx=d))

    num_segments = num_points - 1
    energy = torch.zeros((), device=z_points.device)

    for s in range(num_segments):
        segment_energy = torch.zeros((), device=z_points.device)
        for _ in range(num_mc_samples):
            l = torch.randint(low=0, high=num_decoders_to_use, size=(1,), device=z_points.device).item()
            k = torch.randint(low=0, high=num_decoders_to_use, size=(1,), device=z_points.device).item()

            left = decoded[l][s]
            right = decoded[k][s + 1]
            segment_energy = segment_energy + (right - left).pow(2).sum()

        energy = energy + segment_energy / num_mc_samples

    if scale_by_num_segments:
        energy = num_segments * energy

    return energy


def optimize_geodesic(
    curve: nn.Module,
    energy_fn: Callable[[], torch.Tensor],
    optimizer_name: str = "adam",
    lr: float = 0.1,
    epochs: int = 600,
    verbose: bool = False,
) -> tuple[nn.Module, float, list[float]]:
    trainable_params = [p for p in curve.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("Curve has no trainable parameters.")

    name = optimizer_name.lower()
    if name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
    elif name == "lbfgs":
        optimizer = torch.optim.LBFGS(trainable_params, lr=lr, max_iter=20)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'lbfgs'.")

    history: list[float] = []
    curve.train()

    if name == "adam":
        for step in range(epochs):
            optimizer.zero_grad()
            energy = energy_fn()
            energy.backward()
            optimizer.step()

            energy_value = float(energy.detach().cpu())
            history.append(energy_value)

            if verbose and (step % 50 == 0 or step == epochs - 1):
                print(f"[Adam] step={step:04d} energy={energy_value:.6f}")

    else:
        for step in range(epochs):
            def closure():
                optimizer.zero_grad()
                energy = energy_fn()
                energy.backward()
                return energy

            energy = optimizer.step(closure)
            energy_value = float(energy.detach().cpu())
            history.append(energy_value)

            if verbose and (step % 20 == 0 or step == epochs - 1):
                print(f"[LBFGS] step={step:04d} energy={energy_value:.6f}")

    final_energy = history[-1]
    return curve, final_energy, history


def estimate_geodesic_distance_from_energy(energy_value: float) -> float:
    """
    If the optimized curve is approximately constant-speed on [0, 1],
    then geodesic distance ~= sqrt(energy).
    """
    if energy_value < 0:
        raise ValueError("energy_value must be non-negative.")
    return float(energy_value**0.5)
