# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)


"""DDPM implementation for Part B (image-space + latent-space).

This implements:
- a practical DDPM training objective (noise prediction / MSE),
- a standard reverse diffusion sampler.

The network must implement:
    eps_hat = net(x_t, t_scaled)
where:
- x_t shape: (B, D)
- t_scaled shape: (B, 1) with values in [0,1]
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, beta_1: float = 1e-4, beta_T: float = 2e-2, T: int = 200):
        super().__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        beta = torch.linspace(beta_1, beta_T, T)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1.0 - beta)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - beta, dim=0))

    def _t_to_scaled(self, t_int: torch.Tensor) -> torch.Tensor:
        # (B,) -> (B,1) in [0,1]
        return (t_int.float() / (self.T - 1)).unsqueeze(1)

    def negative_elbo(self, x0: torch.Tensor) -> torch.Tensor:
        """A per-sample objective (noise prediction loss).

        This is the standard training loss used for DDPMs:
        - sample t ~ Uniform({0..T-1})
        - sample eps ~ N(0,I)
        - form x_t = sqrt(alpha_bar[t]) x0 + sqrt(1-alpha_bar[t]) eps
        - predict eps_hat = net(x_t, t)
        - return ||eps - eps_hat||^2
        """
        B = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(x0)

        a_bar = self.alpha_bar[t].view(B, *([1] * (x0.ndim - 1)))
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps

        eps_hat = self.network(x_t.view(B, -1), self._t_to_scaled(t))
        eps_hat = eps_hat.view_as(x0)

        loss = F.mse_loss(eps_hat, eps, reduction="none")
        # sum over non-batch dims -> per-sample
        return loss.view(B, -1).sum(dim=1)

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        return self.negative_elbo(x0).mean()

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        device = self.beta.device
        x_t = torch.randn(shape, device=device)

        for t in range(self.T - 1, -1, -1):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            a_t = self.alpha[t]
            a_bar_t = self.alpha_bar[t]
            b_t = self.beta[t]

            eps_hat = self.network(x_t.view(shape[0], -1), self._t_to_scaled(t_batch)).view_as(x_t)

            coef1 = 1.0 / torch.sqrt(a_t)
            coef2 = (1.0 - a_t) / torch.sqrt(1.0 - a_bar_t)
            mean = coef1 * (x_t - coef2 * eps_hat)

            if t > 0:
                z = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(b_t) * z
            else:
                x_t = mean

        return x_t


def train_ddpm(model: DDPM, optimizer, loader, epochs: int, device: str) -> None:
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"DDPM epoch {epoch+1}/{epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))


class FcNetwork(nn.Module):
    """Tiny MLP for latent DDPM."""
    def __init__(self, input_dim: int, num_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)
