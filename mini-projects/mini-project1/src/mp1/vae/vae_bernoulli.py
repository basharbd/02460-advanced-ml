
# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

"""Bernoulli VAE for Part A (binarized MNIST).

This file is based on the provided week1 VAE code.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as td
from torch.nn import functional as F
from tqdm import tqdm


# -------------------------
# Priors
# -------------------------

class GaussianPrior(nn.Module):
    def __init__(self, M: int):
        super().__init__()
        self.M = M
        self.register_buffer("mean", torch.zeros(M))
        self.register_buffer("std", torch.ones(M))

    def forward(self) -> td.Distribution:
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward().log_prob(z)


class MoGPrior(nn.Module):
    def __init__(self, M: int, num_components: int, device: str = "cpu", init_radius: float = 4.0):
        super().__init__()
        self.M = M
        self.num_components = num_components
        self.device = device
        self.init_radius = init_radius

        self.mean = nn.Parameter(
            torch.randn(num_components, M).uniform_(-init_radius, init_radius),
            requires_grad=False
        ).to(device)

        self.stds = nn.Parameter(
            torch.ones(num_components, M),
            requires_grad=False
        ).to(device)

        self.weights = nn.Parameter(
            torch.ones(num_components),
            requires_grad=False
        ).to(device)

    def forward(self) -> td.Distribution:
        mixture_dist = td.Categorical(probs=F.softmax(self.weights, dim=0))
        comp_dist = td.Independent(td.Normal(loc=self.mean, scale=self.stds), 1)
        return td.MixtureSameFamily(mixture_dist, comp_dist)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward().log_prob(z)


# -------------------------
# Encoder / Decoder
# -------------------------

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net: nn.Module):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x: torch.Tensor) -> td.Distribution:
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net: nn.Module):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z: torch.Tensor) -> td.Distribution:
        logits = self.decoder_net(z)  # (B,28,28)
        return td.Independent(td.Bernoulli(logits=logits), 2)


# -------------------------
# VAE
# -------------------------

class VAE(nn.Module):
    def __init__(self, prior: nn.Module, decoder: nn.Module, encoder: nn.Module):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def _prior_dist(self) -> td.Distribution | None:
        # GaussianPrior & MoGPrior expose forward() -> Distribution
        if hasattr(self.prior, "forward") and not hasattr(self.prior, "inverse"):
            try:
                dist = self.prior()
                if isinstance(dist, td.Distribution):
                    return dist
            except TypeError:
                pass
        return None

    def sample_prior(self, n: int) -> torch.Tensor:
        # GaussianPrior / MoGPrior: sample from returned distribution
        dist = self._prior_dist()
        if dist is not None:
            return dist.sample((n,))
        # Flow prior: has .sample(sample_shape)
        if hasattr(self.prior, "sample"):
            return self.prior.sample((n,))
        raise TypeError("Unsupported prior type for sampling.")

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a scalar ELBO (mean over batch).
        """
        q = self.encoder(x)
        z = q.rsample()

        # Gaussian prior: analytic KL
        if isinstance(self.prior, GaussianPrior):
            pz = self.prior()
            elbo_per_sample = self.decoder(z).log_prob(x) - td.kl_divergence(q, pz)  # (B,)
            return elbo_per_sample.mean()

        # Non-Gaussian prior:
        # KL(q||p) = E_q[ log q(z|x) - log p(z) ]
        reg = q.log_prob(z) - self.prior.log_prob(z)  # (B,)
        elbo_per_sample = self.decoder(z).log_prob(x) - reg  # (B,)
        return elbo_per_sample.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Minimization objective = -ELBO
        return -self.elbo(x)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = self.sample_prior(n)
        return self.decoder(z).sample()


# -------------------------
# Builders
# -------------------------

def build_mlp_encoder(latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, latent_dim * 2),
    )


def build_mlp_decoder(latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )


# -------------------------
# Training / evaluation helpers
# -------------------------

def _prepare_x(x: torch.Tensor, device: str) -> torch.Tensor:
    """
    MNIST from torchvision is (B,1,28,28). Our Bernoulli VAE uses (B,28,28).
    This function ensures consistent shape and dtype.
    """
    x = x.to(device)
    if x.ndim == 4 and x.size(1) == 1:
        x = x.squeeze(1)  # (B,1,28,28) -> (B,28,28)
    return x


def train_vae(model: VAE, optimizer: torch.optim.Optimizer, loader, epochs: int, device: str) -> None:
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"VAE epoch {epoch+1}/{epochs}", leave=False)
        for x, _ in pbar:
            x = _prepare_x(x, device)

            optimizer.zero_grad()
            loss = model(x)          # scalar
            loss.backward()          # OK now
            optimizer.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))


@torch.no_grad()
def eval_elbo_mean(model: VAE, loader, device: str) -> float:
    model.eval()
    elbos = []
    for x, _ in loader:
        x = _prepare_x(x, device)
        elbos.append(model.elbo(x).detach().cpu())  # scalar
    return torch.stack(elbos).mean().item()


@torch.no_grad()
def collect_aggregate_posterior(model: VAE, loader, device: str, max_batches: int = 200) -> torch.Tensor:
    """Return samples z~q(z|x) aggregated over a subset of the loader."""
    model.eval()
    zs = []
    for b, (x, _) in enumerate(loader):
        if b >= max_batches:
            break
        x = _prepare_x(x, device)
        q = model.encoder(x)
        z = q.sample()
        zs.append(z.detach().cpu())
    return torch.cat(zs, dim=0)