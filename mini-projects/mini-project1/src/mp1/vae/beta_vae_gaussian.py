"""β-VAE with Gaussian likelihood for Part B.

We use:
- q(z|x): diagonal Gaussian (encoder outputs mean and log_std)
- p(z): standard Gaussian
- p(x|z): diagonal Gaussian with fixed std (simple + stable)

Objective (maximize):
$$ \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta\, \mathrm{KL}(q(z|x)\Vert p(z)) $$

This is intentionally lightweight for CPU training on MNIST.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm

class BetaVAE(nn.Module):
    def __init__(self, encoder_net: nn.Module, decoder_net: nn.Module, latent_dim: int, beta: float = 1.0, x_std: float = 0.1):
        super().__init__()
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.latent_dim = latent_dim
        self.beta = beta
        self.register_buffer("prior_mean", torch.zeros(latent_dim))
        self.register_buffer("prior_std", torch.ones(latent_dim))
        self.x_std = x_std

    def prior(self) -> td.Distribution:
        return td.Independent(td.Normal(self.prior_mean, self.prior_std), 1)

    def encoder(self, x: torch.Tensor) -> td.Distribution:
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(mean, torch.exp(log_std)), 1)

    def decoder(self, z: torch.Tensor) -> td.Distribution:
        mu = self.decoder_net(z)  # (B,1,28,28)
        return td.Independent(td.Normal(mu, self.x_std), 3)

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        q = self.encoder(x)
        z = q.rsample()
        recon = self.decoder(z).log_prob(x)
        kl = td.kl_divergence(q, self.prior())
        return torch.mean(recon - self.beta * kl, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.elbo(x)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = self.prior().sample((n,))
        return self.decoder(z).mean

def build_encoder(latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, latent_dim * 2),
    )

def build_decoder(latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (1, 28, 28)),
    )

def train_beta_vae(model: BetaVAE, optimizer, loader, epochs: int, device: str) -> None:
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"β-VAE epoch {epoch+1}/{epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
