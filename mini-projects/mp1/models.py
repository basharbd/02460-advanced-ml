from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributions as td


class GaussianPrior(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(latent_dim), requires_grad=False)

    def forward(self) -> td.Distribution:
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    """
    Encoder returns q(z|x) = N(mu(x), diag(exp(log_std(x))^2)).
    """

    def __init__(self, encoder_net: nn.Module):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x: torch.Tensor) -> td.Distribution:
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        std = torch.exp(log_std)
        return td.Independent(td.Normal(loc=mean, scale=std), 1)


class GaussianDecoder(nn.Module):
    """
    Decoder returns p(x|z) = N(mean(z), sigma^2 I) with fixed sigma.
    """

    def __init__(self, decoder_net: nn.Module, obs_std: float = 0.1):
        super().__init__()
        self.decoder_net = decoder_net
        self.obs_std = obs_std

    def forward(self, z: torch.Tensor) -> td.Distribution:
        mean = self.decoder_net(z)
        scale = torch.full_like(mean, fill_value=self.obs_std)
        return td.Independent(td.Normal(loc=mean, scale=scale), 3)

    def mean(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)


def build_encoder_net(latent_dim: int) -> nn.Module:
    """
    Simple CNN encoder for MNIST-sized images.
    Output dimension = 2 * latent_dim for mean and log_std.
    """
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 28 -> 14
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14 -> 7
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 7 -> 4
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 2 * latent_dim),
    )


def build_decoder_net(latent_dim: int) -> nn.Module:
    """
    Simple decoder mirroring the encoder.
    Output shape = [B, 1, 28, 28]
    """
    return nn.Sequential(
        nn.Linear(latent_dim, 32 * 4 * 4),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # 4 -> 7
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7 -> 14
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 14 -> 28
        nn.Sigmoid(),
    )


class VAE(nn.Module):
    def __init__(
        self,
        prior: nn.Module,
        decoder: GaussianDecoder,
        encoder: GaussianEncoder,
    ):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        q = self.encoder(x)
        z = q.rsample()
        recon_log_prob = self.decoder(z).log_prob(x)
        kl_term = q.log_prob(z) - self.prior().log_prob(z)
        return torch.mean(recon_log_prob - kl_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.elbo(x)

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).mean

    def decode_mean(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder.mean(z)


class MultiDecoderVAE(nn.Module):
    """
    Ensemble-decoder VAE:
      - one shared encoder
      - one shared prior
      - multiple independent decoders
    """

    def __init__(
        self,
        prior: nn.Module,
        base_decoder: GaussianDecoder,
        encoder: GaussianEncoder,
        num_decoders: int = 3,
    ):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.num_decoders = num_decoders
        self.decoders = nn.ModuleList(
            [deepcopy(base_decoder) for _ in range(num_decoders)]
        )

    def elbo(self, x: torch.Tensor, decoder_idx: int = 0) -> torch.Tensor:
        if not (0 <= decoder_idx < self.num_decoders):
            raise ValueError(f"decoder_idx must be in [0, {self.num_decoders - 1}]")

        q = self.encoder(x)
        z = q.rsample()
        recon_log_prob = self.decoders[decoder_idx](z).log_prob(x)
        kl_term = q.log_prob(z) - self.prior().log_prob(z)
        return torch.mean(recon_log_prob - kl_term)

    def forward(self, x: torch.Tensor, decoder_idx: int = 0) -> torch.Tensor:
        return -self.elbo(x, decoder_idx=decoder_idx)

    def sample(self, n_samples: int = 1, decoder_idx: int = 0) -> torch.Tensor:
        if not (0 <= decoder_idx < self.num_decoders):
            raise ValueError(f"decoder_idx must be in [0, {self.num_decoders - 1}]")
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoders[decoder_idx](z).sample()

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).mean

    def decode_mean(self, z: torch.Tensor, decoder_idx: int = 0) -> torch.Tensor:
        if not (0 <= decoder_idx < self.num_decoders):
            raise ValueError(f"decoder_idx must be in [0, {self.num_decoders - 1}]")
        return self.decoders[decoder_idx].mean(z)

    def decode_all_means(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor of shape [num_decoders, batch, 1, 28, 28]
        """
        means = [decoder.mean(z) for decoder in self.decoders]
        return torch.stack(means, dim=0)


def build_single_vae(latent_dim: int = 2, obs_std: float = 0.1) -> VAE:
    prior = GaussianPrior(latent_dim)
    encoder = GaussianEncoder(build_encoder_net(latent_dim))
    decoder = GaussianDecoder(build_decoder_net(latent_dim), obs_std=obs_std)
    return VAE(prior=prior, decoder=decoder, encoder=encoder)


def build_multi_decoder_vae(
    latent_dim: int = 2,
    obs_std: float = 0.1,
    num_decoders: int = 3,
) -> MultiDecoderVAE:
    prior = GaussianPrior(latent_dim)
    encoder = GaussianEncoder(build_encoder_net(latent_dim))
    base_decoder = GaussianDecoder(build_decoder_net(latent_dim), obs_std=obs_std)
    return MultiDecoderVAE(
        prior=prior,
        base_decoder=base_decoder,
        encoder=encoder,
        num_decoders=num_decoders,
    )
