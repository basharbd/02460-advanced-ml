# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-02-07)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/flows/realnvp_example.ipynb
# - https://github.com/VincentStimper/normalizing-flows/tree/master

import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm


class GaussianBase(nn.Module):
    def __init__(self, D):
        """Gaussian base distribution with zero mean and unit variance."""
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """Return the base distribution."""
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MaskedCouplingLayer(nn.Module):
    """An affine coupling layer for a normalizing flow."""

    def __init__(self, scale_net, translation_net, mask):
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """Forward transform (base -> data)."""
        x = z
        z_prime = self.mask * x + (1 - self.mask) * (
            x * torch.exp(self.scale_net(self.mask * x)) + self.translation_net(self.mask * x)
        )
        log_det_J = torch.sum((1 - self.mask) * self.scale_net(self.mask * z_prime), dim=-1)
        return z_prime, log_det_J

    def inverse(self, x):
        """Inverse transform (data -> base)."""
        z = x
        z = self.mask * z + (1 - self.mask) * (
            (z - self.translation_net(self.mask * z)) * torch.exp(-self.scale_net(self.mask * z))
        )
        log_det_J = -torch.sum((1 - self.mask) * self.scale_net(self.mask * x), dim=-1)
        return z, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """Forward flow (base -> data)."""
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J

    def inverse(self, x):
        """Inverse flow (data -> base)."""
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J

    def log_prob(self, x):
        """Compute log p(x) under the flow."""
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J

    def sample(self, sample_shape=(1,)):
        """Sample from the flow."""
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]

    def loss(self, x):
        """Negative mean log likelihood."""
        return -torch.mean(self.log_prob(x))


def train(model, optimizer, data_loader, epochs, device):
    """Train a Flow model."""
    model.train()
    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()