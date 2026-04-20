from __future__ import annotations
import argparse
from pathlib import Path
import torch
import numpy as np

from mp1.data import get_mnist_loaders
from mp1.utils import set_seed, save_json
from mp1.plotting import plot_prior_vs_agg_posterior
from mp1.vae.vae_bernoulli import (
    VAE, GaussianPrior, MoGPrior, GaussianEncoder, BernoulliDecoder,
    build_mlp_encoder, build_mlp_decoder,
    train_vae, eval_elbo_mean, collect_aggregate_posterior,
)
from mp1.flows.flow import GaussianBase, MaskedCouplingLayer, Flow


def make_flow_prior(latent_dim: int, n_layers: int = 6, hidden: int = 64) -> Flow:
    base = GaussianBase(latent_dim)

    transformations = []
    mask = torch.zeros(latent_dim)
    mask[latent_dim // 2:] = 1.0

    for _ in range(n_layers):
        mask = 1.0 - mask
        scale_net = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, latent_dim))
        trans_net = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, latent_dim))
        transformations.append(MaskedCouplingLayer(scale_net, trans_net, mask))

    return Flow(base, transformations)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/partA", help="output directory")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--mog_components", type=int, default=10)
    p.add_argument("--fast", action="store_true", help="quick smoke test settings")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    if args.fast:
        args.epochs = 2
        args.runs = 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loaders = get_mnist_loaders(batch_size=args.batch_size, binarized=True, root=str(outdir / "data"), num_workers=0)

    priors = {}
    priors["gaussian"] = lambda: GaussianPrior(args.latent_dim)
    priors["mog"] = lambda: MoGPrior(args.latent_dim, num_components=args.mog_components, device=device)
    priors["flow"] = lambda: make_flow_prior(args.latent_dim)

    results = {}

    for prior_name, prior_factory in priors.items():
        elbos = []
        for run in range(args.runs):
            seed = 1234 + run
            set_seed(seed)

            prior = prior_factory()
            encoder = GaussianEncoder(build_mlp_encoder(args.latent_dim))
            decoder = BernoulliDecoder(build_mlp_decoder(args.latent_dim))
            model = VAE(prior=prior, decoder=decoder, encoder=encoder).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            train_vae(model, opt, loaders.train, epochs=args.epochs, device=device)

            elbo = eval_elbo_mean(model, loaders.test, device=device)
            elbos.append(elbo)

            # Prior vs agg posterior plot (use more posterior samples than prior samples)
            qz = collect_aggregate_posterior(model, loaders.test, device=device, max_batches=100)
            pz = model.sample_prior(min(len(qz), 2000))

            plot_path = outdir / prior_name / f"prior_vs_posterior_run{run}.png"
            plot_prior_vs_agg_posterior(pz, qz, plot_path, title=f"{prior_name} (run {run})")

        elbos = np.array(elbos, dtype=float)
        results[prior_name] = {"elbo_mean": float(elbos.mean()), "elbo_std": float(elbos.std(ddof=1) if len(elbos) > 1 else 0.0), "runs": int(args.runs)}

    save_json(outdir / "summary.json", results)
    print("Saved Part A results to:", outdir / "summary.json")
    print(results)


if __name__ == "__main__":
    main()
