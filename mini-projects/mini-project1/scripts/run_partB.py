from __future__ import annotations
import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision.utils import save_image, make_grid

from mp1.data import get_mnist_loaders
from mp1.utils import set_seed, save_json, Timer
from mp1.metrics.fid_wrapper import compute_fid_mnist
from mp1.diffusion.unet import Unet
from mp1.diffusion.ddpm import DDPM, train_ddpm, FcNetwork
from mp1.vae.beta_vae_gaussian import BetaVAE, build_encoder, build_decoder, train_beta_vae


def plot_three_latent_distributions(
    prior_samples: torch.Tensor,
    posterior_samples: torch.Tensor,
    ddpm_samples: torch.Tensor,
    outpath: Path,
    title: str = "",
) -> None:
    """
    Plot:
      - β-VAE prior p(z)
      - aggregate posterior q(z) (samples from encoder on test set)
      - latent DDPM distribution p_ddpm(z) (samples from latent DDPM)

    If latent_dim > 2, we PCA-project to 2D for visualization.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    outpath.parent.mkdir(parents=True, exist_ok=True)

    P = prior_samples.detach().cpu().numpy()
    Q = posterior_samples.detach().cpu().numpy()
    D = ddpm_samples.detach().cpu().numpy()

    if P.shape[1] > 2:
        pca = PCA(n_components=2)
        pca.fit(np.concatenate([P, Q, D], axis=0))
        P = pca.transform(P)
        Q = pca.transform(Q)
        D = pca.transform(D)

    plt.figure(figsize=(6, 5))
    plt.scatter(P[:, 0], P[:, 1], s=6, alpha=0.35, label="β-VAE prior p(z)")
    plt.scatter(Q[:, 0], Q[:, 1], s=6, alpha=0.35, label="aggregate posterior q(z)")
    plt.scatter(D[:, 0], D[:, 1], s=6, alpha=0.35, label="latent DDPM p(z)")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/partB")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs_ddpm", type=int, default=5)
    p.add_argument("--epochs_vae", type=int, default=5)
    p.add_argument("--epochs_latent_ddpm", type=int, default=5)
    p.add_argument("--T_ddpm", type=int, default=200)
    p.add_argument("--T_latent", type=int, default=200)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--betas", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2])
    p.add_argument("--fid_n", type=int, default=2000)
    p.add_argument("--latent_plot_n", type=int, default=2000)  # NEW
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


@torch.no_grad()
def sample_images_from_ddpm(ddpm: DDPM, n: int) -> torch.Tensor:
    x = ddpm.sample((n, 784))
    x = x.view(n, 1, 28, 28)
    # Map from unconstrained -> [0,1] via sigmoid (simple)
    return torch.sigmoid(x)


def main():
    args = parse_args()
    device = args.device
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        args.epochs_ddpm = 1
        args.epochs_vae = 1
        args.epochs_latent_ddpm = 1
        args.fid_n = 256
        args.latent_plot_n = 512
        args.T_ddpm = 100
        args.T_latent = 100

    loaders = get_mnist_loaders(
        batch_size=args.batch_size,
        binarized=False,
        root=str(outdir / "data"),
        num_workers=0
    )

    # --------- 1) Image-space DDPM ---------
    set_seed(0)
    unet = Unet().to(device)
    ddpm_img = DDPM(unet, T=args.T_ddpm).to(device)
    opt = torch.optim.Adam(ddpm_img.parameters(), lr=1e-3)

    # Flatten MNIST for this U-Net implementation
    flat_train = [(x.view(x.size(0), -1), y) for x, y in loaders.train]
    flat_test  = [(x.view(x.size(0), -1), y) for x, y in loaders.test]

    train_ddpm(ddpm_img, opt, flat_train, epochs=args.epochs_ddpm, device=device)

    # Samples (4)
    x_ddpm_4 = sample_images_from_ddpm(ddpm_img, 4).cpu()
    save_image(make_grid(x_ddpm_4, nrow=4), outdir / "samples_ddpm.png")

    # --------- 2) β-VAE + latent DDPM ---------
    fid_rows = []
    timing_rows = []

    # Real images for FID (take first fid_n from test)
    x_real = []
    for x, _ in loaders.test:
        x_real.append(x)
        if sum(t.size(0) for t in x_real) >= args.fid_n:
            break
    x_real = torch.cat(x_real, dim=0)[: args.fid_n].to(device)

    # FID + timing for DDPM
    with Timer() as t:
        x_gen = sample_images_from_ddpm(ddpm_img, args.fid_n).to(device)
    ddpm_sps = args.fid_n / t.seconds
    fid_ddpm = compute_fid_mnist(x_real, x_gen, device=device, ckpt_path="mnist_classifier.pth")
    fid_rows.append({"model": "ddpm", "beta": None, "fid": float(fid_ddpm)})
    timing_rows.append({"model": "ddpm", "beta": None, "samples_per_sec": float(ddpm_sps)})

    # VAE choice for comparison: we will use β-VAE with beta=1e-4 by default
    chosen_vae_samples = None
    chosen_vae_fid = None
    chosen_vae_sps = None

    for beta in args.betas:
        set_seed(123)  # keep fixed init across betas for fair comparison

        enc = build_encoder(args.latent_dim)
        dec = build_decoder(args.latent_dim)
        beta_vae = BetaVAE(enc, dec, latent_dim=args.latent_dim, beta=beta, x_std=0.1).to(device)
        opt_vae = torch.optim.Adam(beta_vae.parameters(), lr=1e-3)

        train_beta_vae(beta_vae, opt_vae, loaders.train, epochs=args.epochs_vae, device=device)

        # Build latent dataset Z by sampling q(z|x) on TRAIN
        Z = []
        for x, _ in loaders.train:
            x = x.to(device)
            q = beta_vae.encoder(x)
            z = q.sample()
            Z.append(z.detach().cpu())
        Z = torch.cat(Z, dim=0)
        latent_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z),
            batch_size=args.batch_size,
            shuffle=True
        )

        # Latent DDPM (MLP)
        net_z = FcNetwork(args.latent_dim, num_hidden=128).to(device)
        ddpm_z = DDPM(net_z, T=args.T_latent).to(device)
        opt_z = torch.optim.Adam(ddpm_z.parameters(), lr=1e-3)

        # loader yields (z,) so adapt for train_ddpm
        latent_batches = [(z, None) for (z,) in latent_loader]
        train_ddpm(ddpm_z, opt_z, latent_batches, epochs=args.epochs_latent_ddpm, device=device)

        # ---- NEW: Latent distribution plot (prior vs posterior vs latent DDPM) ----
        Nplot = args.latent_plot_n

        # 1) β-VAE prior p(z)=N(0,I)
        z_prior = beta_vae.prior().sample((Nplot,)).to(device)

        # 2) aggregate posterior q(z): sample z~q(z|x) on TEST
        z_post_list = []
        count = 0
        for x_t, _ in loaders.test:
            x_t = x_t.to(device)
            q_t = beta_vae.encoder(x_t)
            z_b = q_t.sample()
            z_post_list.append(z_b)
            count += z_b.size(0)
            if count >= Nplot:
                break
        z_post = torch.cat(z_post_list, dim=0)[:Nplot]

        # 3) latent DDPM samples p_ddpm(z)
        z_ddpm = ddpm_z.sample((Nplot, args.latent_dim))

        plot_three_latent_distributions(
            z_prior, z_post, z_ddpm,
            outdir / f"latent_dist_beta_{beta}.png",
            title=f"Latent distributions (beta={beta})"
        )

        # Generate samples for FID + timing
        with Timer() as t:
            z_gen = ddpm_z.sample((args.fid_n, args.latent_dim))
            x_gen = beta_vae.decoder(z_gen).mean  # (N,1,28,28)
            x_gen = torch.clamp(x_gen, 0.0, 1.0)
        sps = args.fid_n / t.seconds
        fid = compute_fid_mnist(x_real, x_gen.to(device), device=device, ckpt_path="mnist_classifier.pth")

        fid_rows.append({"model": "latent_ddpm", "beta": float(beta), "fid": float(fid)})
        timing_rows.append({"model": "latent_ddpm", "beta": float(beta), "samples_per_sec": float(sps)})

        # 4 samples grid for beta=1e-6
        if abs(beta - 1e-6) < 1e-12:
            save_image(make_grid(x_gen[:4].cpu(), nrow=4), outdir / f"samples_latent_ddpm_beta_{beta}.png")

        # Choose a VAE for the "VAE of your choice" slot (pick beta=1e-4)
        if chosen_vae_samples is None and abs(beta - 1e-4) < 1e-12:
            with Timer() as tv:
                x_vae = beta_vae.sample(args.fid_n)
                x_vae = torch.clamp(x_vae, 0.0, 1.0)
            chosen_vae_samples = x_vae
            chosen_vae_sps = args.fid_n / tv.seconds
            chosen_vae_fid = compute_fid_mnist(x_real, x_vae.to(device), device=device, ckpt_path="mnist_classifier.pth")
            save_image(make_grid(x_vae[:4].cpu(), nrow=4), outdir / "samples_vae_choice.png")

    fid_rows.append({"model": "vae_choice", "beta": 1e-4, "fid": float(chosen_vae_fid)})
    timing_rows.append({"model": "vae_choice", "beta": 1e-4, "samples_per_sec": float(chosen_vae_sps)})

    save_json(outdir / "fid_results.json", {"rows": fid_rows})
    save_json(outdir / "timing_results.json", {"rows": timing_rows})

    print("Saved Part B outputs to:", outdir)
    print("FID results:", fid_rows)
    print("Timing:", timing_rows)


if __name__ == "__main__":
    main()