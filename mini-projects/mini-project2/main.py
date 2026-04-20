from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from data import get_mnist_subset_loaders, add_training_noise
from models import build_single_vae, build_multi_decoder_vae
from curves import PolynomialCurve, LinearCurve
from geodesics import (
    curve_energy_mean_decoder,
    optimize_geodesic,
    estimate_geodesic_distance_from_energy,
)
from analysis_cov import (
    collect_distances_across_models,
    compute_cov_summary,
    save_distances_csv,
    save_cov_summary_csv,
    build_part_b_geodesic_artifacts,
)
from plotting import make_part_a_plot, make_part_b_plot, make_cov_plot_from_csv
from utils import (
    set_seed,
    get_best_device,
    ensure_dir,
    save_checkpoint,
    load_checkpoint,
    get_latent_means,
    get_dataset_tensors_from_loader,
    select_fixed_point_pairs,
    materialize_pairs_from_latents,
    checkpoint_path,
)


def train_single_vae(
    model: torch.nn.Module,
    train_loader,
    device: str,
    epochs: int = 50,
    lr: float = 1e-3,
    noise_std: float = 0.05,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_losses = []

        for x, _ in train_loader:
            x = x.to(device)
            x = add_training_noise(x, std=noise_std)

            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[train] epoch={epoch+1:03d}/{epochs} loss={mean_loss:.4f}")


def train_multi_decoder_vae(
    model: torch.nn.Module,
    train_loader,
    device: str,
    epochs_per_decoder: int = 50,
    lr: float = 1e-3,
    noise_std: float = 0.05,
) -> None:
    """
    Train ensemble-decoder VAE.

    At each step:
      - sample one decoder index uniformly
      - update the shared encoder/prior and the chosen decoder
    """
    if not hasattr(model, "num_decoders"):
        raise ValueError("Expected an ensemble model with attribute num_decoders.")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_epochs = epochs_per_decoder * model.num_decoders

    for epoch in range(total_epochs):
        epoch_losses = []

        for x, _ in train_loader:
            x = x.to(device)
            x = add_training_noise(x, std=noise_std)

            decoder_idx = torch.randint(
                low=0, high=model.num_decoders, size=(1,)
            ).item()

            optimizer.zero_grad()
            loss = model(x, decoder_idx=decoder_idx)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(
            f"[train_ensemble] epoch={epoch+1:03d}/{total_epochs} "
            f"loss={mean_loss:.4f}"
        )


@torch.no_grad()
def eval_single_vae(model: torch.nn.Module, test_loader, device: str) -> float:
    model.eval()
    elbos = []

    for x, _ in test_loader:
        x = x.to(device)
        elbo = model.elbo(x)
        elbos.append(float(elbo.detach().cpu()))

    mean_elbo = sum(elbos) / len(elbos)
    print(f"[eval] mean test ELBO = {mean_elbo:.4f}")
    return mean_elbo


@torch.no_grad()
def sample_and_reconstruct(
    model: torch.nn.Module,
    test_loader,
    device: str,
    out_dir: str | Path,
    num_samples: int = 64,
) -> None:
    out_dir = ensure_dir(out_dir)
    model.eval()

    samples = model.sample(num_samples).cpu()
    save_image(samples.view(num_samples, 1, 28, 28), out_dir / "samples.png")

    x, _ = next(iter(test_loader))
    x = x.to(device)
    recon = model.decode_mean(model.encode_mean(x))

    save_image(x[:32].cpu(), out_dir / "input_batch.png")
    save_image(recon[:32].cpu(), out_dir / "reconstruction_means.png")
    print(f"[sample] saved outputs to {out_dir}")


def run_part_a(
    model: torch.nn.Module,
    test_loader,
    device: str,
    results_dir: str | Path,
    num_pairs: int = 25,
    num_curve_points: int = 20,
    geodesic_epochs: int = 600,
    geodesic_lr: float = 0.1,
    seed: int = 42,
) -> None:
    results_dir = ensure_dir(results_dir)

    test_images, test_labels = get_dataset_tensors_from_loader(test_loader)

    pair_dict = select_fixed_point_pairs(
        images=test_images,
        labels=test_labels,
        num_pairs=num_pairs,
        seed=seed,
    )

    torch.save(pair_dict, results_dir / "fixed_pairs.pt")
    torch.save(test_images, results_dir / "test_images.pt")
    torch.save(test_labels, results_dir / "test_labels.pt")

    latent_means, latent_labels = get_latent_means(model, test_loader, device)

    torch.save(latent_means, results_dir / "latent_means.pt")
    torch.save(latent_labels, results_dir / "latent_labels.pt")

    pairs = materialize_pairs_from_latents(latent_means, pair_dict, device=device)

    linear_curves = []
    optimized_curves = []
    distances = []

    for i, (start, end) in enumerate(pairs):
        print(f"[part_a] optimizing curve {i+1}/{len(pairs)}")

        curve = PolynomialCurve(start, end, degree=3).to(device)

        energy_fn = lambda: curve_energy_mean_decoder(
            curve=curve,
            model=model,
            num_points=num_curve_points,
        )

        curve, final_energy, _ = optimize_geodesic(
            curve=curve,
            energy_fn=energy_fn,
            optimizer_name="adam",
            lr=geodesic_lr,
            epochs=geodesic_epochs,
            verbose=False,
        )

        distance_est = estimate_geodesic_distance_from_energy(final_energy)
        distances.append(distance_est)

        t_vis = torch.linspace(0.0, 1.0, 100, device=device).unsqueeze(1)

        linear_curve = LinearCurve(start, end).to(device)
        linear_curves.append(linear_curve(t_vis).detach().cpu())
        optimized_curves.append(curve(t_vis).detach().cpu())

        print(f"[part_a] curve {i+1} distance ≈ {distance_est:.4f}")

    torch.save(torch.stack(linear_curves), results_dir / "linear_curves.pt")
    torch.save(torch.stack(optimized_curves), results_dir / "geodesic_curves.pt")
    torch.save(torch.tensor(distances), results_dir / "geodesic_distances.pt")

    print(f"[part_a] saved curves and distances to {results_dir}")
    print("[part_a] next step later will be plotting these in plotting.py")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini-project 2 starter pipeline")

    parser.add_argument(
        "mode",
        type=str,
        choices=[
            "train",
            "eval",
            "sample",
            "part_a",
            "plot_part_a",
            "train_ensemble",
            "part_b",
            "plot_part_b",
            "plot_cov",
        ],
        help="What to run.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--obs-std", type=float, default=0.1)

    parser.add_argument("--num-train-data", type=int, default=2048)
    parser.add_argument("--num-test-data", type=int, default=2048)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--experiment-dir", type=str, default="experiments/single")
    parser.add_argument("--results-dir", type=str, default="results/part_a")

    parser.add_argument("--num-pairs", type=int, default=25)
    parser.add_argument("--num-curve-points", type=int, default=20)
    parser.add_argument("--geodesic-epochs", type=int, default=600)
    parser.add_argument("--geodesic-lr", type=float, default=0.1)

    parser.add_argument("--num-decoders", type=int, default=3)
    parser.add_argument("--num-reruns", type=int, default=10)
    parser.add_argument("--rerun-id-for-plot", type=int, default=0)

    parser.add_argument("--distances-csv", type=str, default="results/part_b/distances.csv")
    parser.add_argument("--cov-csv", type=str, default="results/part_b/cov_summary.csv")
    parser.add_argument("--pairs-file", type=str, default="")
    parser.add_argument("--mc-samples", type=int, default=1)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_best_device(args.device)
    print(f"[setup] using device: {device}")

    train_loader, test_loader = get_mnist_subset_loaders(
        batch_size=args.batch_size,
        num_train_data=args.num_train_data,
        num_test_data=args.num_test_data,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
    )

    model = build_single_vae(
        latent_dim=args.latent_dim,
        obs_std=args.obs_std,
    ).to(device)

    ckpt = checkpoint_path(args.experiment_dir, "model.pt")

    if args.mode == "train":
        train_single_vae(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            noise_std=args.noise_std,
        )
        save_checkpoint(model, ckpt)
        print(f"[train] saved checkpoint to {ckpt}")

    elif args.mode == "eval":
        load_checkpoint(model, ckpt, device)
        eval_single_vae(model, test_loader, device)

    elif args.mode == "sample":
        load_checkpoint(model, ckpt, device)
        sample_and_reconstruct(
            model=model,
            test_loader=test_loader,
            device=device,
            out_dir=args.results_dir,
            num_samples=64,
        )

    elif args.mode == "part_a":
        load_checkpoint(model, ckpt, device)
        run_part_a(
            model=model,
            test_loader=test_loader,
            device=device,
            results_dir=args.results_dir,
            num_pairs=args.num_pairs,
            num_curve_points=args.num_curve_points,
            geodesic_epochs=args.geodesic_epochs,
            geodesic_lr=args.geodesic_lr,
            seed=args.seed,
        )

    elif args.mode == "plot_part_a":
        output_path = Path(args.results_dir) / "geodesics_partA.pdf"
        make_part_a_plot(
            results_dir=args.results_dir,
            output_path=output_path,
            show=True,
        )

    elif args.mode == "train_ensemble":
        experiment_dir = ensure_dir(args.experiment_dir)

        for rerun_id in range(args.num_reruns):
            print(f"[train_ensemble] starting rerun {rerun_id+1}/{args.num_reruns}")

            ensemble_model = build_multi_decoder_vae(
                latent_dim=args.latent_dim,
                obs_std=args.obs_std,
                num_decoders=args.num_decoders,
            ).to(device)

            train_multi_decoder_vae(
                model=ensemble_model,
                train_loader=train_loader,
                device=device,
                epochs_per_decoder=args.epochs,
                lr=args.lr,
                noise_std=args.noise_std,
            )

            ckpt_path = Path(args.experiment_dir) / f"model_rerun_{rerun_id}.pt"
            save_checkpoint(ensemble_model, ckpt_path)
            print(f"[train_ensemble] saved checkpoint to {ckpt_path}")

    elif args.mode == "part_b":
        results_dir = ensure_dir(args.results_dir)

        test_images, test_labels = get_dataset_tensors_from_loader(test_loader)

        if args.pairs_file:
            pair_dict = torch.load(args.pairs_file, map_location="cpu")
            print(f"[part_b] loaded fixed pairs from {args.pairs_file}")
        else:
            fixed_pairs_path = results_dir / "fixed_pairs.pt"
            if fixed_pairs_path.exists():
                pair_dict = torch.load(fixed_pairs_path, map_location="cpu")
                print(f"[part_b] loaded fixed pairs from {fixed_pairs_path}")
            else:
                pair_dict = select_fixed_point_pairs(
                    images=test_images,
                    labels=test_labels,
                    num_pairs=args.num_pairs,
                    seed=args.seed,
                )
                torch.save(pair_dict, fixed_pairs_path)
                print(f"[part_b] saved fixed pairs to {fixed_pairs_path}")

        models = []
        for rerun_id in range(args.num_reruns):
            ckpt_path = Path(args.experiment_dir) / f"model_rerun_{rerun_id}.pt"

            ensemble_model = build_multi_decoder_vae(
                latent_dim=args.latent_dim,
                obs_std=args.obs_std,
                num_decoders=args.num_decoders,
            ).to(device)

            load_checkpoint(ensemble_model, ckpt_path, device)
            models.append(ensemble_model)

        distances_df = collect_distances_across_models(
            models=models,
            test_loader=test_loader,
            pair_index_dict=pair_dict,
            device=device,
            num_curve_points=args.num_curve_points,
            geodesic_epochs=args.geodesic_epochs,
            geodesic_lr=args.geodesic_lr,
            num_decoders_list=range(1, args.num_decoders + 1),
            num_mc_samples=args.mc_samples,
        )
        save_distances_csv(distances_df, args.distances_csv)

        cov_df = compute_cov_summary(distances_df)
        save_cov_summary_csv(cov_df, args.cov_csv)

        model_for_plot = models[args.rerun_id_for_plot]
        build_part_b_geodesic_artifacts(
            model=model_for_plot,
            test_loader=test_loader,
            pair_index_dict=pair_dict,
            device=device,
            results_dir=results_dir,
            num_pairs=args.num_pairs,
            num_curve_points=args.num_curve_points,
            geodesic_epochs=args.geodesic_epochs,
            geodesic_lr=args.geodesic_lr,
            num_decoders_used=args.num_decoders,
            num_mc_samples=args.mc_samples,
        )

    elif args.mode == "plot_part_b":
        output_path = Path(args.results_dir) / "geodesics_partB.pdf"
        make_part_b_plot(
            results_dir=args.results_dir,
            output_path=output_path,
            show=True,
        )

    elif args.mode == "plot_cov":
        output_path = Path(args.results_dir) / "cov.pdf"
        make_cov_plot_from_csv(
            csv_path=args.cov_csv,
            output_path=output_path,
            show=True,
        )


if __name__ == "__main__":
    main()
