from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def load_part_a_artifacts(results_dir: str | Path) -> dict[str, torch.Tensor]:
    """
    Load saved tensors produced by Part A / Part B plotting pipelines.
    """
    results_dir = Path(results_dir)

    artifacts = {
        "latent_means": torch.load(results_dir / "latent_means.pt", map_location="cpu"),
        "latent_labels": torch.load(results_dir / "latent_labels.pt", map_location="cpu"),
        "linear_curves": torch.load(results_dir / "linear_curves.pt", map_location="cpu"),
        "geodesic_curves": torch.load(results_dir / "geodesic_curves.pt", map_location="cpu"),
        "geodesic_distances": torch.load(results_dir / "geodesic_distances.pt", map_location="cpu"),
        "fixed_pairs": torch.load(results_dir / "fixed_pairs.pt", map_location="cpu"),
    }
    return artifacts


def _unique_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen and l is not None:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), frameon=True)


def plot_latent_with_geodesics(
    latent_means: torch.Tensor,
    latent_labels: torch.Tensor,
    linear_curves: torch.Tensor | None = None,
    geodesic_curves: torch.Tensor | None = None,
    title: str = "Geodesics in latent space",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    if latent_means.shape[1] != 2:
        raise ValueError("This plot expects a 2D latent space.")

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        latent_means[:, 0],
        latent_means[:, 1],
        c=latent_labels,
        cmap="tab10",
        s=18,
        alpha=0.8,
        label="Latent means",
    )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class label")

    if linear_curves is not None:
        for i, curve in enumerate(linear_curves):
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                "--",
                linewidth=2.0,
                alpha=0.8,
                color="lightcoral",
                label="Straight line" if i == 0 else None,
                zorder=2,
            )

    if geodesic_curves is not None:
        for i, curve in enumerate(geodesic_curves):
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                "-",
                linewidth=2.0,
                alpha=0.9,
                color="deepskyblue",
                label="Geodesic" if i == 0 else None,
                zorder=3,
            )

    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _unique_legend(ax)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot] saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_part_a_plot(
    results_dir: str | Path,
    output_path: str | Path,
    show: bool = True,
) -> None:
    artifacts = load_part_a_artifacts(results_dir)

    plot_latent_with_geodesics(
        latent_means=artifacts["latent_means"],
        latent_labels=artifacts["latent_labels"],
        linear_curves=artifacts["linear_curves"],
        geodesic_curves=artifacts["geodesic_curves"],
        title="Part A: Pull-back geodesics",
        save_path=output_path,
        show=show,
    )


def make_part_b_plot(
    results_dir: str | Path,
    output_path: str | Path,
    show: bool = True,
) -> None:
    artifacts = load_part_a_artifacts(results_dir)

    plot_latent_with_geodesics(
        latent_means=artifacts["latent_means"],
        latent_labels=artifacts["latent_labels"],
        linear_curves=artifacts["linear_curves"],
        geodesic_curves=artifacts["geodesic_curves"],
        title="Part B: Ensemble VAE geodesics",
        save_path=output_path,
        show=show,
    )


def plot_cov_curve(
    cov_df: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        cov_df["num_decoders_used"],
        cov_df["avg_euclidean_cov"],
        marker="o",
        linewidth=2.0,
        label="Euclidean distance",
    )
    ax.plot(
        cov_df["num_decoders_used"],
        cov_df["avg_geodesic_cov"],
        marker="o",
        linewidth=2.0,
        label="Geodesic distance",
    )

    ax.set_xlabel("Number of ensemble decoders")
    ax.set_ylabel("Average CoV across point pairs")
    ax.set_title("Part B: Coefficient of variation")
    ax.set_xticks(sorted(cov_df["num_decoders_used"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot] saved CoV figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_cov_plot_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    show: bool = True,
) -> None:
    cov_df = pd.read_csv(csv_path)
    plot_cov_curve(cov_df, save_path=output_path, show=show)
