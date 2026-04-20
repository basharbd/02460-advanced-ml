from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from curves import PolynomialCurve, LinearCurve
from geodesics import (
    curve_energy_ensemble,
    optimize_geodesic,
    estimate_geodesic_distance_from_energy,
)
from utils import (
    get_latent_means,
    materialize_pairs_from_latents,
    ensure_dir,
)


def coefficient_of_variation(values) -> float:
    values = pd.Series(values).astype(float)
    mean = values.mean()
    std = values.std(ddof=1)
    if abs(mean) < 1e-12:
        return float("nan")
    return float(std / mean)


def compute_cov_summary(distances_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns expected:
      - rerun_id
      - num_decoders_used
      - pair_id
      - euclidean_distance
      - geodesic_distance
    """
    rows = []

    for d in sorted(distances_df["num_decoders_used"].unique()):
        subset_d = distances_df[distances_df["num_decoders_used"] == d]

        pair_cov_rows = []
        for pair_id in sorted(subset_d["pair_id"].unique()):
            subset_pair = subset_d[subset_d["pair_id"] == pair_id]

            euc_cov = coefficient_of_variation(subset_pair["euclidean_distance"])
            geo_cov = coefficient_of_variation(subset_pair["geodesic_distance"])

            pair_cov_rows.append(
                {
                    "num_decoders_used": d,
                    "pair_id": pair_id,
                    "euclidean_cov": euc_cov,
                    "geodesic_cov": geo_cov,
                }
            )

        pair_cov_df = pd.DataFrame(pair_cov_rows)
        rows.append(
            {
                "num_decoders_used": d,
                "avg_euclidean_cov": pair_cov_df["euclidean_cov"].mean(),
                "avg_geodesic_cov": pair_cov_df["geodesic_cov"].mean(),
            }
        )

    return pd.DataFrame(rows)


def save_distances_csv(distances_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    distances_df.to_csv(path, index=False)
    print(f"[part_b] saved distances to {path}")


def save_cov_summary_csv(cov_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    cov_df.to_csv(path, index=False)
    print(f"[part_b] saved CoV summary to {path}")


def collect_distances_across_models(
    models: list[torch.nn.Module],
    test_loader,
    pair_index_dict: dict[str, torch.Tensor],
    device: str,
    num_curve_points: int = 20,
    geodesic_epochs: int = 600,
    geodesic_lr: float = 0.1,
    num_decoders_list: Iterable[int] = (1, 2, 3),
    num_mc_samples: int = 1,
) -> pd.DataFrame:
    rows = []

    for rerun_id, model in enumerate(models):
        model.eval()

        latent_means, _ = get_latent_means(model, test_loader, device)
        pairs = materialize_pairs_from_latents(latent_means, pair_index_dict, device=device)

        for num_decoders_used in num_decoders_list:
            for pair_id, (start, end) in enumerate(pairs):
                curve = PolynomialCurve(start, end, degree=3).to(device)

                energy_fn = lambda: curve_energy_ensemble(
                    curve=curve,
                    model=model,
                    num_points=num_curve_points,
                    num_mc_samples=num_mc_samples,
                    num_decoders_to_use=num_decoders_used,
                )

                curve, final_energy, _ = optimize_geodesic(
                    curve=curve,
                    energy_fn=energy_fn,
                    optimizer_name="adam",
                    lr=geodesic_lr,
                    epochs=geodesic_epochs,
                    verbose=False,
                )

                geo_distance = estimate_geodesic_distance_from_energy(final_energy)
                euc_distance = torch.norm(end - start, p=2).item()

                rows.append(
                    {
                        "rerun_id": rerun_id,
                        "num_decoders_used": num_decoders_used,
                        "pair_id": pair_id,
                        "euclidean_distance": euc_distance,
                        "geodesic_distance": geo_distance,
                    }
                )

                print(
                    f"[part_b] rerun={rerun_id} D={num_decoders_used} "
                    f"pair={pair_id} euc={euc_distance:.4f} geo={geo_distance:.4f}"
                )

    return pd.DataFrame(rows)


def build_part_b_geodesic_artifacts(
    model: torch.nn.Module,
    test_loader,
    pair_index_dict: dict[str, torch.Tensor],
    device: str,
    results_dir: str | Path,
    num_pairs: int = 25,
    num_curve_points: int = 20,
    geodesic_epochs: int = 600,
    geodesic_lr: float = 0.1,
    num_decoders_used: int = 3,
    num_mc_samples: int = 1,
) -> None:
    """
    Create the Part B latent/geodesic visualization artifacts for one chosen
    ensemble model.
    """
    results_dir = ensure_dir(results_dir)

    latent_means, latent_labels = get_latent_means(model, test_loader, device)
    pairs = materialize_pairs_from_latents(latent_means, pair_index_dict, device=device)

    linear_curves = []
    geodesic_curves = []
    distances = []

    for i, (start, end) in enumerate(pairs[:num_pairs]):
        print(f"[part_b] plotting curve {i+1}/{min(num_pairs, len(pairs))}")

        curve = PolynomialCurve(start, end, degree=3).to(device)

        energy_fn = lambda: curve_energy_ensemble(
            curve=curve,
            model=model,
            num_points=num_curve_points,
            num_mc_samples=num_mc_samples,
            num_decoders_to_use=num_decoders_used,
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
        geodesic_curves.append(curve(t_vis).detach().cpu())

    torch.save(latent_means, results_dir / "latent_means.pt")
    torch.save(latent_labels, results_dir / "latent_labels.pt")
    torch.save(torch.stack(linear_curves), results_dir / "linear_curves.pt")
    torch.save(torch.stack(geodesic_curves), results_dir / "geodesic_curves.pt")
    torch.save(torch.tensor(distances), results_dir / "geodesic_distances.pt")
    torch.save(pair_index_dict, results_dir / "fixed_pairs.pt")

    print(f"[part_b] saved plotting artifacts to {results_dir}")
