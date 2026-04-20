from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

def plot_prior_vs_agg_posterior(
    prior_samples: torch.Tensor,
    posterior_samples: torch.Tensor,
    outpath: str | Path,
    title: str = "",
) -> None:
    """Scatter plot: prior vs aggregate posterior.

    If dim>2, we PCA-project to 2D for visualization.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    Xp = prior_samples.detach().cpu().numpy()
    Xq = posterior_samples.detach().cpu().numpy()

    if Xp.shape[1] > 2:
        pca = PCA(n_components=2)
        pca.fit(np.concatenate([Xp, Xq], axis=0))
        Xp = pca.transform(Xp)
        Xq = pca.transform(Xq)

    plt.figure(figsize=(6, 5))
    plt.scatter(Xp[:,0], Xp[:,1], s=6, alpha=0.5, label="prior p(z)")
    plt.scatter(Xq[:,0], Xq[:,1], s=6, alpha=0.5, label="aggregate posterior q(z)")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
