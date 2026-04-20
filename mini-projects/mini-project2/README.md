# Mini-project 2 — Advanced Machine Learning (02460)

This archive contains an implementation for Mini-project 2 in Advanced Machine Learning (02460).

## Project scope

We work on a non-binarized MNIST subset with classes `{0,1,2}` and 2048 observations.

- **Part A**: compute pull-back geodesics induced by the **mean** of a Gaussian decoder.
- **Part B**: extend the model with an **ensemble of decoders**, compute ensemble geodesics, and compare distance reliability using the **coefficient of variation (CoV)** across retrainings.

## Files

- `main.py`  
  Main entry point for training, evaluation, Part A, and Part B.

- `data.py`  
  Loads the MNIST subset and adds the mild training noise used in the handout.

- `models.py`  
  Defines the Gaussian VAE and ensemble-decoder VAE.

- `curves.py`  
  Defines linear and polynomial latent-space curves.

- `geodesics.py`  
  Implements discrete curve-energy approximations and geodesic optimization.

- `analysis_cov.py`  
  Computes Euclidean/geodesic distances across retrainings and summarizes CoV.

- `plotting.py`  
  Produces the final figures for Part A, Part B, and CoV.

- `utils.py`  
  Seeds, checkpointing, fixed-pair selection, latent extraction, and device helpers.

## Installation

Use Python 3.10+ if possible.

```bash
pip install -r requirements.txt
```

## Device notes

On a 2019 Intel MacBook Pro, `mps` is generally **not** available. Start with:

```bash
--device cpu
```

If you run the code elsewhere and `mps` or `cuda` are available, you can use those.

## Part A

### 1) Train a single-decoder VAE

```bash
python main.py train --device cpu --epochs 50 --experiment-dir experiments/single
```

### 2) Evaluate ELBO

```bash
python main.py eval --device cpu --experiment-dir experiments/single
```

### 3) Generate samples and reconstructions

```bash
python main.py sample --device cpu --experiment-dir experiments/single --results-dir results/sample
```

### 4) Compute pull-back geodesics

```bash
python main.py part_a --device cpu --experiment-dir experiments/single --results-dir results/part_a
```

### 5) Plot the Part A figure

```bash
python main.py plot_part_a --results-dir results/part_a
```

This creates:

- `results/part_a/geodesics_partA.pdf`

## Part B

### 1) Train ensemble-decoder VAE retrainings

Example with 3 decoders and 10 retrainings:

```bash
python main.py train_ensemble \
  --device cpu \
  --epochs 50 \
  --num-decoders 3 \
  --num-reruns 10 \
  --experiment-dir experiments/ensemble
```

### 2) Compute distances, CoV, and Part B artifacts

If you want Part B to reuse the exact same pairs as Part A:

```bash
python main.py part_b \
  --device cpu \
  --num-decoders 3 \
  --num-reruns 10 \
  --experiment-dir experiments/ensemble \
  --results-dir results/part_b \
  --pairs-file results/part_a/fixed_pairs.pt \
  --distances-csv results/part_b/distances.csv \
  --cov-csv results/part_b/cov_summary.csv
```

If `--pairs-file` is omitted, the code creates/loads fixed pairs inside `results/part_b/`.

### 3) Plot Part B geodesics

```bash
python main.py plot_part_b --results-dir results/part_b
```

### 4) Plot CoV

```bash
python main.py plot_cov \
  --results-dir results/part_b \
  --cov-csv results/part_b/cov_summary.csv
```

This creates:

- `results/part_b/geodesics_partB.pdf`
- `results/part_b/cov.pdf`

## Notes

- The implementation uses **latent posterior means** when comparing distances across retrainings.
- The same fixed dataset-level point pairs can be reused across retrainings via `fixed_pairs.pt`.
- Geodesics are computed by **direct numerical energy minimization**.
- For Part A, the energy uses the **decoder mean**.
- For Part B, the energy uses a **Monte Carlo average over decoder means** from the ensemble.


