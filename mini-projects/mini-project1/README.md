# Mini-project 1 (02460) 

This repo is a clean `.py` project layout based on the provided course code (weeks 1–3) + project files.

## Quick start (CPU-friendly, Mac Intel)
```bash
# from repo root
python -m scripts.run_partA --fast
python -m scripts.run_partB --fast
```

Outputs (figures + JSON metrics) are written under `outputs/`.

## Notes
- `fid.py` + `mnist_classifier.pth` are instructor-provided and are used exactly as required for FID.
- For final results, run without `--fast` and increase epochs / sample counts.
