# SBPT-Net

Official implementation for the manuscript:

**Superline-Based Synthetic-to-Real Domain Generalization for Gauge Line Segmentation in Shoe Upper Point Clouds**

SBPT-Net represents sparse gauge-line candidates as superline-based point tokens. This repository contains the current preprocessing, model, training, evaluation, and inference code. Dataset files, experiment directories, predictions, and checkpoints are distributed separately.

The Python package remains named `s2rspc` for compatibility with earlier local scripts.

## Current method

1. Local color contrast retains a high-recall candidate set.
2. A spacing-normalized graph groups candidates into superlines. Edge ranking uses the sign-invariant orthogonal tangent residual described by Eq. (2) in the manuscript.
3. Local support regions produce observed representative points and overlapping token memberships.
4. PointMLP predicts an initial logit from centered token XYZ and mean RGB (6D).
5. A center-only encoder maps six structural descriptors and six normalized token attributes (12D) to a structural logit.
6. The final logit uses the fixed fusion rule in the manuscript. Token probabilities are projected to candidate points by averaging all overlapping support-region votes, then scattered to original point indices.

There is no contextual token aggregation module in the current architecture.

## Installation

Python 3.9 or newer and PyTorch 2.1 or newer are required. Install the PyTorch build suitable for your CUDA environment, then run:

```bash
python -m pip install -e .
```

For development checks:

```bash
python -m pip install -e ".[dev]"
pytest
python tools/verify_release.py
```

## Quick software check

The toy demo generates a small artificial point cloud and exercises preprocessing and a forward pass:

```bash
sbptnet-toy-demo --config configs/sbptnet_sts2r.yaml --output toy_output
```

Toy outputs are software checks and are not paper results.

## Data

Raw point clouds are whitespace-separated text files with seven columns:

```text
X Y Z R G B Label
```

Place files under `data/raw/<subset>/`, then preprocess them:

```bash
sbptnet-preprocess --config configs/sbptnet_sts2r.yaml --input-root data/raw
```

The STS2R dataset v1 is available from [Zenodo](https://doi.org/10.5281/zenodo.19528228). Synthetic-data generation is maintained in [MoXiang97/STS2R-code](https://github.com/MoXiang97/STS2R-code).

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the cache schema.

## Training

```bash
sbptnet-train \
  --config configs/sbptnet_sts2r.yaml \
  --token-root data/tokens \
  --output runs/sbptnet_seed42
```

The training command uses synthetic training and validation token files. It first trains PointMLP, freezes it, fits normalization statistics on synthetic training data, and then trains the 12D structural encoder. Checkpoint and operating-threshold selection use synthetic validation data.

The architecture was refined during experiments in which earlier real-set results had already been observed. The real scans should therefore not be interpreted as a development-independent untouched test set. See [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Evaluation

```bash
sbptnet-evaluate \
  --config configs/sbptnet_sts2r.yaml \
  --checkpoint weights/sbptnet_best.pt \
  --token-root data/tokens \
  --output runs/evaluation
```

Metrics are computed in the original point-cloud space. This repository does not include an experiment archive or provisional paper-result tables. Those results should only be published with the matching frozen checkpoint, manifest, and completed validation record.

## Inference

```bash
sbptnet-infer \
  --config configs/sbptnet_sts2r.yaml \
  --checkpoint weights/sbptnet_best.pt \
  --input example.txt \
  --output prediction.npz
```

## License and citation

Repository-authored code is released under the MIT License. The adapted PointMLP implementation remains under Apache-2.0; see [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md). Citation metadata is provided in `CITATION.cff`.
