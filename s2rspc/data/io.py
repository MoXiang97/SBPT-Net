"""Portable point-cloud and paper-aligned token-cache formats."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from ..preprocessing.tokenization import TOKEN_CACHE_SCHEMA_VERSION


def pointmlp_record_seed(subset: str, sample_id: str) -> int:
    identity = f"{subset}/{sample_id}".encode("utf-8")
    return 42 + int(hashlib.sha256(identity).hexdigest()[:7], 16)

def load_raw_point_cloud(path: str | Path) -> Dict[str, np.ndarray]:
    """Load an ASCII point cloud with columns X Y Z R G B Label."""

    path = Path(path)
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Expected at least seven columns (XYZRGBLabel): {path}")
    return {
        "xyz": data[:, :3].astype(np.float32),
        "rgb": data[:, 3:6].astype(np.float32),
        "label": (data[:, 6] > 0.5).astype(np.int64),
        "raw_count": np.asarray(len(data), dtype=np.int64),
    }


def save_candidate_cloud(
    path: str | Path, candidate: Mapping[str, np.ndarray]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: np.asarray(value) for key, value in candidate.items()}
    )


def load_candidate_cloud(path: str | Path) -> Dict[str, np.ndarray]:
    """Load the public NPZ format or the historical nine-column NPY format."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] < 9:
            raise ValueError(
                f"Candidate NPY must contain XYZRGBLabelOrigIndexLCC: {path}"
            )
        return {
            "xyz": data[:, :3].astype(np.float32),
            "rgb": data[:, 3:6].astype(np.float32),
            "label": np.clip(data[:, 6], 0, 1).astype(np.int64),
            "orig_index": data[:, 7].astype(np.int64),
            "lcc": data[:, 8].astype(np.float32),
            "raw_count": np.asarray(int(data[:, 7].max()) + 1, dtype=np.int64),
        }
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def save_tokenized_cloud(
    path: str | Path, arrays: Mapping[str, np.ndarray]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: np.asarray(value) for key, value in arrays.items()}
    )


def load_tokenized_cloud(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays.setdefault('_record_seed', np.asarray(pointmlp_record_seed(path.parent.name, path.stem), dtype=np.int64))
    version = int(np.asarray(arrays.get("schema_version", -1)).item())
    if version != TOKEN_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Token cache {path} uses schema {version}; regenerate it with "
            f"the paper-aligned schema {TOKEN_CACHE_SCHEMA_VERSION}. "
            "Legacy 16D/19D/23D caches are intentionally rejected."
        )
    required = {
        "xyz",
        "rgb",
        "label",
        "orig_index",
        "fragment_id",
        "features",
        "structural_descriptors",
        "token_attributes",
        "patch_member_offsets",
        "patch_member_indices",
        "candidate_orig_index",
        "raw_count",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"Token file is missing fields {missing}: {path}")
    count = len(arrays["label"])
    if arrays["structural_descriptors"].shape != (count, 6):
        raise ValueError(f"Expected six structural descriptors in {path}")
    if arrays["token_attributes"].shape != (count, 6):
        raise ValueError(f"Expected six token attributes in {path}")
    return arrays
