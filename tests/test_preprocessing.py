from pathlib import Path

import numpy as np

from s2rspc.cli.toy_demo import make_toy_cloud
from s2rspc.config import load_config
from s2rspc.preprocessing.pipeline import preprocess_point_cloud
from s2rspc.preprocessing.superline import orthogonal_tangent_residual
from s2rspc.preprocessing.tokenization import (
    STRUCTURAL_DESCRIPTOR_NAMES,
    TOKEN_ATTRIBUTE_NAMES,
)


def test_tangent_residual_is_sign_invariant():
    displacement = np.array([1.2, -0.7, 0.3], dtype=np.float64)
    tangent = np.array([0.4, 0.8, -0.2], dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    reference = orthogonal_tangent_residual(displacement, tangent)
    assert np.isclose(reference, orthogonal_tangent_residual(-displacement, tangent))
    assert np.isclose(reference, orthogonal_tangent_residual(displacement, -tangent))
    assert np.isclose(reference, orthogonal_tangent_residual(-displacement, -tangent))


def test_toy_preprocessing_emits_paper_aligned_cache(tmp_path):
    raw = tmp_path / "toy.txt"
    make_toy_cloud(raw)
    cfg = load_config(Path("configs/sbptnet_sts2r.yaml"))
    candidate, tokenized = preprocess_point_cloud(raw, cfg)
    assert 0 < len(candidate["xyz"]) < int(candidate["raw_count"])
    count = len(tokenized["label"])
    assert count > 0
    assert tokenized["schema_version"].item() == 2
    assert tokenized["structural_descriptors"].shape == (count, 6)
    assert tokenized["token_attributes"].shape == (count, 6)
    assert len(STRUCTURAL_DESCRIPTOR_NAMES) == len(TOKEN_ATTRIBUTE_NAMES) == 6
    assert "patch_points" not in tokenized
    assert "neighbor_index" not in tokenized
    assert len(tokenized["raw_label"]) == int(tokenized["raw_count"])
