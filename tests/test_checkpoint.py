from pathlib import Path

import numpy as np
import pytest
import torch

from s2rspc.config import ModelConfig
from s2rspc.engine.checkpoint import load_checkpoint, save_checkpoint
from s2rspc.models import ARCHITECTURE_ID, SBPTNet
from s2rspc.preprocessing.tokenization import (
    STRUCTURAL_DESCRIPTOR_NAMES,
    TOKEN_ATTRIBUTE_NAMES,
)


def test_checkpoint_round_trip_preserves_release_contract(tmp_path):
    model = SBPTNet(ModelConfig())
    mean = np.arange(6, dtype=np.float32)
    std = np.arange(1, 7, dtype=np.float32)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        attribute_mean=mean,
        attribute_std=std,
        metadata={
            "seed": 42,
            "threshold": 0.4,
            "selection_rule": "synthetic validation",
        },
    )
    loaded, loaded_mean, loaded_std, state = load_checkpoint(path, ModelConfig())
    assert state["architecture_id"] == ARCHITECTURE_ID
    assert state["descriptor_feature_names"] == list(STRUCTURAL_DESCRIPTOR_NAMES)
    assert state["token_attribute_names"] == list(TOKEN_ATTRIBUTE_NAMES)
    np.testing.assert_allclose(loaded_mean, mean)
    np.testing.assert_allclose(loaded_std, std)
    assert state["fusion"]["beta"] == 0.5
    assert state["fusion"]["clip_bound"] == 3.0
    assert state["deviation_clip_bound"] == 10.0
    assert isinstance(loaded, SBPTNet)


def test_legacy_contextual_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"architecture_id": "SBPT-Net-OrderedPT1-Compact16D-H96"}, path)
    with pytest.raises(RuntimeError, match="architecture"):
        load_checkpoint(path, ModelConfig())
