import copy

import numpy as np
import torch

import s2rspc.engine.trainer as trainer
from s2rspc.config import ModelConfig, TrainingConfig
from s2rspc.engine.trainer import (
    binary_segmentation_loss,
    freeze_pointmlp,
    pointmlp_logits_for_cloud,
    structural_objective,
)
from s2rspc.models.sbptnet import SBPTNet


class _PointMLPStub:
    def eval(self):
        return self


class _ChunkRecordingModel:
    def __init__(self):
        self.pointmlp = _PointMLPStub()
        self.chunk_sizes = []

    def initial_logits(self, features):
        self.chunk_sizes.append(int(features.shape[-1]))
        value = float(len(self.chunk_sizes) - 1)
        return torch.full(
            (1, features.shape[-1]), value, dtype=features.dtype, device=features.device
        )


def test_structural_sampling_matches_frozen_without_replacement_rule():
    selected = trainer._structural_indices(
        count=12, size=256, rng=np.random.RandomState(42)
    )

    assert len(selected) == 12
    assert len(np.unique(selected)) == 12
    assert sorted(selected.tolist()) == list(range(12))

def test_pointmlp_cloud_inference_matches_frozen_4096_token_chunking():
    count = 5000
    cloud = {
        "xyz": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        "rgb": np.zeros((count, 3), dtype=np.float32),
    }
    model = _ChunkRecordingModel()

    actual = pointmlp_logits_for_cloud(model, cloud, torch.device("cpu"))

    assert model.chunk_sizes == [4096, 904]
    np.testing.assert_array_equal(actual[:4096], np.zeros(4096, dtype=np.float32))
    np.testing.assert_array_equal(actual[4096:], np.ones(904, dtype=np.float32))


class _RandomLogitModel:
    def __init__(self):
        self.pointmlp = _PointMLPStub()

    def initial_logits(self, features):
        return torch.rand(
            (1, features.shape[-1]), dtype=features.dtype, device=features.device
        )


def test_pointmlp_cloud_inference_uses_record_seed_without_changing_global_rng():
    cloud = {
        "xyz": np.arange(192, dtype=np.float32).reshape(64, 3),
        "rgb": np.zeros((64, 3), dtype=np.float32),
    }
    model = _RandomLogitModel()

    torch.manual_seed(7)
    expected_next = torch.rand(4)
    torch.manual_seed(7)
    first = pointmlp_logits_for_cloud(
        model, cloud, torch.device("cpu"), record_seed=12345
    )
    observed_next = torch.rand(4)
    second = pointmlp_logits_for_cloud(
        model, cloud, torch.device("cpu"), record_seed=12345
    )

    np.testing.assert_array_equal(first, second)
    torch.testing.assert_close(observed_next, expected_next)

def test_structural_objective_matches_equation_8():
    final = torch.tensor([-0.6, 0.8, 1.2], requires_grad=True)
    structural = torch.tensor([-0.2, 0.5, 1.0], requires_grad=True)
    initial = torch.tensor([-20.0, 0.25, 16.0])
    target = torch.tensor([0.0, 1.0, 1.0])
    cfg = TrainingConfig(
        dice_weight=0.5,
        structural_auxiliary_weight=0.1,
        deviation_weight=1e-4,
    )
    actual = structural_objective(
        final, structural, initial, target, cfg, deviation_clip_bound=10.0
    )
    manual = (
        binary_segmentation_loss(final, target, dice_weight=0.5)
        + 0.1 * binary_segmentation_loss(structural, target, dice_weight=0.5)
        + 1e-4 * torch.mean((final - initial.clamp(-10.0, 10.0)) ** 2)
    )
    torch.testing.assert_close(actual, manual)


def test_freeze_pointmlp_keeps_backbone_bit_identical():
    model = SBPTNet(ModelConfig())
    before = {
        name: value.detach().clone()
        for name, value in model.pointmlp.state_dict().items()
    }
    freeze_pointmlp(model)
    optimizer = torch.optim.AdamW(model.structural_encoder.parameters(), lr=1e-3)
    structural = torch.randn(2, 32, 12)
    initial = torch.randn(2, 32)
    target = torch.randint(0, 2, (2, 32)).float()
    structural_logit = model.structural_encoder(structural)
    final = model.fuse(initial, structural_logit)
    loss = structural_objective(
        final, structural_logit, initial, target, TrainingConfig(), 10.0
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = model.pointmlp.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(not parameter.requires_grad for parameter in model.pointmlp.parameters())
