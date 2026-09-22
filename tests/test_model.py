from pathlib import Path

import torch

from s2rspc.config import load_config
from s2rspc.models import ARCHITECTURE_ID, SBPTNet
from s2rspc.models.pointmlp import PointMLP
from s2rspc.models.sbptnet import StructuralEncoder12D, fuse_logits


def test_configuration_and_architecture_match_manuscript():
    cfg = load_config(Path("configs/sbptnet_sts2r.yaml"))
    assert ARCHITECTURE_ID == "SBPT-Net-PointMLP-Structural12D-H96"
    assert cfg.model.pointmlp_input_dim == 6
    assert cfg.model.structural_dim == 12
    assert cfg.model.structural_hidden_dim == 96
    assert cfg.model.fusion_beta == 0.5
    assert cfg.model.fusion_clip_bound == 3.0
    assert cfg.model.deviation_clip_bound == 10.0
    assert cfg.training.attribute_scale_low == 0.75
    assert cfg.training.attribute_scale_high == 1.25
    assert cfg.training.attribute_shift_std == 0.20


def test_pointmlp_and_center_only_structural_forward_shapes():
    cfg = load_config(Path("configs/sbptnet_sts2r.yaml"))
    backbone = PointMLP(num_classes=2, input_channels=6, points=64).eval()
    point_features = torch.randn(2, 6, 64)
    with torch.no_grad():
        logits, auxiliary = backbone(point_features)
    assert auxiliary is None
    assert logits.shape == (2, 2, 64)

    encoder = StructuralEncoder12D(cfg.model).eval()
    assert list(encoder.state_dict()) == [
        "embed.0.weight",
        "embed.0.bias",
        "embed.1.weight",
        "embed.1.bias",
        "out.weight",
        "out.bias",
    ]
    structural = torch.randn(2, 64, 12)
    with torch.no_grad():
        structural_logit = encoder(structural)
    assert structural_logit.shape == (2, 64)
    assert not any(
        "context" in name or "neighbor" in name for name, _ in encoder.named_modules()
    )


def test_fusion_uses_symbolic_bound_and_fixed_beta():
    initial = torch.tensor([-8.0, -2.0, 4.0])
    structural = torch.tensor([1.0, 2.0, 3.0])
    actual = fuse_logits(initial, structural, beta=0.5, clip_bound=3.0)
    expected = 0.5 * initial.clamp(-3.0, 3.0) + 0.5 * structural
    torch.testing.assert_close(actual, expected)


def test_unified_model_has_no_contextual_refiner():
    cfg = load_config(Path("configs/sbptnet_sts2r.yaml"))
    model = SBPTNet(cfg.model).eval()
    assert not hasattr(model, "context_refiner")
    point_features = torch.randn(2, 6, 64)
    structural = torch.randn(2, 64, 12)
    with torch.no_grad():
        initial, structural_logit, final = model(point_features, structural)
    assert initial.shape == structural_logit.shape == final.shape == (2, 64)
