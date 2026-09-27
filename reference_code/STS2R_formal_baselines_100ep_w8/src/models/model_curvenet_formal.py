"""Formal CurveNet adaptation for binary candidate-point segmentation.

The curve grouping, curve aggregation, CIC, LPFA, and feature-propagation
operators are retained from the official CurveNet implementation. The
ShapeNetPart category-conditioning branch is removed because STS2R has one
binary task, RGB is embedded alongside the original geometric LPFA stem, and
one final feature-propagation layer restores predictions from the first
2,048-point CurveNet stage to all 4,096 input candidates.
"""

import torch
import torch.nn as nn

from .curvenet_vendor.curvenet_util import CIC, LPFA, PointNetFeaturePropagation


CURVE_CONFIG = {
    "default": [[100, 5], [100, 5], None, None, None],
}


class CurveNetSeg(nn.Module):
    """CurveNet part-segmentation backbone adapted to XYZRGB binary output."""

    def __init__(self, num_classes=2, input_channels=6, k=32, setting="default"):
        super().__init__()
        if input_channels < 6:
            raise ValueError("CurveNetSeg requires XYZRGB input (six channels).")
        if setting not in CURVE_CONFIG:
            raise ValueError(f"Unknown CurveNet setting: {setting}")

        additional_channel = 32
        curve_config = CURVE_CONFIG[setting]

        # Official geometric stem plus a minimal RGB embedding. Both produce
        # 32 channels and are fused before the CurveNet encoder.
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)
        self.rgb_stem = nn.Sequential(
            nn.Conv1d(3, additional_channel, 1, bias=False),
            nn.BatchNorm1d(additional_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Official ShapeNetPart CurveNet encoder.
        self.cic11 = CIC(
            npoint=2048, radius=0.2, k=k, in_channels=additional_channel,
            output_channels=64, bottleneck_ratio=2,
            curve_config=curve_config[0],
        )
        self.cic12 = CIC(
            npoint=2048, radius=0.2, k=k, in_channels=64,
            output_channels=64, bottleneck_ratio=4,
            curve_config=curve_config[0],
        )
        self.cic21 = CIC(
            npoint=512, radius=0.4, k=k, in_channels=64,
            output_channels=128, bottleneck_ratio=2,
            curve_config=curve_config[1],
        )
        self.cic22 = CIC(
            npoint=512, radius=0.4, k=k, in_channels=128,
            output_channels=128, bottleneck_ratio=4,
            curve_config=curve_config[1],
        )
        self.cic31 = CIC(
            npoint=128, radius=0.8, k=k, in_channels=128,
            output_channels=256, bottleneck_ratio=2,
            curve_config=curve_config[2],
        )
        self.cic32 = CIC(
            npoint=128, radius=0.8, k=k, in_channels=256,
            output_channels=256, bottleneck_ratio=4,
            curve_config=curve_config[2],
        )
        self.cic41 = CIC(
            npoint=32, radius=1.2, k=31, in_channels=256,
            output_channels=512, bottleneck_ratio=2,
            curve_config=curve_config[3],
        )
        self.cic42 = CIC(
            npoint=32, radius=1.2, k=31, in_channels=512,
            output_channels=512, bottleneck_ratio=4,
            curve_config=curve_config[3],
        )
        self.cic51 = CIC(
            npoint=8, radius=2.0, k=7, in_channels=512,
            output_channels=1024, bottleneck_ratio=2,
            curve_config=curve_config[4],
        )
        self.cic52 = CIC(
            npoint=8, radius=2.0, k=7, in_channels=1024,
            output_channels=1024, bottleneck_ratio=4,
            curve_config=curve_config[4],
        )
        self.cic53 = CIC(
            npoint=8, radius=2.0, k=7, in_channels=1024,
            output_channels=1024, bottleneck_ratio=4,
            curve_config=curve_config[4],
        )

        # Official decoder through the 2,048-point first CurveNet stage.
        self.fp4 = PointNetFeaturePropagation(
            in_channel=1024 + 512, mlp=[512, 512], att=[1024, 512, 256]
        )
        self.up_cic5 = CIC(
            npoint=32, radius=1.2, k=31, in_channels=512,
            output_channels=512, bottleneck_ratio=4,
        )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=512 + 256, mlp=[256, 256], att=[512, 256, 128]
        )
        self.up_cic4 = CIC(
            npoint=128, radius=0.8, k=k, in_channels=256,
            output_channels=256, bottleneck_ratio=4,
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=256 + 128, mlp=[128, 128], att=[256, 128, 64]
        )
        self.up_cic3 = CIC(
            npoint=512, radius=0.4, k=k, in_channels=128,
            output_channels=128, bottleneck_ratio=4,
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 64, mlp=[64, 64], att=[128, 64, 32]
        )

        # Without ShapeNetPart's category one-hot vector:
        # XYZ(3) + local(64) + global stage-4(64) + global stage-5(128).
        self.up_cic2 = CIC(
            npoint=2048, radius=0.2, k=k, in_channels=259,
            output_channels=256, bottleneck_ratio=4,
        )
        self.up_cic1 = CIC(
            npoint=2048, radius=0.2, k=k, in_channels=256,
            output_channels=256, bottleneck_ratio=4,
        )

        self.global_conv2 = nn.Sequential(
            nn.Conv1d(1024, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.global_conv1 = nn.Sequential(
            nn.Conv1d(512, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # The formal protocol supplies 4,096 candidates, whereas the official
        # part-segmentation model uses 2,048. This restores all input points.
        self.fp0 = PointNetFeaturePropagation(
            in_channel=256 + additional_channel,
            mlp=[128, 64],
            att=None,
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1),
        )

    @staticmethod
    def _cic(module, xyz, features):
        # The current official repository optionally returns curve indices for
        # visualization. Formal segmentation only needs coordinates/features.
        output = module(xyz, features)
        return output[0], output[1]

    def forward(self, inputs):
        if inputs.ndim != 3:
            raise ValueError(f"Expected [B,C,N], got {tuple(inputs.shape)}")

        original_points = int(inputs.shape[-1])
        # CurveNet selects 100 curve seeds in its first two stages. Formal
        # full-candidate inference can end with a short deterministic chunk,
        # and zero-candidate files are represented by one harmless placeholder.
        # Repeat the last supplied point only inside the model and discard the
        # padded logits afterward; no padded prediction enters any metric.
        if original_points < 100:
            repeat = 100 - original_points
            pad_index = torch.arange(repeat, device=inputs.device) % original_points
            inputs = torch.cat((inputs, inputs.index_select(-1, pad_index)), dim=-1)

        # The STS2R caches store millimetre-scale coordinates. Dividing by 100
        # matches the coordinate scale used by the formal PointNet++ baseline.
        xyz0 = inputs[:, :3, :] / 100.0
        rgb = inputs[:, 3:6, :]
        l0_points = self.lpfa(xyz0, xyz0) + self.rgb_stem(rgb)

        l1_xyz, l1_points = self._cic(self.cic11, xyz0, l0_points)
        l1_xyz, l1_points = self._cic(self.cic12, l1_xyz, l1_points)
        l2_xyz, l2_points = self._cic(self.cic21, l1_xyz, l1_points)
        l2_xyz, l2_points = self._cic(self.cic22, l2_xyz, l2_points)
        l3_xyz, l3_points = self._cic(self.cic31, l2_xyz, l2_points)
        l3_xyz, l3_points = self._cic(self.cic32, l3_xyz, l3_points)
        l4_xyz, l4_points = self._cic(self.cic41, l3_xyz, l3_points)
        l4_xyz, l4_points = self._cic(self.cic42, l4_xyz, l4_points)
        l5_xyz, l5_points = self._cic(self.cic51, l4_xyz, l4_points)
        l5_xyz, l5_points = self._cic(self.cic52, l5_xyz, l5_points)
        l5_xyz, l5_points = self._cic(self.cic53, l5_xyz, l5_points)

        emb1 = self.global_conv1(l4_points).max(dim=-1, keepdim=True)[0]
        emb2 = self.global_conv2(l5_points).max(dim=-1, keepdim=True)[0]

        l4_points = self.fp4(l4_xyz, l5_xyz, l4_points, l5_points)
        l4_xyz, l4_points = self._cic(self.up_cic5, l4_xyz, l4_points)
        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_xyz, l3_points = self._cic(self.up_cic4, l3_xyz, l3_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_xyz, l2_points = self._cic(self.up_cic3, l2_xyz, l2_points)
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        global_features = torch.cat((emb1, emb2), dim=1)
        global_features = global_features.expand(-1, -1, l1_xyz.size(-1))
        x = torch.cat((l1_xyz, l1_points, global_features), dim=1)
        l1_xyz, x = self._cic(self.up_cic2, l1_xyz, x)
        l1_xyz, x = self._cic(self.up_cic1, l1_xyz, x)
        x = self.fp0(xyz0, l1_xyz, l0_points, x)
        return self.classifier(x)[..., :original_points]
