"""Official OpenPoints PointNeXt-S semantic-segmentation architecture.

The encoder, decoder, segmentation head, and CUDA neighborhood operators are
vendored from the official PointNeXt/OpenPoints implementation.  This wrapper
only adapts the paper benchmark interface:

* input:  XYZRGB tensors in ``[B, 6, N]`` or ``[B, N, 6]`` format;
* output: binary point-wise logits in ``[B, 2, N]`` format;
* coordinates: the same centimetre-scale conversion used by the retained
  PointNet++ comparison implementation.

Changing the input feature count and the final class count is a standard task
adaptation and does not alter the PointNeXt backbone architecture.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_ROOT = _PACKAGE_ROOT / "vendor"
_POINTNET2_EXTENSION_ROOT = (
    _VENDOR_ROOT / "openpoints" / "cpp" / "pointnet2_batch"
)
for _path in (_VENDOR_ROOT, _POINTNET2_EXTENSION_ROOT):
    _path_string = str(_path)
    if _path_string not in sys.path:
        sys.path.insert(0, _path_string)

try:
    from openpoints.models.backbone.pointnext import (
        PointNextDecoder,
        PointNextEncoder,
    )
    from openpoints.models.segmentation.base_seg import SegHead
    from openpoints.utils import EasyConfig
except Exception as exc:  # pragma: no cover - emits an actionable runtime error
    raise ImportError(
        "The vendored official OpenPoints PointNeXt implementation could not "
        "be imported. Build vendor/openpoints/cpp/pointnet2_batch first."
    ) from exc


class PointNeXt(nn.Module):
    """PointNeXt-S with the official semantic-segmentation decoder and head."""

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 6,
        coordinate_scale: float = 100.0,
    ):
        super().__init__()
        if input_channels != 6:
            raise ValueError(
                "The formal PointNeXt benchmark expects XYZRGB input "
                f"(6 channels), got {input_channels}."
            )
        self.input_channels = input_channels
        self.coordinate_scale = float(coordinate_scale)

        group_args = EasyConfig(NAME="ballquery", normalize_dp=True)
        self.encoder = PointNextEncoder(
            in_channels=3,
            width=32,
            blocks=[1, 1, 1, 1, 1],
            strides=[1, 4, 4, 4, 4],
            sa_layers=2,
            sa_use_res=True,
            expansion=4,
            radius=0.1,
            nsample=32,
            aggr_args={"feature_type": "dp_fj", "reduction": "max"},
            group_args=group_args,
            conv_args={"order": "conv-norm-act"},
            act_args={"act": "relu"},
            norm_args={"norm": "bn"},
        )
        self.decoder = PointNextDecoder(
            encoder_channel_list=self.encoder.channel_list,
            in_channels=3,
        )
        self.head = SegHead(
            num_classes=num_classes,
            in_channels=self.decoder.out_channels,
            norm_args={"norm": "bn"},
            act_args={"act": "relu"},
        )

    @staticmethod
    def _pad_short_input(x: torch.Tensor, minimum_points: int = 256):
        """Deterministically pad only very short final evaluation chunks."""
        original_points = int(x.shape[-1])
        if original_points >= minimum_points:
            return x, original_points
        repeats = (minimum_points + original_points - 1) // original_points
        padded = x.repeat(1, 1, repeats)[..., :minimum_points]
        return padded, original_points

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Expected a 3-D point tensor, got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels and x.shape[2] == self.input_channels:
            x = x.transpose(1, 2).contiguous()
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {tuple(x.shape)}"
            )

        x, original_points = self._pad_short_input(x)
        positions = (
            x[:, :3, :].transpose(1, 2).contiguous() / self.coordinate_scale
        )
        rgb_features = x[:, 3:6, :].contiguous()

        positions_by_stage, features_by_stage = self.encoder.forward_seg_feat(
            positions,
            rgb_features,
        )
        decoded = self.decoder(
            positions_by_stage,
            features_by_stage,
        ).squeeze(-1)
        logits = self.head(decoded)
        logits = logits[..., :original_points]
        return logits, None

