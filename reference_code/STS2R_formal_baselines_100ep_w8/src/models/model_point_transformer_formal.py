"""Formal pure-PyTorch Point Transformer segmentation baseline.

This module is adapted from the existing STS2R
``src/models/model_point_transformer.py`` implementation. The encoder-decoder
topology, vector-attention equations, channel widths, block depths, transition
stages, and 16-neighbor definition are retained. Three implementation fixes are
made for the locked 4,096-point comparison:

1. vector-attention softmax is applied over the neighbor dimension;
2. exact kNN distance computation is center-chunked and uses ``topk`` rather
   than materializing/sorting a full N-by-N tensor;
3. a stage computes its geometric neighbor indices once and reuses them across
   all Point Transformer blocks at that resolution.

No STE-Net token, superline, structural descriptor, or context feature is used.
The model receives the same XYZRGB candidates as the other formal backbones.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def index_points(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather ``(B,N,C)`` points with an arbitrary ``(B,...)`` index tensor."""
    batch = points.shape[0]
    view_shape = [batch] + [1] * (index.ndim - 1)
    batch_index = torch.arange(batch, device=points.device).view(view_shape)
    batch_index = batch_index.expand_as(index)
    return points[batch_index, index]


def chunked_knn(
    query_xyz: torch.Tensor,
    support_xyz: torch.Tensor,
    k: int,
    query_chunk_size: int = 256,
) -> torch.Tensor:
    """Exact kNN with bounded temporary memory.

    Squared distances are evaluated for chunks of query centers. This changes
    neither the neighbor definition nor the returned nearest-neighbor order.
    """
    if query_xyz.ndim != 3 or support_xyz.ndim != 3:
        raise ValueError("query_xyz and support_xyz must have shape (B,N,3)")
    if query_xyz.shape[0] != support_xyz.shape[0]:
        raise ValueError("query and support batch sizes differ")
    support_n = int(support_xyz.shape[1])
    if support_n == 0:
        raise ValueError("support point set is empty")
    requested_k = max(1, int(k))
    effective_k = min(requested_k, support_n)
    support_sq = torch.sum(support_xyz * support_xyz, dim=-1).unsqueeze(1)
    support_t = support_xyz.transpose(1, 2).contiguous()
    parts = []
    chunk = max(1, int(query_chunk_size))
    for start in range(0, int(query_xyz.shape[1]), chunk):
        end = min(int(query_xyz.shape[1]), start + chunk)
        query = query_xyz[:, start:end]
        query_sq = torch.sum(query * query, dim=-1, keepdim=True)
        distance = query_sq + support_sq - 2.0 * torch.matmul(query, support_t)
        index = torch.topk(
            distance,
            k=effective_k,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
        if effective_k < requested_k:
            padding = index[..., :1].expand(
                *index.shape[:-1], requested_k - effective_k
            )
            index = torch.cat([index, padding], dim=-1)
        parts.append(index)
    return torch.cat(parts, dim=1)


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Batched farthest-point sampling used by the retained transition stages."""
    device = xyz.device
    batch, point_count, _ = xyz.shape
    sample_count = min(max(1, int(npoint)), point_count)
    centroids = torch.zeros(
        batch, sample_count, dtype=torch.long, device=device
    )
    distance = torch.full(
        (batch, point_count), 1e10, dtype=xyz.dtype, device=device
    )
    if random_start:
        farthest = torch.randint(
            0, point_count, (batch,), dtype=torch.long, device=device
        )
    else:
        farthest = torch.zeros(batch, dtype=torch.long, device=device)
    batch_index = torch.arange(batch, dtype=torch.long, device=device)
    for sample_index in range(sample_count):
        centroids[:, sample_index] = farthest
        centroid = xyz[batch_index, farthest].unsqueeze(1)
        squared = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, squared)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


class PointTransformerBlock(nn.Module):
    """Point Transformer vector-attention block over 16 spatial neighbors."""

    def __init__(
        self,
        channels: int,
        share_planes: int = 8,
        nsample: int = 16,
    ):
        super().__init__()
        if channels % share_planes != 0:
            raise ValueError("channels must be divisible by share_planes")
        self.channels = int(channels)
        self.share_planes = int(share_planes)
        self.nsample = int(nsample)
        grouped_channels = self.channels // self.share_planes
        self.linear_q = nn.Linear(self.channels, self.channels)
        self.linear_k = nn.Linear(self.channels, self.channels)
        self.linear_v = nn.Linear(self.channels, self.channels)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, self.channels),
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, grouped_channels),
            nn.BatchNorm1d(grouped_channels),
            nn.ReLU(inplace=True),
            nn.Linear(grouped_channels, grouped_channels),
        )

    def forward(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        neighbor_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, point_count, channels = features.shape
        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} feature channels, got {channels}"
            )
        if neighbor_index is None:
            neighbor_index = chunked_knn(
                xyz, xyz, self.nsample
            )
        neighbor_count = int(neighbor_index.shape[-1])
        query = self.linear_q(features).unsqueeze(2)
        key = index_points(self.linear_k(features), neighbor_index)
        value = index_points(self.linear_v(features), neighbor_index)
        neighbor_xyz = index_points(xyz, neighbor_index)
        relative_xyz = neighbor_xyz - xyz.unsqueeze(2)
        position = self.linear_p(relative_xyz.reshape(-1, 3)).reshape(
            batch, point_count, neighbor_count, self.channels
        )
        logits = self.linear_w(
            (query - key + position).reshape(-1, self.channels)
        ).reshape(
            batch,
            point_count,
            neighbor_count,
            self.channels // self.share_planes,
        )
        # Vector attention is normalized independently over the K neighbors.
        attention = torch.softmax(logits, dim=2)
        grouped_value = (value + position).reshape(
            batch,
            point_count,
            neighbor_count,
            self.share_planes,
            self.channels // self.share_planes,
        )
        attention = attention.unsqueeze(3)
        message = torch.sum(attention * grouped_value, dim=2).reshape(
            batch, point_count, self.channels
        )
        return features + message


class PointTransformerStage(nn.Module):
    """Stack blocks while reusing the exact same spatial kNN at one resolution."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        share_planes: int = 8,
        nsample: int = 16,
    ):
        super().__init__()
        self.nsample = int(nsample)
        self.blocks = nn.ModuleList(
            [
                PointTransformerBlock(
                    channels,
                    share_planes=share_planes,
                    nsample=nsample,
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(
        self, features: torch.Tensor, xyz: torch.Tensor
    ) -> torch.Tensor:
        neighbor_index = chunked_knn(xyz, xyz, self.nsample)
        for block in self.blocks:
            features = block(features, xyz, neighbor_index)
        return features


class TransitionDown(nn.Module):
    def __init__(
        self,
        npoint: int,
        in_channels: int,
        out_channels: int,
        nsample: int = 16,
    ):
        super().__init__()
        self.npoint = int(npoint)
        self.nsample = int(nsample)
        self.out_channels = int(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(int(in_channels) + 3, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, features: torch.Tensor, xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, point_count, channels = features.shape
        if point_count > self.npoint:
            sampled_index = farthest_point_sample(
                xyz,
                self.npoint,
                random_start=self.training,
            )
            new_xyz = index_points(xyz, sampled_index)
        else:
            new_xyz = xyz
        sample_count = int(new_xyz.shape[1])
        neighbor_index = chunked_knn(
            new_xyz, xyz, self.nsample
        )
        grouped_xyz = index_points(xyz, neighbor_index)
        grouped_features = index_points(features, neighbor_index)
        relative_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        grouped = torch.cat([grouped_features, relative_xyz], dim=-1)
        encoded = self.mlp(grouped.reshape(-1, channels + 3)).reshape(
            batch,
            sample_count,
            int(neighbor_index.shape[-1]),
            self.out_channels,
        )
        return torch.max(encoded, dim=2).values, new_xyz


class TransitionUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
    ):
        super().__init__()
        self.skip_projection = nn.Sequential(
            nn.Linear(int(skip_channels), int(out_channels)),
            nn.BatchNorm1d(int(out_channels)),
            nn.ReLU(inplace=True),
        )
        self.coarse_projection = nn.Sequential(
            nn.Linear(int(in_channels), int(out_channels)),
            nn.BatchNorm1d(int(out_channels)),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        coarse_features: torch.Tensor,
        coarse_xyz: torch.Tensor,
        skip_features: torch.Tensor,
        skip_xyz: torch.Tensor,
    ) -> torch.Tensor:
        batch, fine_count, _ = skip_xyz.shape
        neighbor_index = chunked_knn(skip_xyz, coarse_xyz, 3)
        neighbor_xyz = index_points(coarse_xyz, neighbor_index)
        squared_distance = torch.sum(
            (skip_xyz.unsqueeze(2) - neighbor_xyz) ** 2, dim=-1
        )
        reciprocal = 1.0 / (squared_distance + 1e-8)
        weight = reciprocal / reciprocal.sum(dim=2, keepdim=True)
        interpolated = torch.sum(
            index_points(coarse_features, neighbor_index)
            * weight.unsqueeze(-1),
            dim=2,
        )
        skip = self.skip_projection(
            skip_features.reshape(-1, skip_features.shape[-1])
        ).reshape(batch, fine_count, -1)
        coarse = self.coarse_projection(
            interpolated.reshape(-1, interpolated.shape[-1])
        ).reshape(batch, fine_count, -1)
        return skip + coarse


class PointTransformerSeg(nn.Module):
    """Point Transformer encoder-decoder for binary candidate segmentation."""

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 6,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.input_embedding = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        # Retained existing STS2R Point Transformer topology.
        self.tf1 = PointTransformerStage(32, 2)
        self.td1 = TransitionDown(1024, 32, 64)
        self.tf2 = PointTransformerStage(64, 3)
        self.td2 = TransitionDown(512, 64, 128)
        self.tf3 = PointTransformerStage(128, 4)
        self.td3 = TransitionDown(256, 128, 256)
        self.tf4 = PointTransformerStage(256, 3)
        self.td4 = TransitionDown(128, 256, 512)
        self.tf5 = PointTransformerStage(512, 2)

        self.tu1 = TransitionUp(512, 256, 256)
        self.tf6 = PointTransformerStage(256, 2)
        self.tu2 = TransitionUp(256, 128, 128)
        self.tf7 = PointTransformerStage(128, 2)
        self.tu3 = TransitionUp(128, 64, 64)
        self.tf8 = PointTransformerStage(64, 2)
        self.tu4 = TransitionUp(64, 32, 32)
        self.tf9 = PointTransformerStage(32, 2)
        self.output_head = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, int(num_classes), kernel_size=1, bias=False),
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        if inputs.ndim != 3:
            raise ValueError("PointTransformerSeg expects a 3D tensor")
        if inputs.shape[1] != self.input_channels:
            if inputs.shape[2] == self.input_channels:
                inputs = inputs.transpose(1, 2).contiguous()
            else:
                raise ValueError(
                    f"Expected {self.input_channels} XYZRGB channels, "
                    f"got {tuple(inputs.shape)}"
                )
        xyz0 = inputs[:, :3].transpose(1, 2).contiguous()
        features0 = self.input_embedding(inputs).transpose(1, 2).contiguous()
        features0 = self.tf1(features0, xyz0)

        features1, xyz1 = self.td1(features0, xyz0)
        features1 = self.tf2(features1, xyz1)
        features2, xyz2 = self.td2(features1, xyz1)
        features2 = self.tf3(features2, xyz2)
        features3, xyz3 = self.td3(features2, xyz2)
        features3 = self.tf4(features3, xyz3)
        features4, xyz4 = self.td4(features3, xyz3)
        features4 = self.tf5(features4, xyz4)

        up3 = self.tu1(features4, xyz4, features3, xyz3)
        up3 = self.tf6(up3, xyz3)
        up2 = self.tu2(up3, xyz3, features2, xyz2)
        up2 = self.tf7(up2, xyz2)
        up1 = self.tu3(up2, xyz2, features1, xyz1)
        up1 = self.tf8(up1, xyz1)
        up0 = self.tu4(up1, xyz1, features0, xyz0)
        up0 = self.tf9(up0, xyz0)
        logits = self.output_head(up0.transpose(1, 2).contiguous())
        return logits, None


def count_trainable_parameters(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
