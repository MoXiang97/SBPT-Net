"""PointMLP segmentation backbone adapted for six-channel SBPT-Net tokens.

Derived from ma-xu/pointMLP-pytorch under Apache-2.0. This adapted file accepts
XYZRGB token inputs and produces two per-token logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "rrelu":
        return nn.RReLU(inplace=True)
    if name == "selu":
        return nn.SELU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "hardswish":
        return nn.Hardswish(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    if name == "leakyrelu0.2":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    return nn.ReLU(inplace=True)


def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    device = points.device
    bsz = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(bsz, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    bsz, npts, _ = xyz.shape
    npoint = min(int(npoint), int(npts))
    centroids = torch.zeros(bsz, npoint, dtype=torch.long, device=device)
    distance = torch.full((bsz, npts), 1e10, device=device)
    farthest = torch.randint(0, npts, (bsz,), dtype=torch.long, device=device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(bsz, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    nsample = min(int(nsample), int(xyz.shape[1]))
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor"):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize is not None else None
        if self.normalize not in ["center", "anchor", None]:
            self.normalize = None

        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, channel + add_channel))
            self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, channel + add_channel))

    def forward(self, xyz, points):
        bsz, npts, _ = xyz.shape
        groups = min(int(self.groups), int(npts))
        fps_idx = farthest_point_sample(xyz.contiguous(), groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            else:
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(bsz, -1), dim=-1, keepdim=True)
            std = std.unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        anchor = new_points.view(bsz, groups, 1, -1).repeat(1, 1, grouped_points.shape[2], 1)
        new_points = torch.cat([grouped_points, anchor], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            get_activation(activation),
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation="relu"):
        super().__init__()
        hidden = int(channel * res_expansion)
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, hidden, kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(hidden),
            self.act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(hidden, channel, kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(channel, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(hidden, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1.0, bias=True, activation="relu", use_xyz=True):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.operation = nn.Sequential(
            *[
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
                for _ in range(blocks)
            ]
        )

    def forward(self, x):
        bsz, groups, kneighbors, dim = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, dim, kneighbors)
        x = self.transfer(x)
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(bsz, groups, -1)
        return x.permute(0, 2, 1)


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1.0, bias=True, activation="relu"):
        super().__init__()
        self.operation = nn.Sequential(
            *[
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
                for _ in range(blocks)
            ]
        )

    def forward(self, x):
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation="relu"):
        super().__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, bias=bias, activation=activation)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)

    def forward(self, xyz1, xyz2, points1, points2):
        points2_t = points2.permute(0, 2, 1)
        _, npts, _ = xyz1.shape
        _, sampled, _ = xyz2.shape

        if sampled == 1:
            interpolated_points = points2_t.repeat(1, npts, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            interp_k = min(3, sampled)
            dists, idx = dists[:, :, :interp_k], idx[:, :, :interp_k]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2_t, idx) * weight.view(xyz1.shape[0], npts, interp_k, 1), dim=2)

        if points1 is not None:
            points1_t = points1.permute(0, 2, 1)
            new_points = torch.cat([points1_t, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        return self.extraction(self.fuse(new_points))


class PointMLP(nn.Module):
    """PointMLP part-segmentation style network adapted for binary boundary segmentation.

    The architecture follows the official PointMLP segmentation design: hierarchical
    local grouping, pre/pos extraction blocks, feature propagation decoder, and global
    context. Task-specific changes are limited to accepting configurable input
    channels (centered XYZ and mean RGB in SBPT-Net), removing the ShapeNetPart
    object-class token, and outputting two per-point logits.
    """

    def __init__(
        self,
        num_classes=2,
        input_channels=14,
        points=8192,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=True,
        use_xyz=True,
        normalize="anchor",
        dim_expansion=(2, 2, 2, 2),
        pre_blocks=(2, 2, 2, 2),
        pos_blocks=(2, 2, 2, 2),
        k_neighbors=(32, 32, 32, 32),
        reducers=(4, 4, 4, 4),
        de_dims=(512, 256, 128, 128),
        de_blocks=(2, 2, 2, 2),
        gmp_dim=64,
    ):
        super().__init__()
        self.stages = len(pre_blocks)
        self.points = points
        self.embedding = ConvBNReLU1D(input_channels, embed_dim, bias=bias, activation=activation)

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion)
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = points
        en_dims = [last_channel]
        for i in range(self.stages):
            out_channel = last_channel * dim_expansion[i]
            anchor_points = max(1, anchor_points // reducers[i])
            self.local_grouper_list.append(LocalGrouper(last_channel, anchor_points, k_neighbors[i], use_xyz, normalize))
            self.pre_blocks_list.append(
                PreExtraction(
                    last_channel,
                    out_channel,
                    pre_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                    use_xyz=use_xyz,
                )
            )
            self.pos_blocks_list.append(
                PosExtraction(out_channel, pos_blocks[i], groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
            last_channel = out_channel
            en_dims.append(last_channel)

        self.decode_list = nn.ModuleList()
        en_dims = list(reversed(en_dims))
        de_dims = [en_dims[0]] + list(de_dims)
        assert len(en_dims) == len(de_dims) == len(de_blocks) + 1
        for i in range(len(en_dims) - 1):
            self.decode_list.append(
                PointNetFeaturePropagation(
                    de_dims[i] + en_dims[i + 1],
                    de_dims[i + 1],
                    blocks=de_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )

        self.gmp_map_list = nn.ModuleList([ConvBNReLU1D(dim, gmp_dim, bias=bias, activation=activation) for dim in en_dims])
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims), gmp_dim, bias=bias, activation=activation)
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim + de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias),
        )

    def forward(self, x):
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        feat = self.embedding(x)

        xyz_list = [xyz]
        feat_list = [feat]
        for i in range(self.stages):
            xyz, grouped_feat = self.local_grouper_list[i](xyz, feat.permute(0, 2, 1).contiguous())
            feat = self.pre_blocks_list[i](grouped_feat)
            feat = self.pos_blocks_list[i](feat)
            xyz_list.append(xyz)
            feat_list.append(feat)

        xyz_list.reverse()
        feat_list.reverse()
        feat = feat_list[0]
        for i, decoder in enumerate(self.decode_list):
            feat = decoder(xyz_list[i + 1], xyz_list[i], feat_list[i + 1], feat)

        gmp_list = [F.adaptive_max_pool1d(gmp(feat_i), 1) for gmp, feat_i in zip(self.gmp_map_list, feat_list)]
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1))
        feat = torch.cat([feat, global_context.repeat(1, 1, feat.shape[-1])], dim=1)
        return self.classifier(feat), None
