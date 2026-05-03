"""
PointNet++ core operators.

Mirrors the original pt2_utils.py (unchanged logic).
Referenced by both the segmentation and classification models.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def pc_normalize(pc: Tensor) -> Tensor:
    squeeze = False
    if pc.ndim == 3 and pc.shape[0] == 1:
        pc = pc[0]
        squeeze = True
    centroid = pc.mean(dim=0)
    pc = pc - centroid
    m = torch.sqrt(torch.sum(pc ** 2, dim=1)).max()
    pc = pc / m
    if squeeze:
        pc = pc.unsqueeze(0)
    return pc


def fastest_point_sample(xyz: Tensor, n_sample: int) -> Tensor:
    B, N, _ = xyz.shape
    device = xyz.device
    first_indices = torch.randint(0, N, (B,), device=device)
    batch_idx = torch.arange(B, device=device)
    distances = torch.full((B, N), torch.inf, dtype=xyz.dtype, device=device)
    sample_indices = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    sample_indices[:, 0] = first_indices
    for i in range(1, n_sample):
        last_idx = sample_indices[:, i - 1]
        last_points = xyz[batch_idx, last_idx, :].unsqueeze(1)
        dist = torch.cdist(last_points, xyz, p=2).squeeze(1)
        distances = torch.min(dist, distances)
        sample_indices[:, i] = torch.argmax(distances, dim=-1)
    return sample_indices


def ball_query(src: Tensor, centroids: Tensor, radius: float, k: int) -> Tensor:
    B, N, _ = src.shape
    _, M, _ = centroids.shape
    device = centroids.device
    dist = torch.sum(
        (centroids.unsqueeze(2) - src.unsqueeze(1)) ** 2, dim=-1,
    )
    R = radius ** 2
    dist[dist > R] = torch.inf
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
    valid_mask = dist.gather(dim=-1, index=idx) < torch.inf
    indices = torch.where(valid_mask, idx, torch.tensor(-1, device=device))
    return indices


def idx2points(points: Tensor, indices: Tensor) -> Tensor:
    B = points.shape[0]
    device = points.device
    batch_idx = torch.arange(B, device=device).view(B, *[1] * (indices.ndim - 1))
    valid = indices >= 0
    result = points[batch_idx, indices.clamp(min=0)]
    return torch.where(valid.unsqueeze(-1), result, torch.tensor(0.0, device=device))


def idx2points_3d(points: Tensor, idx: Tensor) -> Tensor:
    B = points.shape[0]
    device = points.device
    batch_idx = torch.arange(B, device=device).view(B, 1, 1)
    valid = idx >= 0
    selected = points[batch_idx, :, idx.clamp(min=0)]
    selected = selected.permute(0, 3, 1, 2)
    return torch.where(valid.unsqueeze(1), selected, torch.tensor(0.0, device=device))


def sample_and_group(
        n_sample: int, radius: float, k: int,
        xyz: Tensor, extra_feature: Tensor,
):
    B, N, d = xyz.shape
    fps_idx = fastest_point_sample(xyz, n_sample)
    fps_points = idx2points(xyz, fps_idx)
    ball_idx = ball_query(xyz, fps_points, radius, k)
    group_points = idx2points(xyz, ball_idx)
    group_points = group_points - fps_points.view(B, n_sample, 1, d)
    if extra_feature is not None:
        group_feature = idx2points(extra_feature, ball_idx)
        group_points = torch.cat([group_points, group_feature], dim=-1)
    return fps_points, group_points


def No_DownSample_group(xyz: Tensor, extra_feature: Tensor):
    B, N, d = xyz.shape
    new_xyz = torch.zeros(B, 1, d, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, d)
    if extra_feature is not None:
        group_points = torch.cat([grouped_xyz, extra_feature.view(B, 1, N, -1)], dim=-1)
    else:
        group_points = grouped_xyz
    return new_xyz, group_points


class PointAttention(nn.Module):
    def __init__(self, in_channel: int, hidden_dim: int = 64):
        super().__init__()
        self.query = nn.Conv2d(in_channel, hidden_dim, 1)
        self.key = nn.Conv2d(in_channel, hidden_dim, 1)
        self.value = nn.Conv2d(in_channel, in_channel, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        B, D, K, s = x.shape
        q = self.query(x).permute(0, 3, 2, 1)
        k = self.key(x).permute(0, 3, 2, 1)
        v = self.value(x).permute(0, 3, 2, 1)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weight = self.softmax(attn_score)
        out = torch.matmul(attn_weight, v)
        out = out.permute(0, 3, 2, 1).sum(dim=2, keepdim=True)
        return out


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(
            self, n_sample: int, radius_list: list, k_list: list,
            in_channel: int, mlp_sz: list, group_all: bool,
    ):
        super().__init__()
        self.n_sample = n_sample
        self.radius_list = radius_list
        self.k_list = k_list
        self.group_all = group_all
        self.mlp_sz = mlp_sz
        self.mlp_set = nn.ModuleList()
        self.bns_set = nn.ModuleList()
        self.attention_set = nn.ModuleList()
        for i in range(len(mlp_sz)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_sz[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.mlp_set.append(convs)
            self.bns_set.append(bns)
            self.attention_set.append(PointAttention(last_channel))

    def forward(self, xyz: Tensor, extra_feature: Tensor):
        xyz = xyz.permute(0, 2, 1)
        B, N, _ = xyz.shape
        s = self.n_sample
        fps_points = idx2points(xyz, fastest_point_sample(xyz, s))
        multi_scale_groups = []
        for i, radius in enumerate(self.radius_list):
            K = self.k_list[i]
            idx = ball_query(xyz, fps_points, radius, K)
            group_xyz = idx2points_3d(xyz.permute(0, 2, 1), idx)
            group_xyz -= fps_points.view(B, 3, s, 1)
            if extra_feature is not None:
                group_feature = idx2points_3d(extra_feature, idx)
                group_points = torch.cat([group_xyz, group_feature], dim=1)
            else:
                group_points = group_xyz
            group_points = group_points.permute(0, 1, 3, 2)
            current_mlp = self.mlp_set[i]
            current_bns = self.bns_set[i]
            for j in range(len(current_mlp)):
                group_points = F.relu(current_bns[j](current_mlp[j](group_points)))
            attn_out = self.attention_set[i](group_points)
            multi_scale_groups.append(attn_out.squeeze(2))
        msg_feature = torch.cat(multi_scale_groups, dim=1)
        msg_points = fps_points.permute(0, 2, 1)
        return msg_points, msg_feature


def square_distances(src: Tensor, cen: Tensor) -> Tensor:
    return torch.cdist(src, cen, p=2) ** 2


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp_sz: list):
        super().__init__()
        self.mlp = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channel = in_channel
        for ch in mlp_sz:
            self.mlp.append(nn.Conv1d(last_channel, ch, 1))
            self.bns.append(nn.BatchNorm1d(ch))
            last_channel = ch

    def forward(
            self, src_xyz: Tensor, cen_xyz: Tensor,
            src_feature: Tensor, cen_feature: Tensor,
    ):
        B, _, N = src_xyz.shape
        _, _, S = cen_xyz.shape
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            src_xyz_p = src_xyz.permute(0, 2, 1)
            cen_xyz_p = cen_xyz.permute(0, 2, 1)
            cen_feature = cen_feature.float()
            if S == 1:
                interpolated_points = cen_feature.repeat(1, 1, N)
            else:
                dists = square_distances(src_xyz_p, cen_xyz_p)
                dists, idx = dists.sort(dim=-1)
                dists, idx = dists[:, :, :3], idx[:, :, :3]
                dist_recip = 1.0 / (dists + 1e-8)
                norm = dist_recip.sum(dim=2, keepdim=True)
                weight = dist_recip / norm
                neighbor_feats = idx2points_3d(cen_feature, idx)
                interpolated_points = (neighbor_feats * weight.unsqueeze(1)).sum(dim=-1)
        interpolated_points = interpolated_points.to(dtype=cen_feature.dtype)
        if src_feature is not None:
            upsampled = torch.cat([src_feature, interpolated_points], dim=1)
        else:
            upsampled = interpolated_points
        out = upsampled
        for i in range(len(self.mlp)):
            out = F.relu(self.bns[i](self.mlp[i](out)))
        return out
