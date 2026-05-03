"""
PointNet++ segmentation model.

Encoder: 3× SA_MSG (down-sampling)
Decoder: 3× FP (up-sampling + skip connections)
Head:    per-point classification Conv1d
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..ops import (
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation,
)


class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes: int, n_points: int = 1024):
        super().__init__()

        # ── Encoder ──
        self.sa1 = PointNetSetAbstractionMsg(
            n_sample=n_points // 4, radius_list=[0.1, 0.2],
            k_list=[8, 16], in_channel=3, mlp_sz=[[16, 32], [32, 64]],
            group_all=False,
        )  # → 96-dim
        self.sa2 = PointNetSetAbstractionMsg(
            n_sample=n_points // 16, radius_list=[0.2, 0.4],
            k_list=[8, 16], in_channel=3 + 96, mlp_sz=[[32, 64], [64, 128]],
            group_all=False,
        )  # → 192-dim
        self.sa3 = PointNetSetAbstractionMsg(
            n_sample=n_points // 64, radius_list=[0.4, 0.8],
            k_list=[8, 16], in_channel=3 + 192, mlp_sz=[[64, 128], [128, 256]],
            group_all=False,
        )  # → 384-dim

        # ── Decoder ──
        self.fp3 = PointNetFeaturePropagation(
            in_channel=384 + 192, mlp_sz=[256, 128])
        self.fp2 = PointNetFeaturePropagation(
            in_channel=128 + 96, mlp_sz=[128, 64])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=64 + 3, mlp_sz=[64, 64])

        # ── Head ──
        self.head = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz: Tensor, extra_feat: Tensor = None) -> Tensor:
        B = xyz.shape[0]

        l1_xyz, l1_feat = self.sa1(xyz, extra_feat)     # [B,3,256] [B,96,256]
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)     # [B,3,64]  [B,192,64]
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)     # [B,3,16]  [B,384,16]

        l2_feat = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)  # [B,128,64]
        l1_feat = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_feat)  # [B,64,256]
        x_feat = self.fp1(xyz, l1_xyz, xyz, l1_feat)           # [B,64,1024]

        return self.head(x_feat)  # [B, num_classes, 1024]


if __name__ == "__main__":
    from src.ops import set_seed  # noqa: needed when run directly
    set_seed(42)
    model = PointNetPlusPlus(num_classes=4, n_points=1024).eval()
    out = model(torch.rand(2, 3, 1024))
    print(f"Output: {out.shape}  ← [2, 4, 1024]")
