"""
PointNet++ classification model.

Encoder: 3× SA_MSG (down-sampling)
Global:  max pooling
Head:    MLP classifier
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..ops import (
    PointNetSetAbstractionMsg,
)


class PointNetPlusPlusCls(nn.Module):
    def __init__(self, num_classes: int, n_points: int = 1024):
        super().__init__()

        self.sa1 = PointNetSetAbstractionMsg(
            n_sample=n_points // 4, radius_list=[0.1, 0.2],
            k_list=[8, 16], in_channel=3, mlp_sz=[[16, 32], [32, 64]],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            n_sample=n_points // 16, radius_list=[0.2, 0.4],
            k_list=[8, 16], in_channel=3 + 96, mlp_sz=[[32, 64], [64, 128]],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            n_sample=n_points // 64, radius_list=[0.4, 0.8],
            k_list=[8, 16], in_channel=3 + 192, mlp_sz=[[64, 128], [128, 256]],
            group_all=False,
        )

        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz: Tensor, extra_feat: Tensor = None) -> Tensor:
        l1_xyz, l1_feat = self.sa1(xyz, extra_feat)     # [B,3,N/4]  [B,96,N/4]
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)     # [B,3,N/16] [B,192,N/16]
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)     # [B,3,N/64] [B,384,N/64]

        global_feat = torch.max(l3_feat, dim=2, keepdim=True)[0]
        global_feat = global_feat.view(-1, 384)
        return self.head(global_feat)


if __name__ == "__main__":
    model = PointNetPlusPlusCls(num_classes=6, n_points=1024)
    out = model(torch.rand(2, 3, 1024))
    print(out.shape)  # [2, 6]
