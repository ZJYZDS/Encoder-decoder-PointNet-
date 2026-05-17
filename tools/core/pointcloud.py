"""深度图 → 3D 点云：逐像素投影，带 label 融合。"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from core.realsense import CameraIntrinsics

logger = logging.getLogger(__name__)


class CoordinateGrid:
    """缓存像素坐标网格，避免每帧重复生成 meshgrid。

    相机分辨率固定后，xx/yy 网格只需创建一次。
    """

    def __init__(self, width: int, height: int):
        xx, yy = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        self.xx = xx  # (H, W)
        self.yy = yy  # (H, W)

    @property
    def size(self) -> Tuple[int, int]:
        return self.xx.shape  # (H, W)


def depth_to_pointcloud(
    depth_image: np.ndarray,
    color_image: np.ndarray,
    label_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    grid: CoordinateGrid,
    depth_min: float = 0.4,
    depth_max: float = 4.0,
) -> Optional[np.ndarray]:
    """将对齐的 depth/color + label_map 投影为带标签点云。

    Args:
        depth_image: (H, W) uint16, raw depth（需乘 depth_scale 得米）
        color_image: (H, W, 3) BGR uint8
        label_map:   (H, W) uint8, 0=背景 1=KFS
        intrinsics:  相机内参 (fx, fy, cx, cy, depth_scale)
        grid:        缓存的像素坐标网格
        depth_min:   最小有效深度 (m)
        depth_max:   最大有效深度 (m)

    Returns:
        (N, 7) float32 ndarray: [X, Y, Z, R, G, B, label] 或 None
    """
    z = depth_image.astype(np.float32) * intrinsics.depth_scale  # (H, W) m

    # 有效深度掩码
    valid = (z > depth_min) & (z < depth_max)
    if not valid.any():
        return None

    # 广播投影: X=(x-cx)*z/fx, Y=(y-cy)*z/fy
    X = (grid.xx - intrinsics.cx) * z / intrinsics.fx
    Y = (grid.yy - intrinsics.cy) * z / intrinsics.fy
    xyz = np.stack([X, Y, z], axis=-1)  # (H, W, 3)

    xyz = xyz[valid]
    rgb = color_image[valid][:, ::-1]  # BGR → RGB
    labels = label_map[valid]

    return np.column_stack([xyz, rgb.astype(np.float32), labels.astype(np.float32)])


def pointcloud_stats(pcd: np.ndarray) -> Tuple[int, int, float]:
    """点云统计：(总点数, KFS 点数, KFS 占比)"""
    n_total = len(pcd)
    n_kfs = int(np.count_nonzero(pcd[:, -1] > 0.5))
    ratio = n_kfs / n_total * 100 if n_total > 0 else 0.0
    return n_total, n_kfs, ratio
