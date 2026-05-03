"""Point cloud transformation utilities (sampling, normalization)."""

import torch
import numpy as np
from torch import Tensor


def pc_normalize(pc: Tensor) -> Tensor:
    """Center to origin and scale to unit sphere."""
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


def random_sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """Randomly sample fixed number of points (with replacement if needed)."""
    N = points.shape[0]
    indices = np.random.choice(N, num_points, replace=(N < num_points))
    return points[indices]


def normalize_points_np(points: np.ndarray) -> np.ndarray:
    """NumPy version of pc_normalize."""
    centroid = points.mean(axis=0)
    points = points - centroid
    m = np.sqrt(np.sum(points ** 2, axis=1)).max()
    if m > 0:
        points = points / m
    return points
