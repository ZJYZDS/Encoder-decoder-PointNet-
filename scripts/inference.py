#!/usr/bin/env python3
"""Run inference on a single point cloud and visualize results."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config.default import CONFIG
from src.models import PointNetPlusPlus
from src.utils.visualize import visualize_pointcloud, visualize_comparison


def load_pointcloud(path: str):
    """Load .npy or .pcd point cloud."""
    if path.endswith(".npy"):
        cloud = np.load(path)
        return cloud[:, :3]  # (N, 3)
    elif path.endswith(".pcd"):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points, dtype=np.float32)
        except ImportError:
            raise ImportError("Need open3d to read .pcd files")
    else:
        raise ValueError(f"Unsupported format: {path}")


def preprocess(points: np.ndarray, num_points: int):
    """Sample and normalize to match training."""
    N = points.shape[0]
    indices = np.random.choice(N, num_points, replace=(N < num_points))
    points = points[indices]

    centroid = points.mean(axis=0)
    points = points - centroid
    m = np.sqrt(np.sum(points ** 2, axis=1)).max()
    if m > 0:
        points = points / m

    return points, indices


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud", help="Path to point cloud (.npy or .pcd)")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path (default: checkpoints/...)")
    parser.add_argument("--num_classes", type=int, default=CONFIG.num_classes)
    parser.add_argument("--num_points", type=int, default=CONFIG.num_points)
    parser.add_argument("--show_gt", default=None,
                        help="Optional ground truth .npy for comparison")
    args = parser.parse_args()

    device = torch.device(CONFIG.device)

    # Find checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_dir = os.path.join(CONFIG.checkpoint_dir, "shapenet_part")
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = "best_model.pth"

    # Load model
    model = PointNetPlusPlus(num_classes=args.num_classes,
                              n_points=args.num_points)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"Loaded: {ckpt_path}")

    # Load and preprocess
    raw_points = load_pointcloud(args.cloud)
    points, orig_indices = preprocess(raw_points, args.num_points)
    print(f"Input: {raw_points.shape} → {points.shape}")

    # Inference
    tensor = torch.from_numpy(points).float().unsqueeze(0)  # [1, N, 3]
    tensor = tensor.permute(0, 2, 1).to(device)             # [1, 3, N]

    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Visualize
    gt_labels = None
    if args.show_gt:
        gt_labels = np.load(args.show_gt)
        gt_labels = gt_labels[orig_indices]
        visualize_comparison(points, pred, gt_labels)
    else:
        visualize_pointcloud(points, pred, title="Segmentation Result")


if __name__ == "__main__":
    main()
