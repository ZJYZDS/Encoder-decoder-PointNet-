#!/usr/bin/env python3
"""Prepare custom KFS dataset: PCD → npy, labeling, inspection."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse
import numpy as np
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    o3d = None


def pcd_to_npy(pcd_path: str, out_dir: str):
    """Convert a single PCD file to npy (N, 6) = xyz + rgb."""
    if o3d is None:
        raise ImportError("Need open3d: pip install open3d")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = (np.asarray(pcd.colors, dtype=np.float32) * 255).astype(np.uint8)
    cloud = np.concatenate([points, colors], axis=1)

    stem = Path(pcd_path).stem
    np.save(os.path.join(out_dir, f"{stem}.npy"), cloud)
    print(f"  {stem}: {cloud.shape}")


def label_dataset(data_dir: str):
    """Interactive labeling with Open3D. Ctrl+click to select KFS points."""
    if o3d is None:
        raise ImportError("Need open3d: pip install open3d")

    npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    npy_files = [f for f in npy_files if not f.endswith("_seg.npy")]

    for fpath in npy_files:
        cloud = np.load(fpath)
        points = cloud[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if cloud.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:6] / 255.0)

        print(f"\nLabel: {Path(fpath).name}  ({len(points)} pts)")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"Label - {Path(fpath).name}",
                          width=1280, height=720)
        vis.add_geometry(pcd)
        vis.run()
        picked = vis.get_picked_points()
        vis.destroy_window()

        labels = np.zeros(len(points), dtype=np.int64)
        labels[picked] = 1
        seg_path = fpath.replace(".npy", "_seg.npy")
        np.save(seg_path, labels)
        print(f"  Saved: {Path(seg_path).name}  (KFS: {len(picked)} pts)")


def inspect(data_dir: str):
    """Print dataset statistics."""
    npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    seg_files = [f for f in npy_files if f.endswith("_seg.npy")]
    cloud_files = [f for f in npy_files if not f.endswith("_seg.npy")]

    total_kfs = 0
    for sf in seg_files:
        total_kfs += np.load(sf).sum()

    print(f"\nDataset: {data_dir}")
    print(f"  Clouds:  {len(cloud_files)}")
    print(f"  Labeled: {len(seg_files)}")
    print(f"  Total KFS points: {int(total_kfs)}")


def main():
    parser = argparse.ArgumentParser(description="KFS dataset tools")
    parser.add_argument("--pcd_dir", help="Directory with raw .pcd files")
    parser.add_argument("--out_dir", default="data/processed",
                        help="Output directory for .npy files")
    parser.add_argument("--label", action="store_true",
                        help="Interactive labeling mode")
    parser.add_argument("--inspect", action="store_true",
                        help="Show dataset statistics")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.out_dir)
        return

    if args.pcd_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        pcd_files = sorted(glob.glob(os.path.join(args.pcd_dir, "*.pcd")))
        print(f"Converting {len(pcd_files)} PCD files...")
        for f in pcd_files:
            pcd_to_npy(f, args.out_dir)

    if args.label:
        label_dataset(args.out_dir)


if __name__ == "__main__":
    main()
