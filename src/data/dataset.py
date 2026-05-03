"""Dataset classes for ShapeNetPart and custom KFS data."""

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import pc_normalize, random_sample_points, normalize_points_np

# ──────────────────────────────────────────────
#  ShapeNetPart HDF5 dataset
# ──────────────────────────────────────────────

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class ShapeNetPartDataset(Dataset):
    """ShapeNetPart HDF5 dataset for part segmentation.

    File structure:
        file.h5 ─┬─ data:  [N, 2048, 3]
                 ├─ label: [N, 1]
                 └─ pid:   [N, 2048]
    """

    def __init__(self, h5_paths, num_points=1024, normalize=True):
        assert HAS_H5PY, "Need h5py: pip install h5py"

        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]

        points_list, labels_list, seg_list = [], [], []
        for path in h5_paths:
            f = h5py.File(path, "r")
            points_list.append(f["data"][:])
            labels_list.append(f["label"][:])
            seg_list.append(f["pid"][:])
            f.close()

        self.points = np.concatenate(points_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        self.seg = np.concatenate(seg_list, axis=0)
        self.num_points = num_points
        self.normalize = normalize

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        points = self.points[item].copy()
        seg = self.seg[item].copy()
        cls_label = int(self.labels[item])

        indices = np.random.choice(
            points.shape[0], self.num_points,
            replace=(points.shape[0] < self.num_points))
        points = points[indices]
        seg = seg[indices]

        if self.normalize:
            points = pc_normalize(torch.from_numpy(points).float()).numpy()

        return (
            torch.from_numpy(points).float(),
            torch.from_numpy(seg).long(),
            cls_label,
        )


# ──────────────────────────────────────────────
#  Custom KFS dataset (from .npy files)
# ──────────────────────────────────────────────

class KFSDataset(Dataset):
    """KFS segmentation dataset from labeled .npy files.

    Directory structure:
        data_dir/
            cloud_0000.npy       (N, 3|6)  xyz or xyz+rgb
            cloud_0000_seg.npy   (N,)      labels
            cloud_0001.npy
            cloud_0001_seg.npy
            ...
    """

    def __init__(self, data_dir, num_points=4096, use_rgb=False, normalize=True):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        # Exclude _seg.npy files
        self.files = [f for f in self.files if not f.endswith("_seg.npy")]
        self.num_points = num_points
        self.use_rgb = use_rgb
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cloud = np.load(self.files[idx]).astype(np.float32)

        if self.use_rgb and cloud.shape[1] >= 6:
            points = cloud[:, :3]
            colors = cloud[:, 3:6] / 255.0
        else:
            points = cloud[:, :3]
            colors = None

        seg_path = self.files[idx].replace(".npy", "_seg.npy")
        seg = np.load(seg_path).astype(np.int64)

        indices = np.random.choice(
            points.shape[0], self.num_points,
            replace=(points.shape[0] < self.num_points))
        points = points[indices]
        seg = seg[indices]

        if self.normalize:
            points = normalize_points_np(points)

        out = torch.from_numpy(points).float()
        if colors is not None:
            out = torch.cat([out, torch.from_numpy(colors[indices]).float()], dim=1)

        return out, torch.from_numpy(seg).long()

    @staticmethod
    def from_prepared(data_dir, num_points=4096):
        """Create dataset from prepared npy directory."""
        return KFSDataset(data_dir, num_points=num_points)


# ──────────────────────────────────────────────
#  Generic point cloud dataset (for testing)
# ──────────────────────────────────────────────

class PointCloudDataset(Dataset):
    """Generic point cloud dataset from .npy or .npz files."""

    def __init__(self, points_path=None, labels_path=None,
                 num_points=1024, normalize=True):
        super().__init__()
        self.num_points = num_points
        self.normalize = normalize

        if points_path is not None:
            data = np.load(points_path)
            self.points = data if isinstance(data, np.ndarray) else data["points"]
        else:
            self.points = np.random.rand(100, 2048, 3).astype(np.float32)

        if labels_path is not None:
            data = np.load(labels_path)
            self.labels = data if isinstance(data, np.ndarray) else data["labels"]
        else:
            self.labels = np.zeros(len(self.points), dtype=np.int64)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        points = self.points[item].copy()
        label = self.labels[item]

        points = random_sample_points(points, self.num_points)
        if self.normalize:
            points = pc_normalize(torch.from_numpy(points).float()).numpy()

        return (
            torch.from_numpy(points).float(),
            torch.tensor(label, dtype=torch.long),
        )
