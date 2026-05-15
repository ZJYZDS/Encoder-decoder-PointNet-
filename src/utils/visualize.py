"""Point cloud visualization utilities for part segmentation."""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob

PART_COLORS = np.array([
    [0.65, 0.95, 0.05], [0.35, 0.05, 0.35], [0.65, 0.35, 0.65],
    [0.95, 0.95, 0.65], [0.95, 0.65, 0.05], [0.35, 0.05, 0.05],
    [0.65, 0.05, 0.05], [0.65, 0.35, 0.95], [0.05, 0.05, 0.65],
    [0.65, 0.05, 0.35], [0.05, 0.35, 0.35], [0.65, 0.65, 0.35],
    [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35],
    [0.65, 0.65, 0.65], [0.95, 0.95, 0.05], [0.65, 0.35, 0.05],
    [0.35, 0.65, 0.05], [0.95, 0.65, 0.95], [0.95, 0.35, 0.65],
    [0.05, 0.65, 0.95], [0.65, 0.95, 0.65], [0.95, 0.35, 0.95],
    [0.05, 0.05, 0.95], [0.65, 0.05, 0.95], [0.65, 0.05, 0.65],
    [0.35, 0.35, 0.95], [0.95, 0.95, 0.95], [0.05, 0.05, 0.05],
    [0.05, 0.35, 0.95], [0.65, 0.95, 0.95], [0.95, 0.05, 0.05],
    [0.35, 0.95, 0.35], [0.05, 0.35, 0.05], [0.05, 0.65, 0.35],
    [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95],
    [0.05, 0.95, 0.35], [0.95, 0.35, 0.05], [0.65, 0.35, 0.35],
    [0.35, 0.95, 0.65], [0.35, 0.35, 0.65], [0.65, 0.95, 0.35],
    [0.05, 0.95, 0.65], [0.65, 0.65, 0.95], [0.35, 0.05, 0.95],
    [0.35, 0.65, 0.95], [0.35, 0.05, 0.65],
])


def visualize_pointcloud(points, labels=None, title="Point Cloud", ax=None):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    colors = PART_COLORS[labels % len(PART_COLORS)] if labels is not None else "gray"
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=2, alpha=0.8)
    ax.set_title(title)
    ax.set_axis_off()
    if fig is not None:
        plt.show()
    return ax


def visualize_comparison(points, pred_labels, gt_labels=None):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    visualize_pointcloud(points, pred_labels, title="Prediction", ax=ax1)
    if gt_labels is not None:
        ax2 = fig.add_subplot(122, projection="3d")
        visualize_pointcloud(points, gt_labels, title="Ground Truth", ax=ax2)
    plt.tight_layout()
    plt.show()


def visualize_batch(points, pred, gt=None, num_show=2):
    if pred.ndim == 3:
        pred = pred.argmax(dim=1)
    if points.ndim == 3 and points.shape[1] == 3:
        points = points.permute(0, 2, 1)

    to_np = lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    points_np, pred_np = to_np(points), to_np(pred)
    gt_np = to_np(gt) if gt is not None else None

    for i in range(min(num_show, len(points_np))):
        if gt_np is not None:
            visualize_comparison(points_np[i], pred_np[i], gt_np[i])
        else:
            visualize_pointcloud(points_np[i], pred_np[i], title=f"Sample {i}")


def visualize_seg_clear(points, pred_labels, gt_labels=None, cls_name="",
                        title="", save_path=None):
    """Intuitive segmentation visualization.

    - Prediction (left) vs Ground Truth (right), each part in a distinct tab20 color
    - Error overlay: correctly classified points in gray, errors in red
    - Per-part accuracy shown in title
    """
    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels) if gt_labels is not None else None

    # Determine unique parts present in this object
    all_ids = np.unique(gt_labels if gt_labels is not None else pred_labels)
    n_parts = len(all_ids)
    cmap = plt.get_cmap("tab20" if n_parts <= 20 else "tab20b")
    part_colors = cmap(np.linspace(0, 1, max(n_parts, 1)))[:, :3]  # (N, 3) RGB

    id_to_color = {pid: part_colors[i] for i, pid in enumerate(sorted(all_ids))}

    def _color_points(labels):
        return np.array([id_to_color.get(l, (0.5, 0.5, 0.5))
                         for l in labels], dtype=float)

    n_subplots = 3 if gt_labels is not None else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(n_subplots * 6, 5),
                             subplot_kw={"projection": "3d"})
    if n_subplots == 1:
        axes = [axes]

    # Left: Prediction
    colors_pred = _color_points(pred_labels)
    axes[0].scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=colors_pred, s=2, alpha=0.9)
    axes[0].set_title(f"Prediction{(' - ' + cls_name) if cls_name else ''}")
    axes[0].set_axis_off()

    if gt_labels is not None:
        # Middle: Ground Truth
        colors_gt = _color_points(gt_labels)
        axes[1].scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=colors_gt, s=2, alpha=0.9)
        axes[1].set_title(f"Ground Truth{(' - ' + cls_name) if cls_name else ''}")
        axes[1].set_axis_off()

        # Right: Error overlay (gray = correct, red = wrong)
        correct = (pred_labels == gt_labels)
        acc = correct.mean() * 100
        err_colors = np.where(correct[:, None], [0.6, 0.6, 0.6], [1, 0, 0])
        axes[2].scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=err_colors, s=2, alpha=0.9)
        axes[2].set_title(f"Errors  (Acc: {acc:.1f}%)")
        axes[2].set_axis_off()

    # Legend for parts
    all_ids_sorted = sorted(all_ids)
    legend_elements = []
    for pid in all_ids_sorted:
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(color=id_to_color[pid], label=f"Part {pid}"))
    # Place legend below the figure
    fig.legend(handles=legend_elements, loc="lower center",
               ncols=min(len(legend_elements), 10), fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def inference_and_show(model_path="checkpoints/best_model.pth", num_samples=4):
    """Load model, infer on test set, show results."""
    import glob
    from torch.utils.data import DataLoader
    from ..models import PointNetPlusPlus
    from ..data import ShapeNetPartDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointNetPlusPlus(num_classes=50, n_points=1024)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    h5_dir = os.path.join(os.path.dirname(__file__),
                          "..", "..", "data", "hdf5_data")
    test_files = sorted(glob.glob(os.path.join(h5_dir, "ply_data_test*.h5")))
    dataset = ShapeNetPartDataset(test_files, num_points=1024, normalize=True)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    points, seg, cls_ids = next(iter(loader))
    xyz = points.permute(0, 2, 1).to(device)

    with torch.no_grad():
        pred = model(xyz).argmax(dim=1)

    # Load category names
    cat_names = [
        line.split("\t")[0]
        for line in open(os.path.join(h5_dir, "all_object_categories.txt"))
    ]

    for i in range(num_samples):
        cls_name = cat_names[cls_ids[i]] if cls_ids[i] < len(cat_names) else ""
        visualize_seg_clear(
            points[i].numpy(),
            pred[i].cpu().numpy(),
            gt_labels=seg[i].numpy(),
            cls_name=cls_name,
            title=f"Sample {i}",
        )


if __name__ == "__main__":
    import h5py
    import random

    # ── Setup paths ──
    PROJ_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
    sys.path.insert(0, PROJ_ROOT)
    H5_DIR = os.path.join(PROJ_ROOT, "data", "hdf5_data")
    CKPT_DIR = os.path.join(PROJ_ROOT, "checkpoints", "shapenet_part")

    from src.data import ShapeNetPartDataset
    from src.models import PointNetPlusPlus

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──
    model_path = os.path.join(CKPT_DIR, "best_model.pth")
    model = PointNetPlusPlus(num_classes=50, n_points=1024)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"Model loaded from: {model_path}")

    # ── Load category names ──
    cat_names = [
        line.split("\t")[0]
        for line in open(os.path.join(H5_DIR, "all_object_categories.txt"))
    ]

    # ── Sample test data ──
    test_files = sorted(glob.glob(os.path.join(H5_DIR, "ply_data_test*.h5")))
    dataset = ShapeNetPartDataset(test_files, num_points=1024, normalize=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    points, seg, cls_ids = next(iter(loader))

    # Randomly pick 3 samples
    n_show = 3
    indices = random.sample(range(len(points)), min(n_show, len(points)))
    points, seg, cls_ids = points[indices], seg[indices], cls_ids[indices]
    print(f"Testing {n_show} random samples: {[cat_names[c.item()] for c in cls_ids]}")

    # ── Inference ──
    xyz = points.permute(0, 2, 1).to(device)
    with torch.no_grad():
        logits = model(xyz)
        pred = logits.argmax(dim=1)
    print("Inference done.")

    # ── Visualize ──
    for i in range(len(points)):
        cls_name = cat_names[cls_ids[i].item()]
        visualize_seg_clear(
            points[i].numpy(),
            pred[i].cpu().numpy(),
            gt_labels=seg[i].numpy(),
            cls_name=cls_name,
            title=f"Sample {i} — {cls_name}",
        )
