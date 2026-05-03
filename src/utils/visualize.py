"""Point cloud visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

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
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    colors = PART_COLORS[labels % len(PART_COLORS)] if labels is not None else "gray"
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=2, alpha=0.8)
    ax.set_title(title)
    ax.set_axis_off()
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

    points, seg, _ = next(iter(loader))
    xyz = points.permute(0, 2, 1).to(device)

    with torch.no_grad():
        pred = model(xyz).argmax(dim=1)

    visualize_batch(points, pred.cpu(), seg, num_show=num_samples)
