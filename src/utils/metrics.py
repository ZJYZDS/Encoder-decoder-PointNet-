"""Evaluation metrics for part segmentation."""

import numpy as np
import torch


def compute_iou_per_part(pred: np.ndarray, target: np.ndarray, num_classes: int = 50):
    """Per-part IoU over the whole dataset."""
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        ious.append(intersection / union if union > 0 else np.nan)
    return np.array(ious)


def evaluate(model, test_loader, device, num_classes=50):
    """Run evaluation, return accuracy and per-part IoU."""
    model.eval()
    total_correct = 0
    total_points = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for points, seg, _ in test_loader:
            xyz = points.permute(0, 2, 1).to(device)
            label = seg.to(device)

            logits = model(xyz)
            pred = logits.argmax(dim=1)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            total_correct += (pred == label).sum().item()
            total_points += label.numel()

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    accuracy = total_correct / total_points

    part_ious = compute_iou_per_part(all_preds, all_targets, num_classes)
    valid_mask = ~np.isnan(part_ious)
    miou_all = part_ious[valid_mask].mean() if valid_mask.any() else 0.0

    return {"accuracy": accuracy, "mIoU": miou_all, "part_ious": part_ious}


def print_metrics(metrics, num_classes=50):
    """Pretty-print evaluation results."""
    print("=" * 50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  mIoU:      {metrics['mIoU']:.4f}")
    print("=" * 50)

    part_ious = metrics["part_ious"]
    valid_idx = np.where(~np.isnan(part_ious))[0]
    if len(valid_idx) == 0:
        return

    best = sorted(valid_idx, key=lambda i: part_ious[i], reverse=True)[:10]
    worst = sorted(valid_idx, key=lambda i: part_ious[i])[:10]

    print(f"\nTop-10 best parts:")
    for idx in best:
        print(f"  Part {idx:2d}: {part_ious[idx]:.4f}")
    print(f"\nTop-10 worst parts:")
    for idx in worst:
        print(f"  Part {idx:2d}: {part_ious[idx]:.4f}")
