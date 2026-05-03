#!/usr/bin/env python3
"""Evaluate a trained model on the test set."""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from config.default import CONFIG
from src.models import PointNetPlusPlus
from src.data import ShapeNetPartDataset
from src.utils.metrics import evaluate, print_metrics


def main():
    device = torch.device(CONFIG.device)
    print(f"Device: {device}")

    # Data
    h5_dir = os.path.join(CONFIG.root if hasattr(CONFIG, 'root') else '.',
                          CONFIG.data_dir)
    test_files = sorted(glob.glob(os.path.join(h5_dir, "ply_data_test*.h5")))
    test_ds = ShapeNetPartDataset(test_files, num_points=CONFIG.num_points)
    test_loader = DataLoader(test_ds, batch_size=CONFIG.eval_batch_size)
    print(f"Test samples: {len(test_ds)}")

    # Model
    ckpt_dir = os.path.join(CONFIG.checkpoint_dir, "shapenet_part")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = "best_model.pth"  # fallback to old location

    model = PointNetPlusPlus(num_classes=CONFIG.num_classes,
                              n_points=CONFIG.num_points)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded: {ckpt_path}")

    # Evaluate
    metrics = evaluate(model, test_loader, device, CONFIG.num_classes)
    print_metrics(metrics, CONFIG.num_classes)


if __name__ == "__main__":
    main()
