#!/usr/bin/env python3
"""Train PointNet++ part segmentation on ShapeNetPart or custom KFS data."""

import os
import sys
import time
import glob

# Ensure project root is on PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.default import CONFIG, KFSConfig
from src.ops import set_seed
from src.models import PointNetPlusPlus
from src.data import ShapeNetPartDataset, KFSDataset


def get_dataset(cfg):
    """Return (train_dataset, val_dataset) for the chosen dataset."""
    if cfg.num_classes == 50:
        # ShapeNetPart
        h5_dir = os.path.join(cfg.root if hasattr(cfg, 'root') else '.',
                              cfg.data_dir)
        train_files = sorted(glob.glob(os.path.join(h5_dir, "ply_data_train*.h5")))
        val_files = sorted(glob.glob(os.path.join(h5_dir, "ply_data_val*.h5")))

        train_ds = ShapeNetPartDataset(train_files, num_points=cfg.num_points)
        val_ds = ShapeNetPartDataset(val_files, num_points=cfg.num_points)
    else:
        # Custom KFS dataset
        train_dir = os.path.join(cfg.data_dir, "train")
        val_dir = os.path.join(cfg.data_dir, "val")
        train_ds = KFSDataset(train_dir, num_points=cfg.num_points)
        val_ds = KFSDataset(val_dir, num_points=cfg.num_points)

    return train_ds, val_ds


def train(cfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # Data
    train_ds, val_ds = get_dataset(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  "
          f"Classes: {cfg.num_classes}  Points: {cfg.num_points}")

    # Model
    model = PointNetPlusPlus(num_classes=cfg.num_classes,
                              n_points=cfg.num_points).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Logging
    log_dir = os.path.join(cfg.log_dir, "train")
    writer = SummaryWriter(log_dir)

    checkpoint_dir = os.path.join(cfg.checkpoint_dir, "shapenet_part"
                                  if cfg.num_classes == 50 else "kfs")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()

        for points, seg, _ in train_loader:
            xyz = points.permute(0, 2, 1).to(device)
            label = seg.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xyz), label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, seg, _ in val_loader:
                xyz = points.permute(0, 2, 1).to(device)
                label = seg.to(device)
                val_loss += criterion(model(xyz), label).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), path)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Epoch {epoch:2d}/{cfg.epochs} | "
              f"train: {avg_loss:.4f} | val: {val_loss:.4f} | {elapsed:.1f}s")

    writer.close()
    print(f"\nDone! Best val_loss: {best_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfs", action="store_true", help="Train on KFS dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    cfg = KFSConfig() if args.kfs else CONFIG
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr

    train(cfg)
