"""Project configuration: all hyperparameters and paths in one place."""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class TrainConfig:
    # ── Model ──
    num_classes: int = 50       # ShapeNetPart: 50 parts
    num_points: int = 1024      # Points per sample

    # ── Training ──
    batch_size: int = 16
    epochs: int = 50
    lr: float = 0.001
    seed: int = 42
    device: str = field(default_factory=lambda:
                         "cuda" if torch.cuda.is_available() else "cpu")

    # ── Paths (relative to project root) ──
    data_dir: str = "data/hdf5_data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"

    # ── DataLoader ──
    num_workers: int = 4
    pin_memory: bool = True

    # ── Evaluation ──
    eval_batch_size: int = 16


@dataclass
class KFSConfig(TrainConfig):
    """KFS segmentation: 2 classes (bg + KFS), more points per sample."""
    num_classes: int = 2
    num_points: int = 4096
    epochs: int = 100


# Singleton config instance – import and use directly
CONFIG = TrainConfig()
