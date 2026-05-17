"""配置管理：YAML 加载 + 类型安全的 dataclass + CLI 覆盖。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


@dataclass
class RealSenseConfig:
    depth_width: int = 1280
    depth_height: int = 720
    depth_fps: int = 15
    color_width: int = 1280
    color_height: int = 720
    color_fps: int = 15
    depth_min: float = 0.4
    depth_max: float = 4.0


@dataclass
class YOLOConfig:
    model_path: str = ""
    imgsz: int = 640
    conf: float = 0.5
    iou: float = 0.5
    kfs_classes: List[int] = field(default_factory=lambda: [0, 1])
    device: str = "auto"


@dataclass
class OutputConfig:
    save_dir: str = "output/labeled"


@dataclass
class Config:
    realsense: RealSenseConfig = field(default_factory=RealSenseConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        path = path or _CONFIG_PATH
        if not path.exists():
            logger.warning("配置文件 %s 不存在，使用默认配置", path)
            return cls()

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()

        if "realsense" in raw:
            for k, v in raw["realsense"].items():
                if hasattr(cfg.realsense, k):
                    setattr(cfg.realsense, k, v)

        if "yolo" in raw:
            for k, v in raw["yolo"].items():
                if hasattr(cfg.yolo, k):
                    setattr(cfg.yolo, k, v)

        if "output" in raw:
            for k, v in raw["output"].items():
                if hasattr(cfg.output, k):
                    setattr(cfg.output, k, v)

        return cfg

    def update_from_cli(self, **kwargs):
        """用 CLI 参数覆盖配置（仅支持 output.save_dir / yolo.conf / yolo.iou）。"""
        if "save_dir" in kwargs and kwargs["save_dir"] is not None:
            self.output.save_dir = kwargs["save_dir"]
        if "conf" in kwargs and kwargs["conf"] is not None:
            self.yolo.conf = kwargs["conf"]
        if "iou" in kwargs and kwargs["iou"] is not None:
            self.yolo.iou = kwargs["iou"]

    def resolve_device(self) -> str:
        if self.yolo.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.yolo.device
