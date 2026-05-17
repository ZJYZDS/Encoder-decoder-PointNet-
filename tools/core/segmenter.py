"""YOLO-seg 分割器：模型加载、推理、label map 生成。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOSegmenter:
    """YOLO-seg 模型封装。

    职责:
      1. 加载模型
      2. 在彩色图像上推理，获取 masks
      3. 将 masks 合成为像素级 label map
    """

    def __init__(self, model_path: str, imgsz: int = 640,
                 conf: float = 0.5, iou: float = 0.5,
                 kfs_classes: Optional[List[int]] = None,
                 device: str = "auto"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO 模型文件不存在: {model_path}")
        self._model = YOLO(str(path))
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._kfs_classes: Set[int] = set(kfs_classes or [0, 1])
        self._device = device

        logger.info(
            "YOLO 加载完成: %s, 类别=%s, KFS 映射=%s, device=%s",
            path.name, list(self._model.names.values()), self._kfs_classes, device,
        )

    def predict(self, image: np.ndarray) -> List:
        """对单帧 RGB/BGR 图像做分割推理，返回 YOLO Results 列表。"""
        return self._model.predict(
            source=image,
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
            device=self._device,
        )

    def generate_label_map(self, results: List, orig_h: int, orig_w: int) -> np.ndarray:
        """将 YOLO Results 转为像素级 label map (HxW, uint8: 0=背景, 1=KFS)。"""
        label_map = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for result in results:
            if result is None or result.masks is None or result.boxes is None:
                continue
            for i in range(len(result.masks.data)):
                cls_id = int(result.boxes.cls[i])
                if cls_id not in self._kfs_classes:
                    continue

                # mask 在模型输出分辨率 (如 160x160)，需缩放到原图
                mask = result.masks.data[i].cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                label_map[mask > 0] = 1

        return label_map

    def predict_label_map(self, image: np.ndarray) -> np.ndarray:
        """单步完成推理 + label map 生成。"""
        results = self.predict(image)
        h, w = image.shape[:2]
        return self.generate_label_map(results, h, w)
