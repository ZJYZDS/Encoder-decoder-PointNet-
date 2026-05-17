#!/usr/bin/env python3
"""label-pcd — 低成本带标签点云采集工具

RealSense 1280x720 实时预览 + YOLO-seg 自动语义标注。
按 'a' 一键采集带标签点云 [X, Y, Z, R, G, B, label]，跳过手动标注。

用法:
    python main.py
    python main.py --conf 0.6 --save-dir /path/to/output
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.config import Config
from core.pointcloud import CoordinateGrid, depth_to_pointcloud, pointcloud_stats
from core.realsense import RealSenseManager
from core.segmenter import YOLOSegmenter

# ── 日志 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("label-pcd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="低成本带标签点云采集工具")
    parser.add_argument("--save-dir", default=None, help="输出目录（覆盖配置）")
    parser.add_argument("--conf", type=float, default=None, help="YOLO 置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="YOLO IoU 阈值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 加载配置
    cfg = Config.load()
    cfg.update_from_cli(**vars(args))
    device = cfg.resolve_device()
    logger.info("使用设备: %s", device)

    # 输出目录
    save_dir = Path(cfg.output.save_dir)
    npy_dir = save_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    # 初始化各模块
    segmenter = YOLOSegmenter(
        model_path=cfg.yolo.model_path,
        imgsz=cfg.yolo.imgsz,
        conf=cfg.yolo.conf,
        iou=cfg.yolo.iou,
        kfs_classes=cfg.yolo.kfs_classes,
        device=device,
    )

    with RealSenseManager(
        depth_width=cfg.realsense.depth_width,
        depth_height=cfg.realsense.depth_height,
        depth_fps=cfg.realsense.depth_fps,
        color_width=cfg.realsense.color_width,
        color_height=cfg.realsense.color_height,
        color_fps=cfg.realsense.color_fps,
    ) as cam:

        # 缓存坐标网格（分辨率固定，只需生成一次）
        w, h = cam.resolution
        grid = CoordinateGrid(w, h)
        ri = cam.intrinsics

        logger.info("实时预览已启动 — 按 'a' 采集  'q' 退出")
        capture_count = 0

        while True:
            frames = cam.frames()
            if frames is None:
                continue

            depth_image, color_image = frames
            cv2.imshow("label-pcd (a=capture, q=quit)", color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("a"):
                _capture(depth_image, color_image, segmenter, grid,
                         ri, npy_dir, cfg, capture_count)
                capture_count += 1

            elif key == ord("q") or key == 27:
                logger.info("用户退出")
                break

    cv2.destroyAllWindows()


def _capture(
    depth_image: np.ndarray,
    color_image: np.ndarray,
    segmenter: YOLOSegmenter,
    grid: CoordinateGrid,
    intrinsics,
    npy_dir: Path,
    cfg: Config,
    count: int,
) -> None:
    """单次采集：YOLO 推理 → label map → 点云投影 → 保存。"""
    logger.info("[%d] 处理中 ...", count + 1)

    # 1. YOLO-seg 推理 → label map
    results = segmenter.predict(color_image)
    h, w = color_image.shape[:2]
    label_map = segmenter.generate_label_map(results, h, w)

    kfs_pixels = int(np.count_nonzero(label_map))
    logger.info("  label map 生成完毕 (KFS 像素: %d / %d)",
                 kfs_pixels, label_map.size)

    # 2. 深度 → 带标签点云
    pcd = depth_to_pointcloud(
        depth_image, color_image, label_map,
        intrinsics, grid,
        depth_min=cfg.realsense.depth_min,
        depth_max=cfg.realsense.depth_max,
    )

    if pcd is None or len(pcd) == 0:
        logger.warning("  无有效点云，跳过")
        return

    # 3. 保存
    n_total, n_kfs, ratio = pointcloud_stats(pcd)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = npy_dir / f"labeled_{ts}.npy"
    np.save(str(path), pcd)

    logger.info("  [%d] -> %s", count + 1, path)
    logger.info("  点数: total=%d  KFS=%d (%.1f%%)  bg=%d",
                 n_total, n_kfs, ratio, n_total - n_kfs)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("程序异常退出: %s", e)
        sys.exit(1)
