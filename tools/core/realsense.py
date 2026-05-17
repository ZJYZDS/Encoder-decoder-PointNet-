"""RealSense 相机管理器：提供对齐的 depth/color 帧和相机内参。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """彩色相机内参（用于深度 → 3D 投影）。"""
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float  # raw depth → meters


class RealSenseManager:
    """RealSense 相机的生命周期管理。

    用法:
        with RealSenseManager(cfg) as cam:
            depth, color = cam.frames()  # 每次循环获取对齐帧
    """

    def __init__(self, depth_width: int, depth_height: int, depth_fps: int,
                 color_width: int, color_height: int, color_fps: int):
        self._depth_w = depth_width
        self._depth_h = depth_height
        self._depth_fps = depth_fps
        self._color_w = color_width
        self._color_h = color_height
        self._color_fps = color_fps

        self._pipeline: Optional[rs.pipeline] = None
        self._profile: Optional[rs.pipeline_profile] = None
        self._align: Optional[rs.align] = None
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._filters: list = []

    # ---- 资源管理 ----

    def __enter__(self) -> "RealSenseManager":
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self._depth_w, self._depth_h,
                             rs.format.z16, self._depth_fps)
        config.enable_stream(rs.stream.color, self._color_w, self._color_h,
                             rs.format.bgr8, self._color_fps)
        self._profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)

        # 读取彩色相机内参
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._intrinsics = CameraIntrinsics(
            fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
            depth_scale=depth_sensor.get_depth_scale(),
        )
        logger.info("相机已启动 %dx%d, 内参: fx=%.4f fy=%.4f cx=%.4f cy=%.4f, 深度缩放=%.6f",
                     self._color_w, self._color_h,
                     self._intrinsics.fx, self._intrinsics.fy,
                     self._intrinsics.cx, self._intrinsics.cy,
                     self._intrinsics.depth_scale)

        # 初始化深度滤波器（只需一次）
        self._init_filters()

        return self

    def __exit__(self, *args) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            logger.info("相机已关闭")

    # ---- 滤波器 ----

    def _init_filters(self) -> None:
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 1)

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 1)
        spatial.set_option(rs.option.filter_smooth_delta, 50)

        hole_filling = rs.hole_filling_filter()

        self._filters = [decimation, spatial, hole_filling]

    # ---- 获取帧 ----

    def frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """返回对齐的 (depth_image, color_image) 或 None。"""
        try:
            frames = self._pipeline.wait_for_frames()
            aligned = self._align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                return None

            # 滤波
            for f in self._filters:
                depth_frame = f.process(depth_frame)

            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            return depth, color

        except Exception as e:
            logger.warning("获取帧失败: %s", e)
            return None

    @property
    def intrinsics(self) -> CameraIntrinsics:
        assert self._intrinsics is not None, "相机未初始化"
        return self._intrinsics

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._color_w, self._color_h
