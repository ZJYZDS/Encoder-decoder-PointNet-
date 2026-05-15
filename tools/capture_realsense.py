"""
RealSense D435i 点云采集脚本

用法:
    python tools/capture_realsense.py

按键:
    SPACE    — 保存当前帧点云
    ESC/q    — 退出

保存格式 (适配 KFSDataset):
    data/captured/
        cloud_0000.npy       # (N, 6)  xyz + rgb (rgb 归一化到 0~1)
        cloud_0000_seg.npy   # (N,)    标签文件 (需要后期标注)
        cloud_0001.npy
        cloud_0001_seg.npy
        ...
"""

import os
import sys
import numpy as np
import cv2

# ── 如果没装 pyrealsense2，给提示 ──
try:
    import pyrealsense2 as rs
except ImportError:
    print("请先安装: pip install pyrealsense2")
    sys.exit(1)


SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "captured")
os.makedirs(SAVE_DIR, exist_ok=True)


def get_aligned_frames(pipeline, align):
    """获取对齐后的 RGB + Depth 帧."""
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None
    return depth_frame, color_frame


def depth_to_pointcloud(depth_frame, color_frame, intrinsics):
    """将深度图 + RGB 转为 (N, 6) 点云."""
    depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    color_img = np.asanyarray(color_frame.get_data())

    H, W = depth_img.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    # 生成像素网格
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth_img / 1000.0  # mm → m

    # 只保留有效深度
    mask = (z > 0.1) & (z < 10.0)
    z = z[mask]
    u, v = u[mask], v[mask]

    # 反投影到 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    # RGB
    rgb = color_img[v, u][:, [2, 1, 0]]  # BGR → RGB
    rgb = rgb.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    return np.concatenate([points, rgb], axis=1)


def main():
    # ── 初始化 RealSense ──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    # 设置自动曝光
    sensor = profile.get_device().query_sensor(rs.option.depth_units)
    # 关闭激光投影 (室内近距离用)
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    print("RealSense D435i 已启动")
    print("  SPACE → 保存当前帧    ESC/q → 退出")
    print(f"  保存到: {SAVE_DIR}")

    frame_count = 0
    try:
        while True:
            depth_frame, color_frame = get_aligned_frames(pipeline, align)
            if depth_frame is None:
                continue

            # 显示 RGB 画面
            color_img = np.asanyarray(color_frame.get_data())
            cv2.imshow("RealSense D435i - SPACE 保存, ESC 退出", color_img)
            key = cv2.waitKey(1)

            if key == 27 or key == ord("q"):  # ESC / q
                break
            elif key == ord(" "):  # SPACE
                pts = depth_to_pointcloud(depth_frame, color_frame, intrinsics)
                npy_path = os.path.join(SAVE_DIR, f"cloud_{frame_count:04d}.npy")
                seg_path = os.path.join(SAVE_DIR, f"cloud_{frame_count:04d}_seg.npy")

                np.save(npy_path, pts)
                # 占位标签，标注软件里手动修改
                np.save(seg_path, np.zeros(len(pts), dtype=np.int64))

                print(f"  [{frame_count:04d}] 保存点云: {pts.shape}, 范围 "
                      f"{pts[:, :3].min(axis=0).round(3)} ~ {pts[:, :3].max(axis=0).round(3)}")
                frame_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"采集结束, 共保存 {frame_count} 帧")


if __name__ == "__main__":
    main()
