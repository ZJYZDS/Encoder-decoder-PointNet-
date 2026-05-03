#!/usr/bin/env python3
"""Export trained model to TorchScript and ONNX."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config.default import CONFIG
from src.models import PointNetPlusPlus
import src.ops.pointnet2_ops as ops


NUM_POINTS = 1024


def _fps_deterministic(xyz, n_sample):
    """Deterministic FPS: fixed starting point (ONNX-friendly)."""
    B, N, _ = xyz.shape
    device = xyz.device
    first = torch.zeros(B, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)
    dists = torch.full((B, N), torch.inf, dtype=xyz.dtype, device=device)
    idx = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    idx[:, 0] = first
    for i in range(1, n_sample):
        last = xyz[batch_idx, idx[:, i - 1], :].unsqueeze(1)
        d = torch.cdist(last, xyz, p=2).squeeze(1)
        dists = torch.min(d, dists)
        idx[:, i] = torch.argmax(dists, dim=-1)
    return idx


def _patch_fps():
    orig = ops.fastest_point_sample
    ops.fastest_point_sample = _fps_deterministic
    return orig


def _unpatch_fps(orig):
    ops.fastest_point_sample = orig


def export_torchscript(model, device, save_dir):
    print("\n=== TorchScript ===")
    _patch_fps()
    try:
        ts_model = PointNetPlusPlus(num_classes=CONFIG.num_classes,
                                     n_points=NUM_POINTS)
        ts_model.load_state_dict(model.state_dict())
        ts_model = ts_model.to(device).eval()

        dummy = torch.rand(1, 3, NUM_POINTS).to(device)
        traced = torch.jit.trace(ts_model, dummy)

        path = os.path.join(save_dir, "model.ts")
        traced.save(path)
        print(f"  -> {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
        print(f"  verify: {traced(dummy).shape}")
    finally:
        _unpatch_fps(_patch_fps())


def export_onnx(model, device, save_dir):
    print("\n=== ONNX ===")
    orig = _patch_fps()
    try:
        onnx_model = PointNetPlusPlus(num_classes=CONFIG.num_classes,
                                       n_points=NUM_POINTS)
        onnx_model.load_state_dict(model.state_dict())
        onnx_model = onnx_model.to(device).eval()

        dummy = torch.rand(1, 3, NUM_POINTS).to(device)

        path = os.path.join(save_dir, "model.onnx")
        torch.onnx.export(
            onnx_model, dummy, path,
            input_names=["xyz"], output_names=["logits"],
            dynamic_axes={"xyz": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        print(f"  -> {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    finally:
        _unpatch_fps(orig)


def main():
    device = torch.device(CONFIG.device)
    print(f"Device: {device}")

    # Find checkpoint
    ckpt_dir = os.path.join(CONFIG.checkpoint_dir, "shapenet_part")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = "best_model.pth"

    model = PointNetPlusPlus(num_classes=CONFIG.num_classes,
                              n_points=NUM_POINTS)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded: {ckpt_path}")

    save_dir = os.path.join(CONFIG.checkpoint_dir, "shapenet_part")
    os.makedirs(save_dir, exist_ok=True)

    export_torchscript(model, device, save_dir)
    export_onnx(model, device, save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
