"""Tests for PointNet++ core operators."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from ..src.ops import (
    pc_normalize, fastest_point_sample, ball_query,
    idx2points, idx2points_3d, PointNetSetAbstractionMsg,
)


# ── Fixtures ──

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, N, C, K = 2, 64, 3, 8


def _rand_cloud():
    return torch.randint(-4, 4, (B, N, C), dtype=torch.float32, device=DEVICE)


# ── pc_normalize ──

class TestPCNormalize:
    def test_single_cloud(self):
        pc = torch.randint(-4, 4, (N, C), dtype=torch.float32, device=DEVICE)
        out = pc_normalize(pc)
        assert out.shape == (N, C)
        assert torch.sqrt(torch.sum(out ** 2, dim=1)).max() <= 1.0 + 1e-5

    def test_batch1_no_nan(self):
        pc = torch.randint(-4, 4, (1, N, C), dtype=torch.float32, device=DEVICE)
        out = pc_normalize(pc)
        assert out.shape == (1, N, C)
        assert not out.isnan().any()


# ── fastest_point_sample ──

class TestFPS:
    def test_output_shape(self):
        pc = _rand_cloud()
        idx = fastest_point_sample(pc, K)
        assert idx.shape == (B, K)
        assert idx.min() >= 0
        assert idx.max() < N

    def test_indices_unique(self):
        pc = _rand_cloud()
        idx = fastest_point_sample(pc, K)
        for b in range(B):
            assert len(set(idx[b].tolist())) == K


# ── ball_query ──

class TestBallQuery:
    def test_output_shape(self):
        pc = _rand_cloud()
        fps = fastest_point_sample(pc, 8)
        centroids = idx2points(pc, fps)
        idx = ball_query(pc, centroids, 2.0, K)
        assert idx.shape == (B, 8, K)

    def test_radius_zero(self):
        pc = _rand_cloud()
        fps = fastest_point_sample(pc, 8)
        centroids = idx2points(pc, fps)
        idx = ball_query(pc, centroids, 0.0, K)
        assert (idx < 0).all()


# ── idx2points ──

class TestIdx2Points:
    def test_sentinel_negative_one(self):
        pc = _rand_cloud()
        bad_idx = torch.full((B, K), -1, dtype=torch.long, device=DEVICE)
        out = idx2points(pc, bad_idx)
        assert (out == 0).all()


# ── idx2points_3d ──

class TestIdx2Points3D:
    def test_sentinel_negative_one(self):
        pc = _rand_cloud().permute(0, 2, 1)  # [B, 3, N]
        bad_idx = torch.full((B, 4, K), -1, dtype=torch.long, device=DEVICE)
        out = idx2points_3d(pc, bad_idx)
        assert (out == 0).all()


# ── PointNetSetAbstractionMsg ──

class TestSAMsg:
    def test_forward_xyz_only(self):
        sa = PointNetSetAbstractionMsg(
            n_sample=8, radius_list=[2.0, 4.0],
            k_list=[4, 8], in_channel=3,
            mlp_sz=[[8, 16], [16, 32]], group_all=False,
        ).to(DEVICE).eval()
        xyz = _rand_cloud().permute(0, 2, 1)
        with torch.no_grad():
            pt, feat = sa(xyz, None)
        assert pt.shape == (B, 3, 8)
        assert feat.shape == (B, 48, 8)

    def test_batch1_no_nan(self):
        sa = PointNetSetAbstractionMsg(
            n_sample=4, radius_list=[4.0], k_list=[4],
            in_channel=3, mlp_sz=[[8, 16]], group_all=False,
        ).to(DEVICE).eval()
        xyz = torch.rand(1, 3, 32, device=DEVICE)
        with torch.no_grad():
            pt, feat = sa(xyz, None)
        assert not feat.isnan().any()


# ── Run ──

if __name__ == "__main__":
    pytest = __import__("pytest")
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
