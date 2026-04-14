import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src", "python")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from autonomous_agent import run_autonomous_loop
from neural_renderer import HashEncoder, NerfMLP, PrismGL_Turbo, prism_gl_accumulate, run_turbo_optimizer, volume_render
from perception_slam import VisualOdometry, depth_to_pointcloud, run_slam_pipeline


def _gpu_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for GPU test.")
    return torch.device("cuda")


def test_hash_encoder_output_shape():
    device = _gpu_or_skip()
    encoder = HashEncoder().to(device).half()
    x = torch.rand(4096, 3, device=device, dtype=torch.float16)
    y = encoder(x)
    assert y.shape == (4096, 32)


def test_nerf_mlp_forward():
    device = _gpu_or_skip()
    mlp = NerfMLP().to(device).half()
    f = torch.rand(2048, 32, device=device, dtype=torch.float16)
    d = torch.rand(2048, 3, device=device, dtype=torch.float16)
    out = mlp(f, d)
    assert out.shape == (2048, 4)


def test_volume_rendering():
    device = _gpu_or_skip()
    model = PrismGL_Turbo().to(device).half().eval()
    rays_o = torch.zeros(1024, 3, device=device, dtype=torch.float16)
    rays_d = torch.nn.functional.normalize(torch.randn(1024, 3, device=device, dtype=torch.float16), dim=-1)
    rgb = volume_render(rays_o, rays_d, model, n_samples=16)
    assert rgb.shape == (1024, 3)
    assert torch.isfinite(rgb).all()


def test_prism_gl_accumulation():
    device = _gpu_or_skip()
    current = torch.rand(64, 64, 3, device=device, dtype=torch.float16)
    prev = torch.rand(64, 64, 3, device=device, dtype=torch.float16)
    motion = torch.zeros(64, 64, 2, device=device, dtype=torch.float16)
    out = prism_gl_accumulate(current, prev, motion)
    assert out.shape == current.shape


def test_visual_odometry():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vo = VisualOdometry(k={"fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 60.0})
    depth = torch.ones(120, 160, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    pose = vo.update(depth)
    assert pose.shape == (4, 4)


def test_depth_to_pointcloud():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth = torch.ones(32, 32, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    points = depth_to_pointcloud(depth, {"fx": 50.0, "fy": 50.0, "cx": 16.0, "cy": 16.0})
    assert points.ndim == 2 and points.shape[1] == 3


def test_agent_simulation():
    metrics = run_autonomous_loop(steps=10)
    assert metrics["steps"] == 10.0


def test_renderer_benchmark_gpu():
    device = _gpu_or_skip()
    assert device.type == "cuda"
    _, fps, _ = run_turbo_optimizer()
    assert fps > 10.0


def test_fp16_memory():
    _gpu_or_skip()
    _, _, vram_mb = run_turbo_optimizer()
    assert vram_mb < 3500.0


def test_slam_pipeline_outputs():
    metrics = run_slam_pipeline(n_frames=10)
    assert metrics["frames"] == 10.0
    assert os.path.exists(os.path.join(ROOT, "results", "slam", "pointcloud_map.npy"))
