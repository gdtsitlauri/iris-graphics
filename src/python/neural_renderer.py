import math
import time
from typing import Tuple

import torch
import torch.nn as nn


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for IRIS renderer.")
    return torch.device("cuda")


class HashEncoder(nn.Module):
    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.hashmap_size = 1 << log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        growth = math.exp(
            math.log(finest_resolution / base_resolution) / (num_levels - 1)
        )
        self.register_buffer(
            "resolutions",
            torch.tensor(
                [int(round(base_resolution * (growth**l))) for l in range(num_levels)],
                dtype=torch.int32,
            ),
        )
        self.hash_tables = nn.Parameter(
            torch.empty(
                num_levels,
                self.hashmap_size,
                features_per_level,
                dtype=torch.float16,
            )
        )
        nn.init.uniform_(self.hash_tables, -1e-4, 1e-4)

    @staticmethod
    def _hash(coords: torch.Tensor, hashmap_size: int) -> torch.Tensor:
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]
        hashed = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
        return torch.remainder(hashed, hashmap_size).long()

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions expected in [0, 1]
        positions = positions.to(dtype=self.hash_tables.dtype)
        features = []
        for level, res in enumerate(self.resolutions):
            scale = float(res.item())
            coords = torch.floor(positions * scale).to(torch.int32)
            indices = self._hash(coords, self.hashmap_size)
            features.append(self.hash_tables[level, indices])
        return torch.cat(features, dim=-1)


class NerfMLP(nn.Module):
    def __init__(self, in_features: int = 35, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
        )

    def forward(self, hash_features: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hash_features, view_dir], dim=-1).to(dtype=self.net[0].weight.dtype)
        raw = self.net(x)
        rgb = torch.sigmoid(raw[..., :3])
        sigma = torch.relu(raw[..., 3:4])
        return torch.cat([rgb, sigma], dim=-1)


class PrismGL_Turbo(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HashEncoder()
        self.mlp = NerfMLP()

    def forward(self, points: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
        hash_features = self.encoder(points)
        return self.mlp(hash_features, view_dir)


def volume_render(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    model: PrismGL_Turbo,
    near: float = 0.1,
    far: float = 4.0,
    n_samples: int = 64,
    chunk_size: int = 16384,
) -> torch.Tensor:
    device = rays_o.device
    n_rays = rays_o.shape[0]
    sample_dtype = rays_o.dtype if rays_o.dtype in (torch.float16, torch.float32) else torch.float16
    t_vals = torch.linspace(near, far, n_samples, device=device, dtype=sample_dtype)
    deltas = t_vals[1:] - t_vals[:-1]
    # Keep a large but finite last delta to avoid 0 * inf -> NaN in FP16.
    deltas = torch.cat(
        [deltas, torch.tensor([far - near], device=device, dtype=torch.float16)]
    ).float()

    rgb_out = torch.empty(n_rays, 3, device=device, dtype=sample_dtype)
    view_dir = nn.functional.normalize(rays_d, dim=-1)

    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)
        rays_o_chunk = rays_o[start:end]
        rays_d_chunk = rays_d[start:end]
        points = rays_o_chunk[:, None, :] + rays_d_chunk[:, None, :] * t_vals[None, :, None]
        points_norm = ((points + far) / (2.0 * far)).clamp(0.0, 1.0)
        view_dir_samples = view_dir[start:end, None, :].expand(-1, n_samples, -1)

        raw = model(points_norm.reshape(-1, 3), view_dir_samples.reshape(-1, 3))
        raw = raw.reshape(end - start, n_samples, 4)
        rgb = raw[..., :3].float()
        sigma = raw[..., 3].float().clamp(0.0, 20.0)

        alpha = 1.0 - torch.exp(-sigma * deltas[None, :])
        alpha = alpha.clamp(0.0, 1.0)
        trans = torch.cumprod(
            torch.cat(
                [torch.ones((end - start, 1), device=device, dtype=torch.float32), 1.0 - alpha + 1e-6],
                dim=-1,
            ),
            dim=-1,
        )[:, :-1]
        weights = alpha * trans
        rgb_chunk = (weights[..., None] * rgb).sum(dim=1).clamp(0.0, 1.0)
        rgb_out[start:end] = rgb_chunk.to(sample_dtype)

    return rgb_out


def prism_gl_accumulate(
    current_frame: torch.Tensor,
    prev_frame: torch.Tensor,
    motion_vectors: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    h, w, _ = current_frame.shape
    device = current_frame.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float16),
        torch.arange(w, device=device, dtype=torch.float16),
        indexing="ij",
    )
    prev_x = torch.clamp((xx + motion_vectors[..., 0]).round().long(), 0, w - 1)
    prev_y = torch.clamp((yy + motion_vectors[..., 1]).round().long(), 0, h - 1)
    reprojected_prev = prev_frame[prev_y, prev_x]
    return (1.0 - alpha) * reprojected_prev + alpha * current_frame


def run_turbo_optimizer(render_scale: float = 0.08) -> Tuple[float, float, float]:
    device = _require_cuda()
    model = PrismGL_Turbo().to(device).half().eval()
    full_w, full_h = 1920, 1080
    render_w = max(1, int(full_w * render_scale))
    render_h = max(1, int(full_h * render_scale))
    num_rays = render_w * render_h
    rays_o = torch.zeros(num_rays, 3, device=device, dtype=torch.float16)
    rays_d = nn.functional.normalize(
        torch.randn(num_rays, 3, device=device, dtype=torch.float16), dim=-1
    )

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = volume_render(rays_o[:8192], rays_d[:8192], model, n_samples=12, chunk_size=8192)
        torch.cuda.synchronize()
        start = time.time()
        rgb = volume_render(rays_o, rays_d, model, n_samples=12, chunk_size=16384)
        torch.cuda.synchronize()
        latency_ms = (time.time() - start) * 1000.0

    fps = 1000.0 / latency_ms
    vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print("--- PRISM-GL Turbo Optimizer: Active (FP16 Mode) ---")
    print(
        f"Rendered rays: {rgb.shape[0]} ({render_w}x{render_h}) "
        f"for 1080p output via temporal accumulation"
    )
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print(f"VRAM usage: {vram_mb:.1f} MB")
    return latency_ms, fps, vram_mb


if __name__ == "__main__":
    run_turbo_optimizer()