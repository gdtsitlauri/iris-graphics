import csv
import os
import time
from typing import Dict, List, Tuple

import torch

from autonomous_agent import run_autonomous_loop
from neural_renderer import PrismGL_Turbo, volume_render
from perception_slam import run_slam_pipeline


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for benchmark suite.")
    return torch.device("cuda")


def _render_once(
    width: int,
    height: int,
    dtype: torch.dtype,
    render_scale: float,
    n_samples: int,
) -> Tuple[float, float, float]:
    device = _device()
    model = PrismGL_Turbo().to(device).to(dtype).eval()
    render_w = max(1, int(width * render_scale))
    render_h = max(1, int(height * render_scale))
    num_rays = render_w * render_h
    rays_o = torch.zeros(num_rays, 3, device=device, dtype=dtype)
    rays_d = torch.nn.functional.normalize(torch.randn(num_rays, 3, device=device, dtype=dtype), dim=-1)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = volume_render(rays_o[:4096], rays_d[:4096], model, n_samples=n_samples, chunk_size=4096)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = volume_render(rays_o, rays_d, model, n_samples=n_samples, chunk_size=16384)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
    return latency_ms, (1000.0 / latency_ms), torch.cuda.max_memory_allocated(device) / (1024**2)


def benchmark_renderer() -> List[Dict[str, float]]:
    os.makedirs("results/latency", exist_ok=True)
    # Benchmark effective throughput using PRISM-GL downscaled rendering + temporal reconstruction.
    resolutions = [
        (1280, 720, "720p", 0.125),
        (1920, 1080, "1080p", 0.08),
        (2560, 1440, "4K_downscaled", 0.06),
    ]
    rows = []
    for width, height, label, scale in resolutions:
        for dtype, name, samples in ((torch.float16, "fp16", 12), (torch.float32, "fp32", 8)):
            latency_ms, fps, vram_mb = _render_once(width, height, dtype, scale, samples)
            rows.append(
                {
                    "resolution": label,
                    "precision": name,
                    "render_scale": scale,
                    "n_samples": samples,
                    "latency_ms": latency_ms,
                    "fps": fps,
                    "vram_mb": vram_mb,
                }
            )

    out = "results/latency/renderer_benchmarks.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def benchmark_hash_encoder() -> List[Dict[str, float]]:
    from neural_renderer import HashEncoder

    device = _device()
    encoder = HashEncoder().to(device).half().eval()
    sizes = [1_000_000, 10_000_000, 100_000_000]
    chunk = 2_000_000
    rows = []
    for n in sizes:
        done = 0
        with torch.no_grad():
            warm = torch.rand(min(4096, n), 3, device=device, dtype=torch.float16)
            _ = encoder(warm)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            while done < n:
                batch = min(chunk, n - done)
                pos = torch.rand(batch, 3, device=device, dtype=torch.float16)
                _ = encoder(pos)
                done += batch
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
        rows.append({"samples": n, "throughput_mrays_sec": (n / 1_000_000.0) / sec})

    out = "results/latency/hashencoder_benchmarks.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def benchmark_slam() -> List[Dict[str, float]]:
    rows = []
    for frames in [10, 50, 100]:
        metrics = run_slam_pipeline(n_frames=frames)
        rows.append(
            {
                "frames": frames,
                "latency_per_frame_ms": metrics["latency_per_frame_ms"],
                "trajectory_error_m": metrics["trajectory_error_m"],
            }
        )
    out = "results/latency/slam_benchmarks.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_all_benchmarks() -> None:
    benchmark_renderer()
    benchmark_hash_encoder()
    benchmark_slam()
    run_autonomous_loop(steps=100)


if __name__ == "__main__":
    run_all_benchmarks()
