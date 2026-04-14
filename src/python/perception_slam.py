import os
from typing import Dict, List

import numpy as np
import torch
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def depth_to_pointcloud(depth_map: torch.Tensor, k: Dict[str, float]) -> torch.Tensor:
    h, w = depth_map.shape
    device = depth_map.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth_map.float()
    x = (xx - k["cx"]) * z / k["fx"]
    y = (yy - k["cy"]) * z / k["fy"]
    pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    valid = torch.isfinite(pts).all(dim=-1) & (pts[:, 2] > 0)
    return pts[valid]


class VisualOdometry:
    def __init__(self, k: Dict[str, float], voxel_size: float = 0.1):
        self.k = k
        self.voxel_size = voxel_size
        self.current_pose = torch.eye(4, device=_device(), dtype=torch.float32)
        self.poses = [self.current_pose.clone()]
        self.prev_pc = None
        self.global_voxels = set()

    def _estimate_transform(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        # Lightweight ICP-like alignment through centroid translation.
        src_centroid = src.mean(dim=0)
        dst_centroid = dst.mean(dim=0)
        t = dst_centroid - src_centroid
        transform = torch.eye(4, device=src.device, dtype=torch.float32)
        transform[:3, 3] = t
        return transform

    def _voxelize(self, points_world: torch.Tensor) -> None:
        vox = torch.round(points_world / self.voxel_size).to(torch.int32).cpu().numpy()
        for p in vox:
            self.global_voxels.add((int(p[0]), int(p[1]), int(p[2])))

    def update(self, depth_map: torch.Tensor) -> torch.Tensor:
        pc = depth_to_pointcloud(depth_map, self.k).to(torch.float32)
        if pc.numel() == 0:
            self.poses.append(self.current_pose.clone())
            return self.current_pose

        if self.prev_pc is not None and self.prev_pc.numel() > 0:
            n = min(pc.shape[0], self.prev_pc.shape[0], 20000)
            src = self.prev_pc[:n]
            dst = pc[:n]
            delta = self._estimate_transform(src, dst)
            self.current_pose = self.current_pose @ delta

        self.prev_pc = pc
        self.poses.append(self.current_pose.clone())

        ones = torch.ones((pc.shape[0], 1), device=pc.device, dtype=torch.float32)
        hom = torch.cat([pc, ones], dim=-1)
        world = (self.current_pose @ hom.t()).t()[:, :3]
        self._voxelize(world)
        return self.current_pose

    def export_map(self, output_path: str) -> np.ndarray:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        voxels = np.array(list(self.global_voxels), dtype=np.int32)
        points = voxels.astype(np.float32) * self.voxel_size
        np.save(output_path, points)
        return points


def generate_synthetic_depth_frames(
    n_frames: int = 100,
    h: int = 120,
    w: int = 160,
    device: torch.device = _device(),
) -> List[torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    base = 1.5 + 0.3 * torch.sin(xx * 3.14) + 0.2 * torch.cos(yy * 3.14)
    frames = []
    for i in range(n_frames):
        shift = 0.005 * i
        noise = 0.01 * torch.randn_like(base)
        depth = (base + shift + noise).clamp(0.2, 4.0)
        frames.append(depth.half() if device.type == "cuda" else depth.float())
    return frames


def run_slam_pipeline(n_frames: int = 100) -> Dict[str, float]:
    device = _device()
    os.makedirs("results/slam", exist_ok=True)
    h, w = 120, 160
    k = {"fx": 120.0, "fy": 120.0, "cx": w / 2.0, "cy": h / 2.0}
    frames = generate_synthetic_depth_frames(n_frames=n_frames, h=h, w=w, device=device)
    vo = VisualOdometry(k=k, voxel_size=0.1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if start is not None:
        start.record()
    else:
        import time

        t0 = time.time()

    for depth in frames:
        vo.update(depth)

    if end is not None:
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        total_ms = (time.time() - t0) * 1000.0

    global_map = vo.export_map("results/slam/pointcloud_map.npy")
    poses = torch.stack(vo.poses).cpu().numpy()
    traj = poses[:, :3, 3]
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(traj[:, 0], traj[:, 2], linewidth=1.2)
        plt.title("IRIS VO Trajectory")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.tight_layout()
        plt.savefig("results/slam/trajectory.png", dpi=120)
        plt.close()

    return {
        "frames": float(n_frames),
        "latency_per_frame_ms": float(total_ms / n_frames),
        "trajectory_error_m": float(np.linalg.norm(traj[-1] - traj[0])),
        "map_points": float(global_map.shape[0]),
    }


if __name__ == "__main__":
    metrics = run_slam_pipeline(n_frames=100)
    print(metrics)