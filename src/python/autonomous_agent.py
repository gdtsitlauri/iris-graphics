import csv
import os
import time
from datetime import UTC, datetime
from typing import Dict, List

import torch


class PrismGLAgent:
    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"

    def __init__(self, obstacle_threshold: float = 0.5):
        self.obstacle_threshold = obstacle_threshold
        self.log: List[Dict[str, str]] = []

    def perceive_obstacle(self, depth_map: torch.Tensor) -> bool:
        return bool(torch.min(depth_map).item() < self.obstacle_threshold)

    def decide(self, depth_map: torch.Tensor) -> str:
        h, w = depth_map.shape
        min_depth = torch.min(depth_map).item()
        if min_depth < 0.2:
            return self.STOP
        if self.perceive_obstacle(depth_map):
            left = torch.mean(depth_map[:, : w // 2]).item()
            right = torch.mean(depth_map[:, w // 2 :]).item()
            return self.TURN_LEFT if left > right else self.TURN_RIGHT
        return self.FORWARD

    def act(self, action: str, timestamp: str) -> None:
        self.log.append({"timestamp": timestamp, "action": action})


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synthetic_depth_frame(step: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    corridor = 0.9 + 0.5 * (1 - torch.abs(xx))
    center = 0.5 * torch.sin(torch.tensor(step / 7.0, device=device))
    moving_obstacle = 0.8 * torch.exp(-((xx - center) ** 2 + (yy * 1.2) ** 2) * 10)
    depth = (corridor - moving_obstacle).clamp(0.1, 4.0)
    return depth.half() if device.type == "cuda" else depth.float()


def run_autonomous_loop(steps: int = 100) -> Dict[str, float]:
    device = _device()
    os.makedirs("results/slam", exist_ok=True)
    os.makedirs("results/latency", exist_ok=True)
    agent = PrismGLAgent(obstacle_threshold=0.5)

    sim_csv = "results/slam/agent_simulation.csv"
    bench_csv = "results/latency/agent_benchmarks.csv"
    h, w = 120, 160
    total_distance = 0.0
    obstacles_avoided = 0
    collisions = 0
    latencies_ms = []

    with open(sim_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "timestamp", "min_depth_m", "action"])
        for step in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            depth_map = _synthetic_depth_frame(step, h, w, device)
            action = agent.decide(depth_map)
            timestamp = datetime.now(UTC).isoformat()
            agent.act(action, timestamp)

            min_depth = float(torch.min(depth_map).item())
            if action == PrismGLAgent.FORWARD:
                total_distance += 0.1
            elif action in (PrismGLAgent.TURN_LEFT, PrismGLAgent.TURN_RIGHT):
                obstacles_avoided += 1
            elif action == PrismGLAgent.STOP and min_depth < 0.2:
                collisions += 1

            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            writer.writerow([step, timestamp, f"{min_depth:.3f}", action])

    mean_latency = float(sum(latencies_ms) / max(1, len(latencies_ms)))
    avoid_rate = float(obstacles_avoided / max(1, (obstacles_avoided + collisions)))
    with open(bench_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["steps", "mean_latency_ms", "obstacles_avoided", "collisions", "obstacle_avoid_rate", "total_distance_m"]
        )
        writer.writerow(
            [steps, f"{mean_latency:.4f}", obstacles_avoided, collisions, f"{avoid_rate:.4f}", f"{total_distance:.3f}"]
        )
    return {
        "steps": float(steps),
        "latency_ms": mean_latency,
        "obstacles_avoided": float(obstacles_avoided),
        "collisions": float(collisions),
        "obstacle_avoid_rate": avoid_rate,
        "total_distance_m": total_distance,
    }


if __name__ == "__main__":
    print(run_autonomous_loop(steps=100))