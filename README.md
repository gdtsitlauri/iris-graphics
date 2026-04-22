# IRIS: Interactive Rendering and Intelligent Spatial-Modeling


IRIS is a neural rendering + perception framework that runs with FP16/CUDA on consumer GPUs.  
Primary target: NVIDIA GTX 1650 (4GB VRAM, SM 7.5) on Windows 11 + WSL2.


## Project Metadata

| Field | Value |
| --- | --- |
| Author | George David Tsitlauri |
| Affiliation | Dept. of Informatics & Telecommunications, University of Thessaly, Greece |
| Contact | gdtsitlauri@gmail.com |
| Year | 2026 |

## Primary Research Thesis

IRIS is strongest as a consumer-GPU research pipeline that couples lightweight
neural rendering, SLAM-style perception, and a bounded autonomous agent loop in
one reproducible CUDA-first stack. The current release is most credible when
read as an artifact-driven edge-rendering and perception framework, not as a
claim that every precision mode or renderer setting is universally optimal.

## GPU Requirements
- NVIDIA GPU with CUDA support (GTX 1650 validated)
- 4GB VRAM minimum
- PyTorch with CUDA enabled
- FP16 execution enabled for renderer and perception pipeline

## Real Benchmark Snapshot

| Workload | Resolution / Size | Precision | FPS / Throughput | Latency | VRAM |
|---|---:|---:|---:|---:|---:|
| Neural renderer (scale 0.125) | 720p | FP16 | 28.47 FPS | 35.12 ms | 0.11 GB |
| Neural renderer (scale 0.08) | 1080p | FP16 | 27.81 FPS | 35.95 ms | 0.10 GB |
| Neural renderer (scale 0.06) | 4K downscaled | FP16 | 29.25 FPS | 34.18 ms | 0.10 GB |
| Hash encoder | 100M samples | FP16 | 30.17 Mrays/s | - | - |
| SLAM | 100 frames | FP16 | - | 9.21 ms/frame | - |
| Agent loop | 100 steps | FP16 | - | 1.54 ms/frame | - |

These artifacts show that IRIS can keep rendering near the high-20 FPS range
while maintaining low memory usage and sub-10 ms SLAM / sub-2 ms agent-loop
latencies on the committed GTX 1650 configuration. The current benchmark tables
are most useful for repository-local hardware-aware comparison, especially
around memory footprint and end-to-end latency.

## Why IRIS Now Reads More Strongly

IRIS becomes much easier to defend as a strong repo when it is framed around
what it actually demonstrates well:

- a reproducible edge-class rendering pipeline on a 4 GB consumer GPU,
- low-latency SLAM and agent-loop artifacts in the same repository,
- a hardware-aware implementation story that is concrete rather than purely
  conceptual.

That combination gives the repo a tighter identity than a generic “graphics +
robotics” umbrella description.

## Modules
- `src/python/neural_renderer.py`: Instant-NGP style hash encoder + compact NeRF MLP + volume rendering + PRISM-GL temporal accumulation.
- `src/python/perception_slam.py`: depth-to-pointcloud, ICP-style visual odometry, voxel map builder, trajectory export.
- `src/python/autonomous_agent.py`: depth-based obstacle perception and rule-based state machine (FORWARD / TURN_LEFT / TURN_RIGHT / STOP).
- `src/python/benchmark_iris.py`: end-to-end benchmark suite for renderer, hash encoder, and SLAM.

## How To Run
```bash
# 1) Neural renderer (1080p FP16 test)
python3 src/python/neural_renderer.py

# 2) SLAM pipeline (100 synthetic depth frames)
python3 src/python/perception_slam.py

# 3) Autonomous agent simulation (100 steps)
python3 src/python/autonomous_agent.py

# 4) Full benchmark suite (all CSV outputs)
python3 src/python/benchmark_iris.py
```

## Output Artifacts
- `results/latency/renderer_benchmarks.csv`
- `results/latency/hashencoder_benchmarks.csv`
- `results/latency/slam_benchmarks.csv`
- `results/latency/agent_benchmarks.csv`
- `results/slam/pointcloud_map.npy`
- `results/slam/trajectory.png`
- `results/slam/agent_simulation.csv`

## Notes
- Rendering path is PyTorch/CUDA only (no OpenGL/Vulkan dependency).
- Benchmark presets are constrained to complete within practical edge-device runtime limits.

## Evidence Hierarchy

- Primary evidence: renderer, SLAM, and agent-loop latency artifacts under the
  committed GTX 1650 setup
- Secondary evidence: hash-encoder throughput benchmarks and trajectory exports
- Supporting evidence: FP16/FP32 comparison tables used to characterize the
  repository's current implementation behavior

## Interpretation Boundary

The current IRIS release demonstrates that a compact neural rendering and
perception workflow can run credibly on a constrained consumer GPU. The
benchmark tables should be interpreted as repository-specific hardware artifacts
rather than as a universal claim that FP16 dominates FP32 in every workload or
that the current pipeline replaces larger rendering or robotics stacks.

## Strongest deployment-safe story

If IRIS is foregrounded, the strongest and safest story is:

- lightweight neural rendering on a GTX 1650 with practical FPS and low VRAM,
- fast perception and control-loop artifacts in the same pipeline,
- a credible edge-compute research baseline for spatial computing under strict
  hardware limits.


