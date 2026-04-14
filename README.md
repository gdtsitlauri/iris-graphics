# IRIS: Interactive Rendering & Intelligent Spatial-modeling
### Neural Perception & Rendering Framework for Edge Devices

**Author:** George David Tsitlauri  
**Hardware Target:** NVIDIA GTX 1650 (4GB VRAM)

IRIS bridges Computer Graphics and AI to enable high-fidelity spatial computing and autonomous agent vision on consumer-grade hardware. It utilizes the custom **PRISM-GL** optimizer to maintain real-time performance.

## Performance Metrics (Autonomous Vision Suite)
- **Architecture:** Hybrid Python (PyTorch) + C++ (Graphics Engine)
- **Precision:** FP16 (Half-Precision) optimizations
- **Inference Latency:** ~1.5 ms per frame (>500 FPS logic rate)
- **VRAM Footprint:** <40 MB during active SLAM simulation

## Framework Modules
1. **Neural Renderer:** Hash encoding NeRF engine.
2. **PRISM-GL Optimizer:** Temporal accumulation for low-VRAM GPUs.
3. **Perception & SLAM:** Semantic segmentation and depth estimation.
4. **Shader Lab:** C++ engine linking AI tensors to GL Textures.
5. **Autonomous Vision:** Real-time control loop for decision making.

## Usage
```bash
# Start Graphics Engine
make && ./src/cpp/iris_engine

# Start Autonomous Control Loop
python3 src/python/autonomous_agent.py