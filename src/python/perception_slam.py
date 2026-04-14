import torch
import torch.nn as nn
import time

class IrisPerception(nn.Module):
    def __init__(self, feature_dim=64, num_classes=5):
        super(IrisPerception, self).__init__()
        # 1. Κεφαλή Βάθους (Πόσο μακριά είναι το αντικείμενο;)
        self.depth_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid() # Κανονικοποίηση από 0 (κοντά) έως 1 (μακριά)
        )
        
        # 2. Κεφαλή Σημασιολογίας (Τι είναι αυτό το αντικείμενο;)
        self.semantic_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes) # Κατηγορίες: π.χ. Δρόμος, Κτίριο, Ουρανός, Πεζός, Όχημα
        )

    def forward(self, features):
        depth = self.depth_head(features)
        semantics = self.semantic_head(features)
        return depth, semantics

def test_perception():
    device = torch.device("cuda")
    # Τοποθέτηση στην GPU σε FP16
    perception_module = IrisPerception().to(device).half()
    
    # Στο SLAM δεν χρειαζόμαστε 1080p (2M rays). 
    # Ένα Point Cloud των 100.000 σημείων είναι υπεραρκετό για χαρτογράφηση!
    num_points = 100000
    simulated_features = torch.rand(num_points, 64, device=device).half()

    print("--- IRIS: Spatial Perception & SLAM Module ---")
    
    # Warm-up GPU
    for _ in range(5):
        _ = perception_module(simulated_features[:100])

    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        depth, semantics = perception_module(simulated_features)
        # Βρίσκουμε την επικρατέστερη κατηγορία για κάθε σημείο
        predicted_classes = torch.argmax(semantics, dim=1)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = (end_time - start_time) * 1000

    print(f"Processed Spatial Points: {num_points} (SLAM Resolution)")
    print(f"Depth Map Output: {depth.shape}")
    print(f"Semantic Map Output: {semantics.shape}")
    print(f"Inference Latency: {latency:.2f} ms")
    print(f"VRAM Usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

if __name__ == "__main__":
    test_perception()