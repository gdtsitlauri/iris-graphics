import torch
import torch.nn as nn
import time

class PrismGL_Turbo(nn.Module):
    def __init__(self):
        super(PrismGL_Turbo, self).__init__()
        # Χρησιμοποιούμε μικρότερα αλλά ταχύτερα layers
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True), # inplace για εξοικονόμηση μνήμης
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

def run_turbo_optimizer():
    device = torch.device("cuda")
    # Μετατρέπουμε το μοντέλο σε Half Precision (FP16)
    model = PrismGL_Turbo().to(device).half() 
    
    num_rays = 1920 * 1080
    # Τα δεδομένα εισόδου επίσης σε FP16
    rays = torch.rand(num_rays, 3, device=device).half()

    print(f"--- PRISM-GL Turbo Optimizer: Active (FP16 Mode) ---")
    
    # Warm-up (πάντα χρειάζεται στην GPU για να "ξυπνήσει")
    for _ in range(5):
        _ = model(rays[:1000])

    torch.cuda.synchronize() # Συγχρονισμός για ακριβή μέτρηση
    start_time = time.time()
    
    with torch.no_grad():
        # Επεξεργασία
        output = model(rays)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = (end_time - start_time) * 1000
    fps = 1000 / latency

    print(f"Frame Processed: 1080p | Mode: FP16")
    print(f"Latency: {latency:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

if __name__ == "__main__":
    run_turbo_optimizer()