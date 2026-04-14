import torch
import time
import os
import csv
from perception_slam import IrisPerception

def run_autonomous_loop():
    device = torch.device("cuda")
    print("[IRIS VISION] Booting Autonomous Control System...")
    
    # Διασφάλιση ύπαρξης φακέλου αποτελεσμάτων
    os.makedirs('results/latency', exist_ok=True)
    csv_path = 'results/latency/benchmarks.csv'
    
    # Προετοιμασία CSV
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Latency_ms', 'Min_Depth_m', 'Action'])
    
    # Φόρτωση του "Οπτικού Φλοιού"
    perception = IrisPerception().to(device).half()
    
    print("[IRIS VISION] Connecting to vehicle telemetry... [ONLINE]")
    print(f"[IRIS VISION] Logging results to: {csv_path}")
    print("==========================================================")
    
    # Τρέχουμε 10 "ticks" αποφάσεων για πιο γεμάτο dataset
    for step in range(1, 11):
        start_time = time.time()
        
        # 1. Simulated FOV (Field of View)
        fov_features = torch.rand(10000, 64, device=device).half()
        
        # 2. Perception Inference
        with torch.no_grad():
            depth, semantics = perception(fov_features)
            
            # 3. Decision Logic
            min_depth = torch.min(depth).item()
            
            if min_depth < 0.2:
                action = "EMERGENCY EVADE"
                status = "DANGER"
            elif min_depth < 0.5:
                action = "ADJUST TRAJECTORY"
                status = "CAUTION"
            else:
                action = "MAINTAIN COURSE"
                status = "CLEAR"
                
        # Χρονισμός (Latency)
        torch.cuda.synchronize()
        latency = (time.time() - start_time) * 1000
        
        # Αποθήκευση στο CSV
        csv_writer.writerow([step, f"{latency:.2f}", f"{min_depth:.2f}", action])
        
        print(f"Tick {step:02d} | Vision Latency: {latency:.1f}ms | Closest Obstacle: {min_depth:.2f}m | Action: [{status}] {action}")
        
        # 60Hz rate limit simulation
        time.sleep(max(0, 0.016 - (latency/1000)))

    csv_file.close()
    print("==========================================================")
    print(f"[SUCCESS] Benchmarks saved to {csv_path}")

if __name__ == "__main__":
    run_autonomous_loop()