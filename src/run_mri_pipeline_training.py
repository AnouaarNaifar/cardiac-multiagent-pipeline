"""
Runs the full YOLO26n then U-Net fine-tuning, sequentially (same single GPU,
no benefit to running them concurrently). Wrapper only - see
mri_yolo_train.py / mri_unet_train.py for the actual training logic.
"""
import subprocess
import sys
import time

if __name__ == "__main__":
    t0 = time.time()
    print("=== Starting YOLO26n training ===", flush=True)
    subprocess.run([sys.executable, "src/mri_yolo_train.py"], check=True)

    print("=== Starting U-Net training ===", flush=True)
    subprocess.run([sys.executable, "src/mri_unet_train.py"], check=True)

    print(f"=== FULL_PIPELINE_TRAINING_DONE total_time_sec={time.time() - t0:.1f} ===", flush=True)
