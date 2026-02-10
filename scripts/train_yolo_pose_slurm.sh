#!/bin/bash
#SBATCH --job-name="Sbatch Example"
#SBATCH --output=logs/job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=student@msoe.edu
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-gpu=4

echo "Checking GPU..."
nvidia-smi || echo "nvidia-smi not available"
/home/manchadoa/TheScript/soccer-venv/bin/python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"GPU count: {torch.cuda.device_count()}")' || echo "PyTorch not available"

# ---- 2. run Ultralytics YOLOv8-pose training ----
ultralytics pose train \
    data=/home/manchadoa/TheScript/data/keypoint-msoe-6/data.yaml \
    model=yolov8x-pose.pt \
    batch=16 \
    mosaic=0.0 \
    plots=True \
    epochs=100 \
    imgsz=640,640 \
    device=0                     # “0” binds to the single GPU we requested
