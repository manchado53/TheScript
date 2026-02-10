#!/bin/bash

#SBATCH --job-name="Soccer Keypoint Model"
#SBATCH --output=logs/job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4

SCRIPT_NAME="Soccer Keypoint Model"
# CONTAINER="/data/containers/msoe-tensorflow.sif"
PYTHON_FILE="/home/manchadoa/TheScript/src/main.py"
conda activate soccer
## SCRIPT
echo "SBATCH SCRIPT: ${SCRIPT_NAME}"
srun hostname; pwd; date;
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=8 bash --login -c "conda activate soccer; python /home/manchadoa/TheScript/src/main.py"

echo "END: " $SCRIPT_NAME