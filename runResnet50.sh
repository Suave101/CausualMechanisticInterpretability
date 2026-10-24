#!/bin/bash
#SBATCH --job-name=resnet50_causal_sweep
#SBATCH --output=resnet50_sweep_%j.log
#SBATCH --error=resnet50_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# 1. Load Cluster Modules
module load datasets/cifar-10

# 2. Print Environment Verification
echo "=== Execution Node: $(hostname) ==="
echo "=== CIFAR10_ROOT: $CIFAR10_ROOT ==="
echo "=== AIP_DATASETS: $AIP_DATASETS ==="
nvidia-smi

# 3. Run the ResNet-50 10%-100% Sweep
python experiments/experiment_resnet50_sweep.py