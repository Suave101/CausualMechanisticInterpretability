#!/bin/sh
# --- Slurm Job Configuration ---
#SBATCH --job-name=CMI_RN50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=experimentOutput.log
#SBATCH --partition=gpu2
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB

# 1. Load Cluster Modules
module load datasets/cifar-10

# 2. Print Environment Verification
echo "=== Execution Node: $(hostname) ==="
echo "=== CIFAR10_ROOT: $CIFAR10_ROOT ==="
echo "=== AIP_DATASETS: $AIP_DATASETS ==="
nvidia-smi

# 3. Run the ResNet-50 10%-100% Sweep with uv
uv run python experiments/experiment_resnet50_sweep.py