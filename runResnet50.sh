#!/bin/sh
# --- Slurm Job Configuration ---
#SBATCH --job-name=CMI_RN50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=CMI_RN50_%A_%a.log
#SBATCH --partition=gpu2
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adoyle2025@my.fit.edu  # FIT TRACKS email notification
#
# Teacher seeds come from the array index: submit with --array=0-2 for three teachers (error bars).
# Each array task trains its own ResNet-50 teacher (skipped if the checkpoint exists) and then runs
# the zero-shot restructuring suite. The suite resumes from results/resnet50_seed<N>/suite.json,
# so resubmitting after a timeout continues where it stopped.

set -e

SEED=${SLURM_ARRAY_TASK_ID:-0}
if [ "$SEED" -eq 0 ]; then
    CKPT=checkpoints/resnet50_cifar10.pth
else
    CKPT=checkpoints/resnet50_cifar10_seed${SEED}.pth
fi

# 1. Load Cluster Modules
module load datasets/cifar-10

# 2. Print Environment Verification
echo "=== Execution Node: $(hostname) | Seed: $SEED ==="
echo "=== CIFAR10_ROOT: $CIFAR10_ROOT ==="
echo "=== AIP_DATASETS: $AIP_DATASETS ==="
nvidia-smi

# 3. Train the ResNet-50 teacher (SGD, cosine LR, 200 epochs) unless it already exists
if [ ! -f "$CKPT" ]; then
    uv run python experiments/train_teacher_cifar.py --model resnet50 --epochs 200 \
        --seed "$SEED" --num_workers "${SLURM_CPUS_PER_TASK:-8}" --checkpoint "$CKPT"
fi

# 4. Zero-shot restructuring suite: DeepAbstract, causal variants, selection controls
uv run python experiments/experiment_restructuring_suite.py --model resnet50 --seed "$SEED" --checkpoint "$CKPT"
