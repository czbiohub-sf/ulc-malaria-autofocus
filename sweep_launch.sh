#!/bin/bash

#SBATCH --job-name=SSAF Sweep
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=./slurm-outputs/slurm-%j.out


env | grep "^SLURM" | sort

echo
echo "starting yogo training..."
echo

nvidia-smi

wandb online
conda run wandb agent --count 1 "$@"
wandb offline
