#!/bin/bash

#SBATCH --job-name=ULCMalariaSSAFTraining
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=./slurm-outputs/slurm-%j.out

echo
echo "Single-Shot Auto Focus (SSAF) Training"
echo

env | grep "^SLURM" | sort

echo
echo "starting ssaf training..."
echo

nvidia-smi

wandb online
conda run "$@"
wandb offline

rm -rf /tmp/training_data
