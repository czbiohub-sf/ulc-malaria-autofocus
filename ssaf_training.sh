#!/bin/bash

#SBATCH --job-name=ULCMalariaSSAFTraining
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1

echo
echo "Single-Shot Auto Focus (SSAF) Training"
echo

env | grep "^SLURM" | sort

echo
echo "starting ssaf training..."
echo

nvidia-smi

conda run python3 train.py
