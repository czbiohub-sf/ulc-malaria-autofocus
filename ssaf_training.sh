#!/bin/bash

#SBATCH --job-name=SSAFTraining
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=32
#SBATCH --output=./slurm-outputs/slurm-%j.out

echo
echo "Single-Shot Auto Focus (SSAF) Training"
echo

env | grep "^SLURM" | sort

echo
echo "starting ssaf training..."
echo

nvidia-smi

wandb enabled
wandb online
conda run "$@"
wandb offline


if [ $(squeue -u axel.jacobsen -t RUNNING | wc -l) -eq 2 ];
then
  rm -rf /tmp/training_data
fi
