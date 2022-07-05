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
echo "copying to /tmp/training_data..."
echo

# curious about transfer time
echo $(date '+%d/%m/%Y %H:%M:%S')
mkdir -p /tmp/training_data
tar -xf /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM\ Scope/ssaf_trainingdata/2022-06-10-1056/training_data.tar.gz -C /tmp
echo $(date '+%d/%m/%Y %H:%M:%S')

echo
echo "transferred"
echo

echo
echo "starting ssaf training..."
echo

nvidia-smi

conda run python3 train.py

echo
echo "removing training data..."
echo

rm -rf /tmp/training_data/*
rmdir /tmp/training_data
