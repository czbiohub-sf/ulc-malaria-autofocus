#!/bin/bash

#SBATCH --job-name=ULCMalariaSSAFTraining
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --output=./slurm-outputs/slurm-%j.out

echo
echo "Single-Shot Auto Focus (SSAF) Training"
echo

# --gpus-per-node=A100:1

env | grep "^SLURM" | sort

if [[ $# -eq 0 ]]; then
  cmd="python3 train.py"
else
  cmd="$@"
fi

echo "command is"
echo $cmd

#echo
#echo "copying to /tmp/training_data..."
#echo

# curious about transfer time
#echo $(date '+%d/%m/%Y %H:%M:%S')
#mkdir -p /tmp/training_data
#tar -xf /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM\ Scope/ssaf_trainingdata/2022-06-10-1056/training_data.tar.gz -C /tmp
#echo $(date '+%d/%m/%Y %H:%M:%S')

#echo
#echo "transferred"
#echo

echo
echo "starting ssaf training..."
echo


nvidia-smi

wandb online
conda run $cmd
wandb offline

#echo
#echo "removing training data..."
#echo

#rm -rf /tmp/training_data/*
#rmdir /tmp/training_data
