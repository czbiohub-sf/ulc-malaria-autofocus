#!/bin/bash


#SBATCH --job-name=ULCMalariaSSAFreruns
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-outputs/slurm-%j.out

env | grep "^SLURM" | sort

if [ $# -lt 1 ];
then
  echo "Missing argument, expected syntax: source rerun_ssaf_all_datasets.sh <scope folder>"
else
  eval "python3 ssaf_rerun_all.py $1"
fi
