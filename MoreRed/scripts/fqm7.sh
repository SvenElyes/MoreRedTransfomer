#!/bin/bash
#SBATCH --job-name=fixqm7
#SBATCH --partition=cpu-2d
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/QM7/fixqm7%j.out

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run --nv -B /tmp:/input-data old_container3.sif \
    python -u notebooks/denoise.py
