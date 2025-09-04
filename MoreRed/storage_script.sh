#!/bin/bash
#SBATCH --job-name=my_cpu_job
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/squashfs_test-%j.out


# 1. copy the squashed dataset to the nodes /tmp 
cp /home/space/datasets-sqfs/QM9.sqfs /tmp/

# 2. bind the squashed data# Check if the file exists
if [ -f /home/space/datasets-sqfs/QM7-X.sqfs ]; then
    echo "File exists."
else
    echo "File does not exist."
fi
#set to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ /opt/apps/pytorch-2.0.1-gpu.sif python storage_test.py
