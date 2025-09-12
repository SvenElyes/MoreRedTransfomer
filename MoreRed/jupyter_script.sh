#!/bin/bash
#SBATCH --job-name=morered_jupyter
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/test-jupyter-with-data-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
cp /home/space/datasets-sqfs/QM9.sqfs /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer

#apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ morered_container.sif jupyter notebook --ip 0.0.0.0 --no-browser
apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ jupyter_container.sif jupyter notebook --ip 0.0.0.0 --no-browser
#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)