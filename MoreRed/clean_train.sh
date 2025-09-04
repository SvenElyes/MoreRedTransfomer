#!/bin/bash
#SBATCH --job-name=morered_train
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/train_mdtrain-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
cp /home/space/datasets-sqfs/QM9.sqfs /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container.sif python -u src/scripts/mrdtrain experiment=vp_gauss_ddpm_qm9 run.data_dir="/input-data/energy_U0"
#apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ morered_container.sif python -u src/scripts/mrdtrain experiment=md_et_backbone run.data_dir="/input-data/energy_U0"

#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)


