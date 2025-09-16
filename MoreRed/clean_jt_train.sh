#!/bin/bash
#SBATCH --job-name=morered_train
#SBATCH --constraint="80gb|40gb"
#SBATCH --partition=gpu-9m
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/CLEAN_JT/clean_jt_train_mdtrain-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
cp /home/space/datasets-sqfs/QM9.sqfs /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container3.sif python -u src/scripts/mrdtrain experiment=vp_gauss_morered_jt_qm9 run.data_dir="/input-data/energy_U0"
#apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ morered_container.sif python -u src/scripts/mrdtrain experiment=md_et_backbone run.data_dir="/input-data/energy_U0"

#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)


