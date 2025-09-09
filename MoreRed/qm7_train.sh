#!/bin/bash
#SBATCH --job-name=morered_train_qm7
#SBATCH --partition=gpu-9m
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/train_mdtrain_qm7-%j.out

# 1. copy qm7.db to  /tmp 
cp -r /home/bbdc2/quantum/max/datasets/qm7x/ /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run --nv -B /tmp:/input-data old_container3.sif \
    python -u src/scripts/mrdtrain experiment=md_et_backbone_qm7 run.data_dir="/input-data" +matmul_precision="medium"

#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)
