#!/bin/bash
#SBATCH --job-name=morered_train_qm7
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/MDET/qm7_train_mdtrain-%j.out
#SBATCH --mem=32G
#SBATCH --constraint="h100|80gb"


# 1. copy the squashed dataset to the nodes /tmp 
cp /home/svenelzes/MoreRedTransfomer/MoreRed/qm7x_svenelzes.sqfs /tmp/


##SBATCH --constraint="h100|80gb"


# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
#apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container.sif python -u ../src/scripts/mrdtrain experiment=my_vp_gauss_clean run.data_dir="/input-data/energy_U0"
apptainer run --nv -B /tmp/qm7x_svenelzes.sqfs:/input-data:image-src=/ old_container3.sif python -u src/scripts/mrdtrain experiment=md_et_backbone_qm7 run.data_dir="/input-data" +matmul_precision="medium" \
data.batch_size=128 data.num_workers=4 data.num_train=25680 data.num_val=7201 trainer.max_epochs=10000 data.datapath="/input-data/svenelzes_qm7x.db" data.split_file="/home/svenelzes/MoreRedTransfomer/MoreRed/qm7/new_split.npz" data.only_equilibrium=True

#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)


