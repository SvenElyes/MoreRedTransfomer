#!/bin/bash
#SBATCH --job-name=mdetJT_qm7
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6       # <-- set 6 CPUs per task
#SBATCH --exclude=head076,head024
#SBATCH --output=logs/LONG_TRAIN/mdetJT_morered-%j.out
#SBATCH --constraint="h100|80gb"


# 1. copy the squashed dataset to the nodes /tmp 
cp /home/svenelzes/MoreRedTransfomer/MoreRed/qm7x_svenelzes.sqfs /tmp/


# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
#apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container.sif python -u ../src/scripts/mrdtrain experiment=my_vp_gauss_clean run.data_dir="/input-data/energy_U0"
apptainer run --nv -B /tmp/qm7x_svenelzes.sqfs:/input-data:image-src=/ old_container3.sif python -u src/scripts/mrdtrain experiment=md_et_backbone_JT_qm7 run.data_dir="/input-data" +matmul_precision="medium" \
data.batch_size=128 data.num_workers=6 data.num_train=25680 data.num_val=7201 trainer.max_epochs=10000 \
data.split_file="/home/svenelzes/MoreRedTransfomer/MoreRed/qm7/new_split.npz" data.only_equilibrium=True \
data.datapath="/input-data/svenelzes_qm7x.db"

#small_batchszie because we augment it (3 copies per sample)
#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)


