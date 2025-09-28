#!/bin/bash
#SBATCH --job-name=base_ddpm
#SBATCH --partition=gpu-2d
##SBATCH --constraint="h100|80gb"
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6       # <-- set 6 CPUs per task
#SBATCH --exclude=head076,head024
#SBATCH --output=logs/LONG_TRAIN/base_ddpm-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
cp /home/space/datasets-sqfs/QM9.sqfs /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
#apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container.sif python -u src/scripts/mrdtrain experiment=my_vp_gauss_clean run.data_dir="/input-data/energy_U0"
apptainer run --nv -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container3.sif python -u src/scripts/mrdtrain experiment=vp_gauss_ddpm_qm9 run.data_dir="/input-data/energy_U0" +matmul_precision="medium" \
data.batch_size=48 data.num_workers=6 data.num_train=55000 data.num_val=10000 data.num_test=10000 trainer.max_epochs=10000

#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)


