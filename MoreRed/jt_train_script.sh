#!/bin/bash
#SBATCH --job-name=JT_ET
#SBATCH --partition=gpu-9m
#SBATCH --constraint="80gb|40gb"
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/MDET_JT/train_JT_ET-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
cp /home/space/datasets-sqfs/QM9.sqfs /tmp/


# 2. Start GPU monitoring in background
(
while true; do
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,nounits -l 5
    sleep 5
done
) > logs/MDET_JT/gpu_usage_$SLURM_JOB_ID.log 2>&1 &


# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
#apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container.sif python -u src/scripts/mrdtrain experiment=my_vp_gauss_clean run.data_dir="/input-data/energy_U0"
apptainer run --nv -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container3.sif python -u src/scripts/mrdtrain experiment=md_et_backbone_JT run.data_dir="/input-data/energy_U0" +matmul_precision="medium"

#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)


