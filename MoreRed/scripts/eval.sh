#!/bin/bash
#SBATCH --job-name=eval_qm7
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/eval/qm7-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
cp /home/svenelzes/MoreRedTransfomer/MoreRed/qm7x_svenelzes.sqfs /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
#apptainer run -B /tmp/QM9.sqfs:/input-data:image-src=/ old_container.sif python -u ../src/scripts/mrdtrain experiment=my_vp_gauss_clean run.data_dir="/input-data/energy_U0"
apptainer run --nv -B /tmp/qm7x_svenelzes.sqfs:/input-data:image-src=/ old_container3.sif python -u src/scripts/mrdeval experiment=eval_md run.data_dir="/input-data" +matmul_precision="medium" \
data.batch_size=6 data.num_workers=3 callbacks.sampling.log_rmsd=True  \
trainer.max_epochs=2 data.datapath="/input-data/svenelzes_qm7x.db" globals.checkpoint_path="/home/svenelzes/MoreRedTransfomer/MoreRed/runs/70cbcfb4-a106-11f0-8f11-e8ebd33abc78" 

#dont forget to adjust the ssh command to connect to the correct head and the port(adjust the port if needed)


