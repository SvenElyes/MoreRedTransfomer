import os
os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6


import sys, os
repo_path = "src"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from matplotlib import pyplot as plt

import torch
import numpy as np
from omegaconf import OmegaConf
from ase.visualize.plot import plot_atoms
from ase import Atoms
from schnetpack import properties
import schnetpack.transform as trn
from schnetpack.datasets import QM9
from tqdm import tqdm
import ase

from morered.datasets import QM9Filtered, QM7X
from morered.noise_schedules import PolynomialSchedule
from morered.processes import VPGaussianDDPM
from morered.utils import scatter_mean, check_validity, generate_bonds_data, batch_center_systems
from morered.sampling import DDPM, MoreRedJT, MoreRedAS, MoreRedITP


print("Hello")
# path to store the dataset as ASE '.db' files
split_file_path = "split.npz"

# model path
models_path = "models"

# MoreRed-JT: joint denoiser and time predictor
morered_jt = torch.load(os.path.join(models_path, "qm9_morered_jt.pt"), map_location="cpu")

# MoreRed-AS, MoreRed-ITP and plain DDPM use the same denoier model
ddpm_denoiser = torch.load(os.path.join(models_path, "qm9_ddpm.pt"), map_location="cpu")

# Seperate model for time prediction for MoreRed-AS and MoreRed-ITP
time_predictor = torch.load(os.path.join(models_path, "qm9_time_predictor.pt"), map_location="cpu")

# define the noise schedule
noise_schedule = PolynomialSchedule(T=1000, s=1e-5, dtype=torch.float64, variance_type="lower_bound")

# define the forward diffusion process
diff_proc = VPGaussianDDPM(noise_schedule, noise_key="eps", invariant=True, dtype=torch.float64)


ddpm_sampler = DDPM(
    diff_proc,
    ddpm_denoiser,
    time_key="t",
    noise_pred_key="eps_pred",
    cutoff=5.,
    recompute_neighbors=False,
    save_progress= True,
    progress_stride = 1,
    results_on_cpu = True,
)

mrd_jt_sampler = MoreRedJT(
    diff_proc,
    morered_jt,
    noise_pred_key="eps_pred",
    time_key="t",
    time_pred_key="t_pred",
    convergence_step=0,
    cutoff=5.,
    recompute_neighbors=False,
    save_progress= True,
    progress_stride = 1,
    results_on_cpu = True
)

mrd_as_sampler = MoreRedAS(
    diff_proc,
    ddpm_denoiser,
    time_predictor,
    noise_pred_key="eps_pred",
    time_pred_key="t_pred",
    time_key="t",
    cutoff=5.,
    recompute_neighbors=False,
    save_progress= True,
    progress_stride = 1,
    results_on_cpu = True
)

mrd_itp_sampler = MoreRedITP(
    diff_proc,
    ddpm_denoiser,
    time_predictor,
    noise_pred_key="eps_pred",
    time_pred_key="t_pred",
    time_key="t",
    cutoff=5.,
    recompute_neighbors=False,
    save_progress= True,
    progress_stride = 1,
    results_on_cpu = True
)

tut_path = "./tut"

os.makedirs(tut_path, exist_ok=True)

transforms=[
    trn.CastTo64(),
    trn.SubtractCenterOfGeometry(),
]

# path to store the dataset as ASE '.db' files
datapath = os.path.join(tut_path, "svenelzes_qm7x.db")
S = np.load(os.path.join(tut_path, split_file_path))
print(S["train_idx"].shape, S["val_idx"].shape, S["test_idx"].shape)
data = QM7X(
     datapath=datapath,
     only_equilibrium=False,
     batch_size=8,
     transforms=transforms,
     split_file = os.path.join(tut_path, split_file_path),
     num_train=128,
     num_val=128,
     num_test=128,
     num_workers=2,
     pin_memory=False,
     load_properties=["rmsd"],
 )

 # prepare and setup the dataset
data.prepare_data()
data.setup()

# train split here is not the same as during training
batch = next(iter(data.train_dataloader()))
t = 150
batch[properties.R], _ = diff_proc.diffuse(batch[properties.R], batch[properties.idx_m], t=torch.tensor(t))

print(batch[properties.R].shape)
print((scatter_mean(batch[properties.R], batch[properties.idx_m], batch[properties.n_atoms]) < 1e-5).all())

# ddpm require time step as input
relaxed_ddpm, num_steps_ddpm, hist_ddpm = ddpm_sampler.denoise(batch, t=t)

# morered models dont need time step as input but max steps to interrupt the denoising process
relaxed_jt, num_steps_jt, hist_jt = mrd_jt_sampler.denoise(batch, max_steps=1000)
relaxed_as, num_steps_as, hist_as = mrd_as_sampler.denoise(batch, max_steps=1000)
relaxed_itp, num_steps_itp, hist_itp = mrd_itp_sampler.denoise(batch, max_steps=1000)