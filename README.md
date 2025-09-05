# Molecular Diffusion with Non-Equivariant Backbone Representations

This project explores the use of **non-equivariant backbone representations** in molecular diffusion models. The goal is to investigate whether relaxing equivariance constraints can improve flexibility and training efficiency in molecular modeling.

## Projects Integrated

1. **[MoreRed (Diffusion)](https://github.com/khaledkah/MoreRed)**  
   - Uses **PaiNN** as an **equivariant representation** of molecules.  
   - Equivariant representations can restrict the model and impose certain limitations on training and further development.

2. **[Simple-MDET: EdgeTransformer](https://github.com/mx-e/simple-md)**  
   - Demonstrates **non-equivariance** in the context of molecular dynamics.  
   - Achieved by applying rotations and flips to molecules during training.
   - uses an EdgeTransformer

3. I took the EdgeTransformer and tried to make it work within the MoreRed (and in turn schnetpack and PyTorch Lightning) context.


## Code & Setup

- Designed to run on **HPC systems**.
- **SLURM script**: [MoreRed/train_script.sh](MoreRed/train_script.sh)
- **Container**: Build a Singularity image from [MoreRed/old_container.def](MoreRed/old_container.def)
- **Dependencies**:  
  - `schnetpack` has been updated to `v2.1.1` in `MoreRed/requirements.txt` to ensure compatibility with the EdgeTransformer implementation and CUDA support.

## TODO

- Refactor code: move **data-related operations** from EdgeTransformer to a more appropriate location.
- Further modularization to improve maintainability and reproducibility.

