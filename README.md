# Molecular Diffusion with Non-Equivariant Backbone Representations

This project explores the use of **non-equivariant backbone representations** in molecular diffusion models. The goal is to investigate whether relaxing equivariance constraints can improve flexibility and training efficiency in molecular modeling.

## Refactor Goals :suspect: :trollface:

- Make Qm7 work (dont forget to bind correctly)
- Move the Mask thing into Dataloader. :white_check_mark:
- See if the loss works? Shoudl though, the 0 dont change a thing?
- Check if the random transformation and center really works
- investigate the numbers of train samples from the MoreRed paper? Something is sus
- properties.Z nuclear charge from schnetpack wiki 
- Investigate if we need to valid_forces = f[mask] even, or the output is padded fine?
- Extend the namespace schnetspace.structure with our own custom, combining both the heads_pe namespaces in there.(with frozendict)
- found atleast spin multiplicty 
- which one is charge
'partial_charges', 'total_charge' or properties.Z
- remove the force adjustmend of the transformer
- move maybe the mask to general transform hehe :3
- create alias file like Ole D.
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
## Things I did specifically
- Mayor changes are to be seen in MoreRed/src/morered/model/heads_pe.py
- MoreRed uses a single concetanted list with all molecules in the batch in that single list. It then uses a different list to keep track of which part of that list belonds to which molecule. The ET expects a padded mutltidimensional tensor. So i wrote code to convert these two into each other.
 - Adding the time as a parameter. I was not sure so i appeneded it a bit ugly
 ```
#time is the same, its just broadcasted, so its safe to do this
            t0 = inputs[self.time_key][0]
            #append it with 12, so it maches num_heads in transformer layer
            time = th.full((x.shape[0],x.shape[1],x.shape[2],12),t0, device=x.device, dtype=x.dtype)
            x = th.cat([x, time], dim=-1)
```
- included postprocess in the model, as we ripped it out of the schnetpack pipeline
- slight structure changes to fall in line with the Atomistic Task 

## TODO

- Refactor code: move **data-related operations** from EdgeTransformer to a more appropriate location.(The reforming from long lsit to masked and padding and back).
- Include the JT Appraoch of MoreRed, where the EdgeTransfomer also predicts time. This should be very simple, as we already have the heads part given.

- adding Rotational Noise to the diffusion process in the hope of learning equivariance more.
- investigate long waiting time before start of training (possible JIT precompilation)
- tink about removing the post process forces (Post-processed 2.63% of forces with mask) part from ET, does it make sense in the context of forces?

