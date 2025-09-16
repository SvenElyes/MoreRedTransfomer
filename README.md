# Molecular Diffusion with Non-Equivariant Backbone Representations

This project explores the use of **non-equivariant backbone representations** in molecular diffusion models. The goal is to investigate whether relaxing equivariance constraints can improve flexibility and training efficiency in molecular modeling.

## Refactor Goals :suspect: :trollface:


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
  - `schnetpack` has been updated to `v2.1.1` in `MoreRed/requirements.txt` to ensure compatibility with the EdgeTransformer implementation and CUDA support. (MoreRed was built on 2.0.3)
## Things I did specifically
- Mayor changes are to be seen in the specific model file [Model]  (MoreRed/src/morered/model/heads_pe.py)
- MoreRed uses a single concetanted list with all molecules in the batch in that single list. It then uses a different list to keep track of which part of that list belonds to which molecule. The ET expects a padded mutltidimensional tensor. The extension has been made in the QM9 data file [Here] (MoreRed/datasets/qm9_filtered.py)
 - Adding the time as a parameter. I was not sure so i appeneded it a bit ugly
 ```
#time is the same, its just broadcasted, so its safe to do this
            t0 = inputs[self.time_key][0]
            #append it with 12, so it maches num_heads in transformer layer
            time = th.full((x.shape[0],x.shape[1],x.shape[2],12),t0, device=x.device, dtype=x.dtype)
            x = th.cat([x, time], dim=-1)
```
Just some thoughts
   - this is about how where appending time influeces the workflow. This is especially important in the context of JT. As far as i understand. PAINN gives us a representation, we work with in heads.py. This representation(with scalar and vector operation), is then given to a gated block. Time is appended towards the input(scalar) The gated block is expanded by one dimension(for the time). In the case of JT, the time is predicted by a head, that is also working with the painn_represenation
   - so in the context of our Transformer model, the time needs to be appended after the embedding or? Because if we embedd we cant predict time on that? Or maybe we cut out the time then?
   - i think the best thing to do, is appending time after the embedding, if JTcase we predict time or just append true time.
    
- included postprocess in the model, as we ripped it out of the schnetpack pipeline (Center the batch), also checked if this really works (it does :troll_face:)
- removed the models post process, which regulated too big of forces, as the contect changed
- slight structure changes to fall in line with the Atomistic Task (variable of gradient needs to be specified, certain parameters certainly named, so the Task finds it)


## TODO
- masked thing also has to go into test and val set or?
- Refactor code: move **data-related operations** from EdgeTransformer to a more appropriate location.(The reforming from long lsit to masked and padding). :white_check_mark: 
   - vectorizing the loading of data into the padded arrays? Is it possible? Maybe move a bit ahead
- Include the JT Appraoch of MoreRed, where the EdgeTransfomer also predicts time. This should be very simple, as we already have the heads part given.
- adding Rotational Noise to the diffusion process in the hope of learning equivariance more.
- investigate long waiting time before start of training (possible JIT precompilation)
- Make a comparison run of the models own postprocessrun and have it as a Hydra Variable?
- Make Qm7 work (dont forget to bind correctly)
   -using QM7x from quantum max datasets doesnt work bc its missing some key dcit metadata json file  
   -usign QM7x.sqfs from shared_datasets i get 
      svenelzes@hydra:~/MoreRedTransfomer/MoreRed$ file /home/space/datasets-sqfs/QM7-X.sqfs
      /home/space/datasets-sqfs/QM7-X.sqfs: data
      which means its not correc squashfs(accrdoing to chatgpt)
      FATAL ERROR: Can't find a valid SQUASHFS superblock on QM7-X.sqfs
   -redownlaoded and parsed qm7 and will move it into /home/space/datasets-sqfs/(TU BERLIN HYDRA HPC) under QM7-X_svenelzes.sqfs (in progress)

- investigate the numbers of train samples from the MoreRed paper? Something is sus :white_check_mark:
   - 130000 samples in QM9, 55000 in datatraining (with batchsize 4) we get 13500 steps in one epoch. Seems fine
- properties.Z nuclear charge from schnetpack wiki. Is it the atom_number or maybe even the charge we set to 1 by deafult
- Investigate if we need to valid_forces = f[mask] even, or the output is padded fine? :white_check_mark:
   - seems fine to do it like this. Returning padded output would mean we would need 
- Extend the namespace schnetspace.structure with our own custom, combining both the heads_pe namespaces in there.(with frozendict)
- found atleast spin multiplicty in the keys avaibale of schnetpack.properties? But is it in QM9/QM7-X
- which one is charge 'partial_charges', 'total_charge' or properties.Z (all in schnetpack.properties)
- move maybe the mask to general transform 
- create alias file like Ole D. to streamline working in terminal
- investigate if we can extend batch size and what and how? 
- do a trainrun of normal circumstances to have proper control value :suspect:
- do a summary of pdf
- clean up folder structure of .sh .py and container/apptainer stuff.
- clean up logs and where each different case (JT, normal, clean etc. go)
- understanding the cast64 cast32 its purpose and when and how they are used in the original dataset.
- should self.required_derivatives = ["forces"] also include Time in the context.
- how do in ET adn normal MoreRed propagte the loss of time?


- the time is appended in MoreRed after everything but right before the head? Maybe i Have appended the time too early? bc now a JT is a bit difficult?
- about batch size, from MoreRed
from undergoing large unrealistic movements during the initial sampling steps. We use a large batch size of
128 molecules to improve the accuracy of the loss estimation, as it involves uniformly sampling a single
diffusion step t per molecule per batch instead of using the whole trajectory for each molecule in the batch.

- in what context/ scenario do you detach the time head?
- should i implement a version where painn atleast precits time (no me gusta lo)
- using 128 batchsize on 3090 RTX gives OOM.
- make tensorbardlogger work
- investigate Sampler in QM9 Dataclass (and AtomsDataMOdule)
- make the rotation and flipping part of a transform and give it rhough the trasnformer attribute in the Hydra YAML, under data.
- solve the idea of batchsize more elegant.(we have to choose small abtch size because we augment it by *3 in the train) Make 
- make things ready for the great train. Does DDPM and JT work corrrectly(check with logs in the heads to see if everything is smoth)
- check from mdet,w hich h1000 heads are fine (bc of cuda)
- investigate which think in morered.py kills my mask (it was compute neighbors)
- introdcue a flag that makes the mask optional, so it still runs (in normal mode)
- profiler doesnt work with my MDET because its not perfectiny in line pytorchlightning lingo and callback
- th .compile makes for small batches(train no sense). 5mintues extra.
- number of Parameters?