import logging
import os
import pickle
from typing import Optional

import numpy as np
from schnetpack import properties
from schnetpack.data import AtomsLoader, load_dataset
from schnetpack.datasets import QM9
import schnetpack.properties as structure
from tqdm import tqdm


from scipy.spatial.transform import Rotation
import torch as th
import torch
__all__ = ["QM9Filtered"]
log = logging.getLogger(__name__)

class QM9Filtered(QM9):
    """
    QM9 dataset with a filter on the number of atoms.
    Only molecules of specific size are loaded.
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        n_atoms_allowed: Optional[int] = None,
        shuffle_train: bool = True,
        indices_path: str = "n_atoms_indices.pkl",
        n_overfit_molecules: Optional[int] = None,
        permute_indices: bool = False,
        repeat_indices: int = 0,
        train_rotate: bool = True,
        train_reflection: bool = True,
        train_rotate_n_copies: int = 2,
        **kwargs
    ):
        """
        Args:
            datapath: path to directory containing QM9 database.
            batch_size: batch size.
            n_atoms_allowed: the exact number of atoms of each molecule.
            shuffle_train: whether to shuffle the training set.
            indices_path: path to pickle file containing indices of molecules
                          with specific number of atoms.
            n_overfit_molecules: number of molecules to overfit on.
            permute_indices: whether to permute the indices of molecules with
                             specific number of atoms.
            repeat_indices: whether to repeat the indices of molecules
        """
        super().__init__(datapath=datapath, batch_size=batch_size, **kwargs)

        self.n_atoms_allowed = n_atoms_allowed
        self.indices_path = indices_path
        self.shuffle_train = shuffle_train
        self.n_overfit_molecules = n_overfit_molecules
        self.permute_indices = permute_indices
        self.repeat_indices = repeat_indices
        self.train_rotate = train_rotate
        self.train_reflection = train_reflection    
        self.train_rotate_n_copies = train_rotate_n_copies

    def setup(self, stage: Optional[str] = None):
        """
        Overwrites the ``setup`` method to load molecules with given number of atoms.
        """
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # (re)load datasets
        if self.dataset is None:
            self.dataset = load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
            )

            # use only molecules with specific number of atoms
            if self.n_atoms_allowed is not None and self.n_atoms_allowed > 0:
                # load indices
                if os.path.exists(self.indices_path):
                    with open(self.indices_path, "rb") as file:
                        indices = pickle.load(file)
                else:
                    indices = {}

                # get indices of molecules with specific number of atoms
                if self.n_atoms_allowed in indices.keys():
                    indices = indices[self.n_atoms_allowed]
                else:
                    tmp = []
                    for i in tqdm(range(len(self.dataset))):
                        if self.dataset[i][properties.n_atoms] == self.n_atoms_allowed:  # type: ignore
                            tmp.append(i)
                    indices[self.n_atoms_allowed] = tmp

                    # persist indices
                    with open(self.indices_path, "wb") as file:
                        pickle.dump(indices, file)
                    indices = indices[self.n_atoms_allowed]

            # get all indices (with any number of atoms)
            else:
                indices = list(range(len(self.dataset)))

            # overfit on a subset of molecules
            if self.n_overfit_molecules is not None and self.n_overfit_molecules > 0:
                # permute indices before overfitting
                if self.permute_indices:
                    indices = np.random.permutation(indices).tolist()

                indices = indices[: self.n_overfit_molecules] * (
                    int(len(indices) / self.n_overfit_molecules)
                    + (len(indices) % self.n_overfit_molecules)
                )

                if self.repeat_indices > 1:
                    indices = indices * self.repeat_indices

                logging.warning(
                    "Overfitting on {} molecules with indices {}".format(
                        self.n_overfit_molecules, indices[: self.n_overfit_molecules]
                    )
                )

            # subset dataset
            self.dataset = self.dataset.subset(indices)

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # partition dataset
            log.info(f"Partitioning dataset with {len(self.dataset)} molecules and train dataset size {len(self.train_idx)}")
            self._train_dataset = self.dataset.subset(self.train_idx)  # type: ignore
            self._val_dataset = self.dataset.subset(self.val_idx)  # type: ignore
            self._test_dataset = self.dataset.subset(self.test_idx)  # type: ignore
            log.info(f"Train dataset has {len(self._train_dataset)} molecules and is of type {type(self._train_dataset)}")
            """
            structure keys
            [2025-09-09 15:57:01,623][morered.datasets.qm9_filtered][INFO] -
              structure list schnetpack ['Final', 'R', 'R_strained', 'Rij', 'Rij_lr', 'Z',
               '__annotations__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__',
                '__spec__', 'cell', 'cell_strained', 'dipole_derivatives', 'dipole_moment', 'electric_field', 
                'energy', 'forces', 'hessian', 'idx', 'idx_i', 'idx_i_lr', 'idx_i_triples', 'idx_j', 'idx_j_lr',
                 'idx_j_triples', 'idx_k_triples', 'idx_m', 'lidx_i', 'lidx_j', 'magnetic_field', 'masses',
                  'n_atoms', 'n_nbh', 'nuclear_magnetic_moments', 'nuclear_spin_coupling', 'offsets', 'offsets_lr',
                   'partial_charges', 'pbc', 'polarizability', 'polarizability_derivatives', 'position', 
                   'required_external_fields', 'seg_m', 'shielding', 'spin_multiplicity', 'strain', 'stress', 'total_charge']
            """
        ## MORERED ADJUSTMENT
        log.info(f"loaded {type(self.dataset)} with {len(self.dataset)} molecules")
        self._setup_transforms()

    def train_dataloader(self) -> AtomsLoader:
        """
        get training dataloader
        """
        if self._train_dataloader is None:
            log.info(f"instantion train dataloader with batch size {self.batch_size}")
            self._train_dataloader = AtomsLoader(
                self.train_dataset,  # type: ignore
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle_train,
                pin_memory=self._pin_memory is not None and self._pin_memory,
                collate_fn = self.train_collate_fn
            )
        return self._train_dataloader

    def _get_random_rotations(self,n_samples, device, dtype: th.dtype | None = None) -> th.Tensor:
        R = Rotation.random(
            n_samples,
        ).as_matrix()
        dtype = th.get_default_dtype()
        R = th.tensor(R, dtype=dtype)
        return R.to(device)


    def _get_random_reflections(
        self,n_samples, device, reflection_share=0.5, eps=1e-9, dtype: th.dtype | None = None
    ) -> th.Tensor:
        dtype = th.get_default_dtype()
        # get random normal vectors
        normals = th.randn(n_samples, 3, dtype=dtype).to(device)  # (n_samples, 3)
        normals = normals / (th.norm(normals, dim=1, keepdim=True) + eps)  # (n_samples, 3)

        # get householder matrix
        normals = normals.unsqueeze(2)
        outer = th.matmul(normals, normals.transpose(1, 2))  # (n_samples, 3, 3)
        identity = th.eye(3, dtype=normals.dtype, device=normals.device).unsqueeze(0)  # (1, 3, 3)
        householder = identity.repeat(n_samples, 1, 1)  # (n_samples, 3, 3)
        # selectively reflect
        sample_mask = th.rand(n_samples) < reflection_share
        householder[sample_mask] -= 2 * outer[sample_mask]
        assert householder.dtype == dtype
        return householder
    
    def train_collate_fn(self,batch):
        """
        Build batch from systems and properties & apply padding

        Args:
            examples (list):

        Returns:
            dict[str->torch.Tensor]: mini-batch of atomistic systems
        """
        elem = batch[0]
        idx_keys = {structure.idx_i, structure.idx_j, structure.idx_i_triples}
        # Atom triple indices must be treated separately
        idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}
        ##MoreRed Adjustment
        #randomly rotate adn reflect batch (by creating copies)
        if self.train_rotate:
            augmented_batch = []
            augmented_batch.extend(batch)  # include original batch
            for sample in batch:
                positions = sample[structure.R]  # original positions
                # Create n copies
                for _ in range(self.train_rotate_n_copies):
                    new_sample = {k: v.clone() for k, v in sample.items()}  # deep copy
                    # Random rotation
                    R = self._get_random_rotations(1, positions.device)  # returns (1,3,3)
                    rotated_positions = torch.bmm(positions.unsqueeze(0), R).squeeze(0)
                    new_sample[structure.R] = rotated_positions

                    if self.train_reflection:
                        H = self._get_random_reflections(1, positions.device, reflection_share=0.5)
                        rotated_positions = torch.bmm(rotated_positions.unsqueeze(0), H).squeeze(0)
                        new_sample[structure.R] = rotated_positions
                    augmented_batch.append(new_sample)
            batch = augmented_batch  # replace original batch with augmented batch 
        
        coll_batch = {}
        for key in elem:
            if (key not in idx_keys) and (key not in idx_triple_keys):
                coll_batch[key] = torch.cat([d[key] for d in batch], 0)
            elif key in idx_keys:
                coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

        seg_m = torch.cumsum(coll_batch[structure.n_atoms], dim=0)
        seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
        idx_m = torch.repeat_interleave(
            torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms], dim=0
        )
        coll_batch[structure.idx_m] = idx_m

        for key in idx_keys:
            if key in elem.keys():
                coll_batch[key] = torch.cat(
                    [d[key] + off for d, off in zip(batch, seg_m)], 0
                )

        # Shift the indices for the atom triples
        for key in idx_triple_keys:
            if key in elem.keys():
                indices = []
                offset = 0
                for idx, d in enumerate(batch):
                    indices.append(d[key] + offset)
                    offset += d[structure.idx_j].shape[0]
                coll_batch[key] = torch.cat(indices, 0)

        ##MORERED ADJUMENT

        #generate batch that is properly padded and doesnt rely on idx list, so the EdgeTransfomer can work with it
        device = coll_batch["_atomic_numbers"].device
        bs = coll_batch[structure.n_atoms].shape[0]
        max_atoms = coll_batch[structure.n_atoms].max()


        mask = th.arange(max_atoms, device=device).unsqueeze(0) < coll_batch["_n_atoms"].unsqueeze(1)

    
        atomic_numbers_padded = th.zeros(bs, max_atoms, dtype=coll_batch["_atomic_numbers"].dtype, device=coll_batch["_atomic_numbers"].device)
        positions_padded = th.zeros(bs, max_atoms, 3, dtype=coll_batch["_positions"].dtype, device=coll_batch["_positions"].device)
        
        for i in range(bs):
            n = coll_batch[structure.n_atoms][i]
            atomic_numbers_padded[i, :n] = coll_batch["_atomic_numbers"][coll_batch["_idx_m"] == i]
            positions_padded[i, :n] = coll_batch[structure.R][coll_batch["_idx_m"] == i]

        coll_batch["mask"] = mask
        coll_batch["_atomic_numers_padded"] = atomic_numbers_padded
        coll_batch["_positions_padded"] = positions_padded

        return coll_batch

    def test():
        pass
        return None
        #so QM9 Filtered inherits from QM9
        #https://schnetpack.readthedocs.io/en/latest/_modules/datasets/qm9.html#QM9
        #QM9 intherits from AtomsDtaModule
        #in QM9 preparte_dataset it calls load_dataset(self.datapath,self.format) and there the len(dataset) is 133885
        #https://schnetpack.readthedocs.io/en/latest/_modules/data/atoms.html#load_dataset
        #load dataset returns an BaseAtomsData object (by calling ASEAtomsData)
        #AseAtomsData inherits from BaseAtomsData

        #AtomsDataModule(from which QM9 Inherits) https://schnetpack.readthedocs.io/en/latest/_modules/data/datamodule.html#AtomsDataModule
        #AtomsDataModule inherits from LightningDataModule (official Pytorch Lighniting)



        #We have intersting functions: AtomsDataModule getaomsrefs and 
        #https://schnetpack.readthedocs.io/en/latest/_modules/data/loader.html#AtomsLoader

