import math
from functools import partial
from typing import Literal, Optional

import torch as th
#from lib.types import PipelineConfig, PropertyType, property_dims, property_type
#from lib.types import Property as Props
from torch import nn
from torch.nn import functional as F

#from lib.utils.dist import set_dtype, assert_dtype
#from loguru import logger

NODE_FEATURES_OFFSET = 128
__all__ = ["PairEncoder"]

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import torch as th
from frozendict import frozendict
from torch.utils.data import Dataset
from morered.transform import BatchSubtractCenterOfMass

NODE_FEATURES_OFFSET = 128

import logging
log = logging.getLogger(__name__)


def set_dtype(fp_type: th.dtype | None) -> th.dtype:
    if fp_type is None:
        return th.get_default_dtype()
    if fp_type != th.get_default_dtype():
        log.warning(
            f"Using {fp_type} precision, deviating from default precision {th.get_default_dtype()}. This is probably unintended."
        )
    return fp_type


def assert_dtype(tensor: th.Tensor, dtype: th.dtype) -> None:
    assert tensor.dtype == dtype, f"Tensor should be of type {dtype}, but is {tensor.dtype}"


class Property(Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value) -> "Property":
        # This allows Property["energy"] to work the same as Property.energy
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")

    energy = "energy"
    formation_energy = "formation_energy"
    charge = "charge"
    multiplicity = "multiplicity"
    atomic_numbers = "_atomic_numbers"
    mask = "mask"
    forces = "forces"
    force_magnitudes = "force_magnitudes"
    force_directions = "force_directions"
    force_directions_euler = "force_directions_euler"

    positions = "_positions" #adjusted for Morered
    dipole = "dipole"
    mol_idx = "mol_idx"
    i_idx = "i_idx"
    j_idx = "j_idx"
    energy_atomref = "energy_atomref"
    formation_energy_atomref = "formation_energy_atomref"
    r_ij = "r_ij"
    i_idx_local = "i_idx_local"
    j_idx_local = "j_idx_local"
    n_atoms = "n_atoms"
    _meta = "_meta"

    ##MORERED ADJUSTEMNT
    atomic_numbers_padded = "_atomic_numbers_padded"
    positions_padded="_positions_padded"


class PropertyType(Enum):
    mol_wise = 1
    atom_wise = 2
    edge_wise = 3


property_dims = frozendict(
    {
        Property.energy: 1,
        Property.formation_energy: 1,
        Property.energy_atomref: 1,
        Property.charge: 1,
        Property.multiplicity: 1,
        Property.atomic_numbers: 1,
        Property.mask: 1,
        Property.forces: 3,
        Property.force_magnitudes: 1,
        Property.force_directions: 3,
        Property.force_directions_euler: 2,
        Property.positions: 3,
        Property.mol_idx: 1,
        Property.dipole: 3,
        Property.i_idx: 1,
        Property.j_idx: 1,
        Property.energy_atomref: 1,
        Property.formation_energy_atomref: 1,
        Property.r_ij: 3,
        Property.i_idx_local: 1,
        Property.j_idx_local: 1,
        Property.n_atoms: 1,
    }
)

property_dtype_dict = frozendict(
    {
        Property.energy: th.float32,
        Property.formation_energy: th.float32,
        Property.charge: th.int64,
        Property.multiplicity: th.int64,
        Property.atomic_numbers: th.int64,
        Property.mask: th.bool,
        Property.forces: th.float32,
        Property.force_magnitudes: th.float32,
        Property.force_directions: th.float32,
        Property.force_directions_euler: th.float32,
        Property.positions: th.float32,
        Property.mol_idx: th.int64,
        Property.dipole: th.float32,
        Property.i_idx: th.int64,
        Property.j_idx: th.int64,
        Property.energy_atomref: th.float32,
        Property.formation_energy_atomref: th.float32,
        Property.r_ij: th.float32,
        Property.i_idx_local: th.int64,
        Property.j_idx_local: th.int,
        Property.n_atoms: th.int64,
    }
)


# we need this workaround to be flexible with float32 and float64
def property_dtype(prop: Property) -> th.dtype:
    dtype = property_dtype_dict[prop]
    if dtype == th.float32 and th.get_default_dtype() == th.float64:
        return th.float64
    return dtype


property_type = frozendict(
    {
        Property.energy: PropertyType.mol_wise,
        Property.formation_energy: PropertyType.mol_wise,
        Property.charge: PropertyType.mol_wise,
        Property.multiplicity: PropertyType.mol_wise,
        Property.atomic_numbers: PropertyType.atom_wise,
        Property.mask: PropertyType.atom_wise,
        Property.forces: PropertyType.atom_wise,
        Property.force_magnitudes: PropertyType.atom_wise,
        Property.force_directions: PropertyType.atom_wise,
        Property.force_directions_euler: PropertyType.atom_wise,
        Property.positions: PropertyType.atom_wise,
        Property.mol_idx: PropertyType.edge_wise,
        Property.dipole: PropertyType.mol_wise,
        Property.i_idx: PropertyType.edge_wise,
        Property.j_idx: PropertyType.edge_wise,
        Property.energy_atomref: PropertyType.mol_wise,
        Property.formation_energy_atomref: PropertyType.mol_wise,
        Property.r_ij: PropertyType.atom_wise,
        Property.energy_atomref: PropertyType.mol_wise,
        Property.i_idx_local: PropertyType.edge_wise,
        Property.j_idx_local: PropertyType.edge_wise,
    }
)

edge_props_to_local_map = frozendict({Property.i_idx: Property.i_idx_local, Property.j_idx: Property.j_idx_local})


@dataclass
class FeatureStats:
    mean: float
    var: float
    atomref: dict[int, float]

Props = Property

class PairEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        embd_dim: int,
        num_3d_kernels: int,
        cls_token: bool,
        num_heads: int,
        activation: str,
        ffn_multiplier: int,
        attention_dropout: float,
        ffn_dropout: float,
        head_dropout: float,
        norm_first: bool,
        norm: Literal["batch", "layer"],
        decomposer_type: Literal["pooling", "diagonal"],
        target_heads: list[str],
        head_project_down: bool,
        compose_dipole_from_charges: bool = False,
        use_electronic_embeddings: bool = True,
        #added for morered compability
        include_time: bool = False,
        output_key: str = "eps_pred",
        time_head: Optional[nn.Module] = None,
        time_key: Optional[str] = None,
        detach_time_head: bool = False,
        
        do_postprocessing: bool = True,
        do_preprocessing: bool = True,

    ) -> None:
        super().__init__()
        #MORERED ADJUSTMENT as schnetpack Task needs required derivatives
        self.required_derivatives = ["forces"]
        self.include_time = include_time
        self.time_key = time_key 
        self.embedding = PairEmbedding(embd_dim, num_3d_kernels, cls_token, use_electronic_embeddings)
        self.composer = Composer(embd_dim)
        assert decomposer_type in ["pooling", "diagonal"], "Invalid decomposer type"
        self.decomposer_class = DiagonalDecomposer if decomposer_type == "diagonal" else PoolingDecomposer
        self.decomposer = self.decomposer_class(embd_dim)
        self.layers = nn.ModuleList(
            [
                EdgeTransformerLayer(
                    embd_dim,
                    num_heads,
                    ffn_dropout,
                    ffn_multiplier,
                    attention_dropout,
                    activation,
                    norm,
                    norm_first,
                )
                for _ in range(n_layers)
            ]
        )
        
        target_heads = [Props[t] for t in target_heads]
        self.heads = nn.ModuleDict(
            {
                str(target): NodeLevelRegressionHead(
                    target,
                    embd_dim=embd_dim,
                    cls_token=cls_token,
                    activation=activation,
                    head_dropout=head_dropout,
                    project_down=head_project_down,
                )
                for target in target_heads
            }
        )
        if Props.dipole in self.heads and compose_dipole_from_charges:
            self.heads["dipole"].set_compose_from_charges(True)
        
        if self.include_time:
            self.layers = nn.ModuleList(
            [
                EdgeTransformerLayer(
                    embd_dim+12,
                    num_heads,
                    ffn_dropout,
                    ffn_multiplier,
                    attention_dropout,
                    activation,
                    norm,
                    norm_first,
                )
                for _ in range(n_layers)
            ]
        )
            self.decomposer = self.decomposer_class(embd_dim+12)
            self.heads = nn.ModuleDict(
            {
                str(target): NodeLevelRegressionHead(
                    target,
                    embd_dim=embd_dim+12,
                    cls_token=cls_token,
                    activation=activation,
                    head_dropout=head_dropout,
                    project_down=head_project_down,
                )
                for target in target_heads
            }
        )

        
        ##MORERED ADJUSTMENT
        #postprocessing in schnetpack is done using a nn.modulelist, where each postprocessing is done as a nn.module

        self.do_postprocessing = do_postprocessing
        transform_model = BatchSubtractCenterOfMass(name=output_key, dim=3)
        self.postprocessors = nn.ModuleList([transform_model])
        self.output_key = output_key

        self.do_preprocessing = do_preprocessing
        rotatation_model = None
        self.preprocessors = nn.ModuleList([rotatation_model])


    def initialize_transforms(self, datamodule):
        """
        Called once during `setup("fit")` in AtomisticTask.
        Use this to attach any preprocessing transforms that depend on dataset statistics.
        """
        self.transforms = []
        
    def reset_parameters(self) -> None:
        self.apply(self._init_weights)

    def post_process(self, out: dict) -> dict:
        limit = 2.0
        forces = out[Props.forces]
        forces_norm = th.norm(forces, dim=-1)
        mask = forces_norm < limit
        if mask.sum() < forces_norm.numel():
            out[Props.forces] = forces * mask.unsqueeze(-1)
            log.warning(f"Post-processed {1 - (mask.sum() / mask.numel()):.2%} of forces with mask")
        return out

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs) -> dict:

        """Input 
        Input keys(['_idx', 'energy_U0', '_n_atoms', '_atomic_numbers', '_positions', '_cell', '_pbc', 'original__positions', 'eps', 't', '_idx_i_local', '_idx_j_local', '_offsets', '_idx_m', '_idx_i', '_idx_j', '_Rij', 'scalar_representation', 'vector_representation'])
         shapes: [torch.Size([128]), torch.Size([128]), torch.Size([128]), torch.Size([1577]), torch.Size([1577, 3]), torch.Size([128, 3, 3]), torch.Size([384]), torch.Size([1577, 3]), torch.Size([1577, 3]), torch.Size([1577]), torch.Size([19014]), torch.Size([19014]), torch.Size([19014, 3]), torch.Size([1577]), torch.Size([19014]), torch.Size([19014]), torch.Size([19014, 3]), torch.Size([1577, 256]), torch.Size([1577, 3, 256])]
        """

        log.info(f"Start Ecnoder input keys: {inputs.keys()} and shapes: {[inputs[k].shape for k in inputs.keys()]}")
        
        h, e, mask = self.embedding(inputs)
        x = self.composer((h, e, mask)) #inject node info into edge ?b,n,n,embed_dim

        unbatch = mask.unsqueeze(2) * mask.unsqueeze(1)  # B x N x N
        x_mask = unbatch.unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)  # B x N x N x N

        ###MORERED ADJUSTMENT
        #append time to input
        if self.include_time:

            #time is the same, its just broadcasted, so its safe to do this
            t0 = inputs[self.time_key][0]
            #append it with 12, so it maches num_heads in transformer layer
            time = th.full((x.shape[0],x.shape[1],x.shape[2],12),t0, device=x.device, dtype=x.dtype)
            x = th.cat([x, time], dim=-1)

            
        for layer in self.layers:
            x = layer(x, x_mask)

        h = self.decomposer(x)
        out = {Props[k]: head(h, inputs) for k, head in self.heads.items()}

        
        out["embd"] = h
        out = self.post_process(out)
        ##MORERED ADJUSTMENT

        #         log.info("Type of out: %s", type(out))
        #         log.info("Keys in out: %s", list(out.keys()))
        #         log.info("Example value type in out: %s", type(next(iter(out.values()))))
        #         log.info(f"Forces shape: {out[Props.forces].shape if Props.forces in out else 'N/A'}")'
        #we have to flatten(remove the padding) the output again. 
        f = out[Props.forces]
        valid_forces = f[mask]
        #log.info(f"Valid forces shape: {valid_forces.shape}")
        
        #rudementary postrproess not in line with schnetpack strucutre.
        
        inputs[self.output_key] = valid_forces
        for post in self.postprocessors:
            valid_forces= post(inputs)

        #predict time  
        return inputs


def pairwise_directions_from_positions(positions: th.Tensor, eps=1e-5) -> th.Tensor:
    b, n, _ = positions.shape  # positions: (b, n, 3)
    positions_i = positions.unsqueeze(2).expand(-1, -1, n, -1)  # (b, n, n, 3)
    positions_j = positions.unsqueeze(1).expand(-1, n, -1, -1)  # (b, n, n, 3)
    directions = positions_j - positions_i  # (b, n, n, 3)
    normed_directions = directions / (th.norm(directions, dim=-1, keepdim=True) + eps)
    # get azimuth and polar angles
    azimuth = th.atan2(normed_directions[..., 1], normed_directions[..., 0])  # (b, n, n)
    polar = th.acos(normed_directions[..., 2])  # (b, n, n)
    return th.stack([azimuth, polar], dim=-1)  # (b, n, n, 2)


def add_graph_level_token(positions, atomic_numbers, mask) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    positions = th.cat(
        [
            th.zeros_like(positions[:, :1]),
            positions,
        ],
        dim=1,
    )
    atomic_numbers = th.cat(
        [
            th.ones_like(atomic_numbers[:, :1]) * 101,
            atomic_numbers,
        ],
        dim=1,
    )
    mask = th.cat(
        [
            th.ones_like(mask[:, :1]),
            mask,
        ],
        dim=1,
    )
    return positions, atomic_numbers, mask


class PairEmbedding(nn.Module):
    def __init__(self, embd_dim: int, num_3d_kernels: int, cls_token: int, use_electronic_embeddings: bool) -> None:
        super().__init__()
        self.num_3d_kernels = num_3d_kernels
        self.cls_token = cls_token
        self.use_electronic_embeddings = use_electronic_embeddings

        # node features
        self.nuclear_embedding = NuclearEmbedding(embd_dim, use_electronic_embeddings)
        self.multiplicity_embed = nn.Embedding(NODE_FEATURES_OFFSET, embd_dim)
        self.charge_embed = nn.Embedding(NODE_FEATURES_OFFSET, embd_dim)
        nn.init.normal_(self.multiplicity_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.charge_embed.weight, mean=0.0, std=0.02)

        # pair features
        self.m3d_embed = Gaussian3DEmbed(embd_dim, (2 * NODE_FEATURES_OFFSET + 1), self.num_3d_kernels)
        self.directional_embed = FourierDirectionalEmbed(embd_dim, num_kernel=128)

    def forward(self, inputs) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        log.info(f"PairEmbedding input keys: {inputs.keys()} and shapes: {[inputs[k].shape for k in inputs.keys()]}")
        log.info(f"Props keys: {[Props.positions, Props.atomic_numbers, Props.mask, Props.multiplicity, Props.charge]}")

        #MORERED ADJUSTMENT

        device = inputs["_atomic_numbers"].device
        #TODO: Remove this and move to DataLoader
        #introduce a mask, which tells us which atoms are present in the graph
        bs = inputs["_n_atoms"].shape[0]
        max_atoms = inputs["_n_atoms"].max()
        
        #arange(max_atoms).unsqueeze(0) -> (1,max_atoms) and inputs ["_n_atoms"].unsqueeze(1) -> (bs,1)

        mask = th.arange(max_atoms, device=device).unsqueeze(0) < inputs["_n_atoms"].unsqueeze(1)
    
        #multiplicity and charge a re both scalar( 1 dim, look up in the frozendict)
        multiplicity = th.ones_like(inputs["_n_atoms"],device=device).unsqueeze(-1)
        charge = th.zeros_like(inputs["_n_atoms"],device=device).unsqueeze(-1) #reference to simple_md where they did the same

        #reshape positions and atomic numbers to allign with the padded approach.
        atomic_numbers_padded = th.zeros(bs, max_atoms, dtype=inputs["_atomic_numbers"].dtype, device=inputs["_atomic_numbers"].device)
        positions_padded = th.zeros(bs, max_atoms, 3, dtype=inputs["_positions"].dtype, device=inputs["_positions"].device)


        #log.info(f"Postionspadded key f{Props.positions} vaklue: {Props.positions.value} dtype: {type(Props.positions.value)} and type {type(Props.positions)}")
        #log.info(f"Atomic nujmbers Props key f{Props.atomic_numbers} vaklue: {Props.atomic_numbers.value} dtype: {type(Props.atomic_numbers.value)} and type {type(Props.atomic_numbers)}")
        #TODO vectorize this please hahahh
        for i in range(bs):
            n = inputs["_n_atoms"][i]
            atomic_numbers_padded[i, :n] = inputs[Props.atomic_numbers.value][inputs["_idx_m"] == i]
            positions_padded[i, :n] = inputs[Props.positions.value][inputs["_idx_m"] == i]

        positions, atomic_numbers, mask, multiplicity, charge = (
            positions_padded,
            atomic_numbers_padded,
            mask,
            multiplicity,
            charge,
        )
        tensors = {
            "positions": positions_padded,
            "atomic_numbers": atomic_numbers_padded,
            "mask": mask,
            "multiplicity": multiplicity,
            "charge": charge,
        }

        #for name, tensor in tensors.items():
            #log.info(f"{name:<15} | shape: {tuple(tensor.shape)} | dtype: {tensor.dtype}")
        # add graph level token at the beginning of the sequence/batch. This one is used to encode the graph level information
        if self.cls_token:
            positions, atomic_numbers, mask = add_graph_level_token(positions, atomic_numbers, mask)
        #embeds information about atoms. Uses a prewritten table. Max-Z is 101 because MORERED uses a maximum of 100 atoms
        h = self.nuclear_embedding(atomic_numbers.long())  # (b,n,e)

        multipl_embed = self.multiplicity_embed(multiplicity)  # (b,1,e)
        charge_embed = self.charge_embed(charge + (NODE_FEATURES_OFFSET // 2))  # (b,1,e)
        g = multipl_embed + charge_embed  # (b,1,e)
        if self.cls_token:
            h[:, 0] += g.squeeze(1)
        else:
            h += g

        # pair features
        D = th.cdist(positions, positions, p=2)  # (b,n,n)
        directions = pairwise_directions_from_positions(positions)  # (b,n,n,3)

        # we don't have edge types
        edge_types = th.zeros_like(D).long().unsqueeze(-1)  # (b,n,n,1)
        e = self.m3d_embed(D, edge_types)  # (b,n,n,e)
        e += self.directional_embed(directions)  # (b,n,n,e)

        return h, e, mask

#composes back to original embedding dimension. inject node infomration into edge_information
class Composer(nn.Module):
    def __init__(
        self,
        embed_dim,
        linear: bool = True,
    ) -> None:
        super().__init__()
        concat_dim = 2 * embed_dim
        self.node_proj = MLP(concat_dim, embed_dim, linear=linear)

    def forward(self, inputs) -> th.Tensor:
        h, e, _ = inputs
        # create pair of node embeddings
        h_i = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)  # (b,n,n,e)
        h_ij = th.cat([h_i, h_i.transpose(1, 2)], dim=-1)  # (b,n,n,2e)
        h_e = self.node_proj(h_ij)  # (b,n,n,e)

        x = e + h_e  # (b,n,n,e)
        return x


class PoolingDecomposer(nn.Module):
    def __init__(self, embd_dim) -> None:
        super().__init__()
        self.node_dim = embd_dim

        self.out_proj = MLP(embd_dim, 2 * embd_dim)
        self.node_mlp = MLP(embd_dim, embd_dim)

    def forward(self, x) -> th.Tensor:
        x1, x2 = self.out_proj(x).chunk(2, dim=-1)  # (b,n,n,2e)
        x = x1 + x2.transpose(1, 2)  # (b,n,n,e)
        x = x.sum(dim=2)  # (b,n,e)
        x = self.node_mlp(x)  # (b,n,e)
        return x


class DiagonalDecomposer(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.out_proj = MLP(embed_dim, embed_dim)
        self.node_mlp = MLP(embed_dim, embed_dim)

    def forward(self, x) -> th.Tensor:
        x = self.out_proj(x)  # (b,n,n,e)
        x = x.diagonal(dim1=1, dim2=2).transpose(-1, -2)  # (b,n,e)
        x = self.node_mlp(x)  # (b,n,e)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_multiplier: int = 2,
        dropout: float = 0,
        activation: str = "relu",
        norm: str = "batch",
    ) -> None:
        super().__init__()

        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "gelu":
            activation_fn = nn.GELU
        else:
            raise ValueError(f"Activation function {activation} is not supported")

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_multiplier),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        if norm == "batch":
            self.norm = nn.BatchNorm1d(embed_dim)
            self.norm_aggregate = nn.BatchNorm1d(embed_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(embed_dim)
            self.norm_aggregate = nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"Norm {norm} is not supported")

        self.dropout_aggregate = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.dropout = dropout

    @th.compile
    def forward(self, x_prior, x) -> th.Tensor:
        x = self.dropout_aggregate(x)
        x = x_prior + x
        x = self.norm_aggregate(x)
        x = self.mlp(x) + x
        return self.norm(x)


class EdgeTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: int,
        attention_dropout: float,
        activation: str = "relu",
        norm: str = "batch",
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first

        self.embed_dim = embed_dim
        self.attention = FastEdgeAttention(embed_dim, num_heads, attention_dropout)

        if norm_first:
            self.norm = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim, ffn_multiplier, dropout, activation, norm)

    def forward(self, x_in, mask=None) -> th.Tensor:
        x = x_in
        if self.norm_first:
            x = self.norm(x)

        x_upd = self.attention(x, x, x, ~mask) if mask is not None else self.attention(x, x, x)
        x = self.ffn(x_in, x_upd)
        return x


def triang_attn(q, k) -> th.Tensor:
    out = q.unsqueeze(3) * k.unsqueeze(1)
    return out.sum(dim=5)


def val_fusion(v1, v2) -> th.Tensor:
    return v1.unsqueeze(3) * v2.unsqueeze(1)


def final_comp(att, val) -> th.Tensor:
    out = att.unsqueeze(-1) * val
    return out.sum(dim=2)


class FastEdgeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, dtype: th.dtype | None = None) -> None:
        super().__init__()
        self.dtype = set_dtype(dtype)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.qlin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.klin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v1lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v2lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.olin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    @th.compile
    def forward(self, query, key, value, mask=None) -> th.Tensor:
        num_batches = query.size(0)
        num_nodes_q = query.size(1)
        num_nodes_k = key.size(1)

        left_k = self.qlin(query)
        right_k = self.klin(key)
        left_v = self.v1lin(value)
        right_v = self.v2lin(value)

        left_k = left_k.view(num_batches, num_nodes_q, num_nodes_q, self.num_heads, self.d_k)
        right_k = right_k.view(num_batches, key.size(1), key.size(2), self.num_heads, self.d_k)
        left_v = left_v.view_as(right_k)
        right_v = right_v.view_as(right_k)

        if hasattr(self, "norms"):
            left_k = self.norms[0](left_k)
            right_k = self.norms[1](right_k)

        scores = triang_attn(left_k, right_k) / math.sqrt(self.d_k)

        if mask is not None:
            scores_dtype = scores.dtype
            scores = scores.to(self.dtype).masked_fill(mask.unsqueeze(4), -1e9).to(scores_dtype)

        att = F.softmax(scores, dim=2)
        att = self.dropout(att)
        val = val_fusion(left_v, right_v)

        if hasattr(self, "norms"):
            val = self.norms[2](val)

        x = final_comp(att, val)
        x = x.view(num_batches, num_nodes_q, num_nodes_k, self.embed_dim)
        return self.olin(x)


class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, dropout: float = 0.0, linear: bool = False) -> None:
        if not linear:
            hidden_dim = output_dim

            layers = [
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(dropout),
            ]
            super().__init__(*layers)
        else:
            super().__init__(
                nn.Linear(input_dim, output_dim),
            )


def get_output_mlp(
    embd_dim: int,
    activation_fn: any,
    project_down: bool,
    head_dropout: float,
    output_dim: int,
):
    return nn.Sequential(
        nn.Linear(embd_dim, embd_dim // 2 if project_down else embd_dim),
        activation_fn(),
        nn.Dropout(head_dropout),
        nn.Linear(
            embd_dim // 2 if project_down else embd_dim,
            embd_dim // 4 if project_down else embd_dim,
        ),
        activation_fn(),
        nn.Dropout(head_dropout),
        nn.Linear(embd_dim // 4 if project_down else embd_dim, output_dim, bias=False),
    )


class NodeLevelRegressionHead(nn.Module):
    def __init__(
        self,
        target: Props,
        embd_dim: int,
        cls_token: bool,
        activation: str,
        head_dropout: float,
        project_down: bool,
        dtype: th.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dtype = set_dtype(dtype)
        self.embd_dim = embd_dim
        self.cls_token = cls_token
        self.project_down = project_down
        self.head_dropout = head_dropout
        self.final_ln_node = nn.LayerNorm(embd_dim)
        if activation == "relu":
            self.act_fn = nn.ReLU
        elif activation == "gelu":
            self.act_fn = nn.GELU
        else:
            raise ValueError(f"Activation function {activation} is not supported")
        self.target = target
        self.output_dim = property_dims[target]
        self.mlp = get_output_mlp(embd_dim, self.act_fn, project_down, head_dropout, self.output_dim)
        self.target_type: PropertyType = property_type[target]
        assert self.target_type in [
            PropertyType.mol_wise,
            PropertyType.atom_wise,
        ], f"Invalid target type {self.target_type}"
        self.dipole_compose_from_charges = False  # only relevant for dipole

    def reset_parameters(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_dipole_compose_from_charges(self, compose: bool) -> None:
        if self.taget != Props.dipole:
            raise ValueError("Can only set compose from charges for dipole head")
        if compose and self.cls_token:
            raise ValueError("Cannot compose dipole from charges with cls token present")
        self.dipole_compose_from_charges = compose
        self.mlp = get_output_mlp(
            embd_dim=self.embd_dim,
            activation_fn=self.act_fn,
            project_down=self.project_down,
            head_dropout=self.head_dropout,
            output_dim=1,
        )

    def forward(self, h, inputs) -> th.Tensor:
        #TODO: Think about how to make this cleaner

        #MoreRed adjustment
        #construct the map again, in the todo one should make it part of the input and remove all this nonsense from the forawrd 
        #mask = inputs[Props.mask]
        #MORERED ADJUSTMENT
        #introduce a mask, which tells us which atoms are present in the graph
        bs = inputs["_n_atoms"].shape[0]
        max_atoms = inputs["_n_atoms"].max()
        
        #arange(max_atoms).unsqueeze(0) -> (1,max_atoms) and inputs ["_n_atoms"].unsqueeze(1) -> (bs,1)
        device = inputs["_n_atoms"].device

        mask = th.arange(max_atoms,device=device).unsqueeze(0) < inputs["_n_atoms"].unsqueeze(1)

        h = h.clone()  # (b,n,e)
        h = self.final_ln_node(h)  # (b,n,e)
        mask = mask.to(self.dtype).unsqueeze(-1)

        if self.target == Props.dipole and self.dipole_compose_from_charges:
            positions = inputs[Props.positions]
            h = h * mask
            charges = self.mlp(h)
            dipole = th.sum(charges * positions, dim=1)
            return dipole

        if self.target_type == PropertyType.mol_wise:
            h = (
                h[:, 0, :]  # (b,e)
                if self.cls_token
                else (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # (b,e)
            )
        elif self.target_type == PropertyType.atom_wise:
            if self.cls_token:
                h = h[:, 1:, :]  # (b,n-1,e)
            h = h * mask

        return self.mlp(h)


# fmt: off
# up until Z = 100; vs = valence s, vp = valence p, vd = valence d, vf = valence f.
# electron configuration follows the Aufbauprinzip. Exceptions are in the Lanthanides and Actinides (5f and 6d subshells are energetically very close).
electron_config = th.tensor([
  #  Z 1s 2s 2p 3s 3p 4s  3d 4p 5s  4d 5p 6s  4f  5d 6p 7s 5f 6d   vs vp  vd  vf
  [  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  0, 0,  0,  0], # n
  [  1, 1, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # H
  [  2, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # He
  [  3, 2, 1, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # Li
  [  4, 2, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Be
  [  5, 2, 2, 1, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 1,  0,  0], # B
  [  6, 2, 2, 2, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 2,  0,  0], # C
  [  7, 2, 2, 3, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 3,  0,  0], # N
  [  8, 2, 2, 4, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 4,  0,  0], # O
  [  9, 2, 2, 5, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 5,  0,  0], # F
  [ 10, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 6,  0,  0], # Ne
  [ 11, 2, 2, 6, 1, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # Na
  [ 12, 2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Mg
  [ 13, 2, 2, 6, 2, 1, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 1,  0,  0], # Al
  [ 14, 2, 2, 6, 2, 2, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 2,  0,  0], # Si
  [ 15, 2, 2, 6, 2, 3, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 3,  0,  0], # P
  [ 16, 2, 2, 6, 2, 4, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 4,  0,  0], # S
  [ 17, 2, 2, 6, 2, 5, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 5,  0,  0], # Cl
  [ 18, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 6,  0,  0], # Ar
  [ 19, 2, 2, 6, 2, 6, 1,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # K
  [ 20, 2, 2, 6, 2, 6, 2,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Ca
  [ 21, 2, 2, 6, 2, 6, 2,  1, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  1,  0], # Sc
  [ 22, 2, 2, 6, 2, 6, 2,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  2,  0], # Ti
  [ 23, 2, 2, 6, 2, 6, 2,  3, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  3,  0], # V
  [ 24, 2, 2, 6, 2, 6, 1,  5, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  5,  0], # Cr
  [ 25, 2, 2, 6, 2, 6, 2,  5, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  5,  0], # Mn
  [ 26, 2, 2, 6, 2, 6, 2,  6, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  6,  0], # Fe
  [ 27, 2, 2, 6, 2, 6, 2,  7, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  7,  0], # Co
  [ 28, 2, 2, 6, 2, 6, 2,  8, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  8,  0], # Ni
  [ 29, 2, 2, 6, 2, 6, 1, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0, 10,  0], # Cu
  [ 30, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0, 10,  0], # Zn
  [ 31, 2, 2, 6, 2, 6, 2, 10, 1, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 1, 10,  0], # Ga
  [ 32, 2, 2, 6, 2, 6, 2, 10, 2, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 2, 10,  0], # Ge
  [ 33, 2, 2, 6, 2, 6, 2, 10, 3, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 3, 10,  0], # As
  [ 34, 2, 2, 6, 2, 6, 2, 10, 4, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 4, 10,  0], # Se
  [ 35, 2, 2, 6, 2, 6, 2, 10, 5, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 5, 10,  0], # Br
  [ 36, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 6, 10,  0], # Kr
  [ 37, 2, 2, 6, 2, 6, 2, 10, 6, 1,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # Rb
  [ 38, 2, 2, 6, 2, 6, 2, 10, 6, 2,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Sr
  [ 39, 2, 2, 6, 2, 6, 2, 10, 6, 2,  1, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  1,  0], # Y
  [ 40, 2, 2, 6, 2, 6, 2, 10, 6, 2,  2, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  2,  0], # Zr
  [ 41, 2, 2, 6, 2, 6, 2, 10, 6, 1,  4, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  4,  0], # Nb
  [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 1,  5, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  5,  0], # Mo
  [ 43, 2, 2, 6, 2, 6, 2, 10, 6, 2,  5, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  5,  0], # Tc
  [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 1,  7, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  7,  0], # Ru
  [ 45, 2, 2, 6, 2, 6, 2, 10, 6, 1,  8, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  8,  0], # Rh
  [ 46, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 0,  0, 0, 10,  0], # Pd
  [ 47, 2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0, 10,  0], # Ag
  [ 48, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0, 10,  0], # Cd
  [ 49, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0,  0,  0, 0, 0, 0, 0,  2, 1, 10,  0], # In
  [ 50, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0,  0,  0, 0, 0, 0, 0,  2, 2, 10,  0], # Sn
  [ 51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0,  0,  0, 0, 0, 0, 0,  2, 3, 10,  0], # Sb
  [ 52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0,  0,  0, 0, 0, 0, 0,  2, 4, 10,  0], # Te
  [ 53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0,  0,  0, 0, 0, 0, 0,  2, 5, 10,  0], # I
  [ 54, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 0, 0, 0,  2, 6, 10,  0], # Xe
  [ 55, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # Cs
  [ 56, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Ba
  [ 57, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  1, 0, 0, 0, 0,  2, 0,  1,  0], # La
  [ 58, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  1,  1, 0, 0, 0, 0,  2, 0,  1,  1], # Ce
  [ 59, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  3,  0, 0, 0, 0, 0,  2, 0,  0,  3], # Pr
  [ 60, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  4,  0, 0, 0, 0, 0,  2, 0,  0,  4], # Nd
  [ 61, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  5,  0, 0, 0, 0, 0,  2, 0,  0,  5], # Pm
  [ 62, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  6,  0, 0, 0, 0, 0,  2, 0,  0,  6], # Sm
  [ 63, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  7,  0, 0, 0, 0, 0,  2, 0,  0,  7], # Eu
  [ 64, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  7,  1, 0, 0, 0, 0,  2, 0,  1,  7], # Gd
  [ 65, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  9,  0, 0, 0, 0, 0,  2, 0,  0,  9], # Tb
  [ 66, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10,  0, 0, 0, 0, 0,  2, 0,  0, 10], # Dy
  [ 67, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11,  0, 0, 0, 0, 0,  2, 0,  0, 11], # Ho
  [ 68, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12,  0, 0, 0, 0, 0,  2, 0,  0, 12], # Er
  [ 69, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13,  0, 0, 0, 0, 0,  2, 0,  0, 13], # Tm
  [ 70, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  0, 0, 0, 0, 0,  2, 0,  0, 14], # Yb
  [ 71, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  1, 0, 0, 0, 0,  2, 0,  1, 14], # Lu
  [ 72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  2, 0, 0, 0, 0,  2, 0,  2, 14], # Hf
  [ 73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  3, 0, 0, 0, 0,  2, 0,  3, 14], # Ta
  [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  4, 0, 0, 0, 0,  2, 0,  4, 14], # W
  [ 75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  5, 0, 0, 0, 0,  2, 0,  5, 14], # Re
  [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  6, 0, 0, 0, 0,  2, 0,  6, 14], # Os
  [ 77, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  7, 0, 0, 0, 0,  2, 0,  7, 14], # Ir
  [ 78, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14,  9, 0, 0, 0, 0,  1, 0,  9, 14], # Pt
  [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 0, 0, 0,  1, 0, 10, 14], # Au
  [ 80, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 0, 0, 0,  2, 0, 10, 14], # Hg
  [ 81, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 0, 0, 0,  2, 1, 10, 14], # Tl
  [ 82, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 0, 0, 0,  2, 2, 10, 14], # Pb
  [ 83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 0, 0, 0,  2, 3, 10, 14], # Bi
  [ 84, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4, 0, 0, 0,  2, 4, 10, 14], # Po
  [ 85, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5, 0, 0, 0,  2, 5, 10, 14], # At
  [ 86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 0, 0, 0,  2, 6, 10, 14], # Rn
  [ 87, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 1, 0, 0,  1, 0,  0,  0], # Fr
  [ 88, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 0,  2, 0,  0,  0], # Ra
  [ 89, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 1,  2, 0,  1,  0], # Ac
  [ 90, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 2,  2, 0,  2,  0], # Th
  [ 91, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 2, 1,  2, 0,  1,  2], # Pa
  [ 92, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 3, 1,  2, 0,  3,  1], # U
  [ 93, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 4, 1,  2, 0,  1,  4], # Np
  [ 94, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 0,  2, 0,  0,  6], # Pu
  [ 95, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 0,  2, 0,  0,  7], # Am
  [ 96, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 1,  2, 0,  1,  7], # Cm
  [ 97, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 9, 0,  2, 0,  0,  9], # Bk
  [ 98, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 10,0,  2, 0,  0, 10], # Cf
  [ 99, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 11,0,  2, 0,  0, 11], # Es
  [100, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 12,0,  2, 0,  0, 12],  # Fm
  [101, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  0, 0,  0,  0]  # Md
], dtype=th.float64)
# fmt: on
electron_config = electron_config / th.max(electron_config, axis=0).values
MAX_Z = 101


class NuclearEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        use_electronic_embeddings: bool,
        zero_init=True,
        cls_token=True,
        dtype: th.dtype | None = None,
    ) -> None:
        super().__init__()
        self.cls_token = cls_token
        self.embedding = nn.Embedding(MAX_Z + 1, embedding_dim)
        self.dtype = set_dtype(dtype)
        self.use_electronic_embeddings = use_electronic_embeddings
        if self.use_electronic_embeddings:
            self.register_buffer("electron_config", electron_config.clone().detach().to(self.dtype))
            self.config_linear = nn.Linear(electron_config.shape[1], embedding_dim)
        self.reset_parameters(zero_init)

    def reset_parameters(self, zero_init=True) -> None:
        if zero_init:
            nn.init.zeros_(self.embedding.weight)
            if self.use_electronic_embeddings:
                nn.init.zeros_(self.config_linear.weight)
        else:
            nn.init.uniform_(self.embedding, -math.sqrt(3), math.sqrt(3))
            if self.use_electronic_embeddings:
                nn.init.orthogonal_(self.config_linear.weight)

    def forward(self, atomic_numbers) -> th.Tensor:
        embedding = self.embedding(atomic_numbers)  # (B, N, embedding_dim)
        if self.use_electronic_embeddings:
            electronic_embedding = self.config_linear(
                self.electron_config[atomic_numbers.long()]
            )  # (B, N, embedding_dim)
            if self.cls_token:
                electronic_embedding[:, 0, :] = 0.0
            embedding += electronic_embedding
        return embedding


class FourierDirectionalEmbed(nn.Module):
    def __init__(self, num_heads, num_kernel) -> None:
        assert num_kernel % 2 == 0

        super().__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel
        min_angle = 0.0001
        max_azimuthal = 2 * math.pi
        max_polar = math.pi

        wave_lengths_azimuthal = th.exp(
            th.linspace(math.log(2 * min_angle), math.log(2 * max_azimuthal), num_kernel // 2)
        )
        wave_lengths_polar = th.exp(th.linspace(math.log(2 * min_angle), math.log(2 * max_polar), num_kernel // 2))
        angular_freqs_azimuthal = 2 * math.pi / wave_lengths_azimuthal
        angular_freqs_polar = 2 * math.pi / wave_lengths_polar
        self.register_buffer("angular_freqs_azimuthal", angular_freqs_azimuthal)
        self.register_buffer("angular_freqs_polar", angular_freqs_polar)

        self.proj = nn.Linear(num_kernel * 2, num_heads)

    def forward(self, direction) -> th.Tensor:
        azimuthal, polar = direction.unbind(-1)
        phase_azimuthal = azimuthal.unsqueeze(-1) * self.angular_freqs_azimuthal
        phase_polar = polar.unsqueeze(-1) * self.angular_freqs_polar
        sinusoids = th.cat(
            [
                th.sin(phase_azimuthal),
                th.cos(phase_azimuthal),
                th.sin(phase_polar),
                th.cos(phase_polar),
            ],
            dim=-1,
        )
        out = self.proj(sinusoids)
        return out


class NonLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden=None) -> None:
        super().__init__()
        if hidden is None:
            hidden = input_size
        self.layer1 = nn.Linear(input_size, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x) -> th.Tensor:
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class Gaussian3DEmbed(nn.Module):
    def __init__(self, num_heads: int, num_edges: int, num_kernel: int) -> None:
        super().__init__()
        self.gbf = GaussianLayer(num_kernel, num_edges)
        self.gbf_proj = NonLinear(num_kernel, num_heads)

    def forward(self, dist, node_type_edge) -> th.Tensor:
        edge_feature = self.gbf(dist, node_type_edge.long())  # (b, n, n, K)
        gbf_result = self.gbf_proj(edge_feature)  # (b, n, n, H)
        return gbf_result


@th.jit.script
def gaussian(x, mean, std) -> th.Tensor:
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return th.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3, dtype: th.dtype | None = None) -> None:
        super().__init__()
        self.K = K
        self.dtype = set_dtype(dtype)
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 7)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types) -> th.Tensor:
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.to(self.dtype).view(-1)
        std = self.stds.weight.to(self.dtype).view(-1).abs() + 1e-2
        return gaussian(x.to(self.dtype), mean, std).type_as(self.means.weight)
