from jax import tree_util
from jax import numpy as jnp
from graphufs.gefs import GEFSDeviationEmulator
from prototypes.gefs.config import BaseGEFSEmulator, _scratch



class GEFSMSETrainer(BaseGEFSEmulator):

    local_store_path = f"{_scratch}/graph-ufs/gefs/OnlyOneDegree/forecast-training"

class GEFSMSEPreprocessor(GEFSMSETrainer):

    batch_size = 64

class GEFSDeviationTrainer(GEFSDeviationEmulator, BaseGEFSEmulator):

    local_store_path = f"{_scratch}/graph-ufs/gefs/OnlyOneDegree/deviation-training"
    num_epochs = 28

class GEFSEvaluator(GEFSMSETrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(3, 3*8*10+1, 3)]
    sample_stride = 9
    evaluation_checkpoint_id = 64
    batch_size = 32


tree_util.register_pytree_node(
    GEFSMSEPreprocessor,
    GEFSMSEPreprocessor._tree_flatten,
    GEFSMSEPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSMSETrainer,
    GEFSMSETrainer._tree_flatten,
    GEFSMSETrainer._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSDeviationTrainer,
    GEFSDeviationTrainer._tree_flatten,
    GEFSDeviationTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSEvaluator,
    GEFSEvaluator._tree_flatten,
    GEFSEvaluator._tree_unflatten
)
