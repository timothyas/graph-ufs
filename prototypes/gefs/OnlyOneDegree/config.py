from jax import tree_util
from jax import numpy as jnp
from prototypes.gefs.config import BaseGEFSEmulator, _scratch

class GEFSTrainer(BaseGEFSEmulator):

    local_store_path = f"{_scratch}/graph-ufs/gefs/OnlyOneDegree"

class GEFSPreprocessor(GEFSTrainer):

    batch_size = 64

class GEFSPreprocessed(GEFSTrainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.

    Note that it is a happy coincidence that we needed to create a separate compilable
    input/output transform to satisfy jax, since we will still need those transforms (but not the xarray ones) here.
    """
    input_transforms = None
    output_transforms = None


class GEFSEvaluator(GEFSTrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(3, 3*8*10+1, 3)]
    sample_stride = 9
    evaluation_checkpoint_id = 64
    batch_size = 32


tree_util.register_pytree_node(
    GEFSTrainer,
    GEFSTrainer._tree_flatten,
    GEFSTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSPreprocessor,
    GEFSPreprocessor._tree_flatten,
    GEFSPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSPreprocessed,
    GEFSPreprocessed._tree_flatten,
    GEFSPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSEvaluator,
    GEFSEvaluator._tree_flatten,
    GEFSEvaluator._tree_unflatten
)
