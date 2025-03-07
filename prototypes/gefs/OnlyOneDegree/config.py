from jax import tree_util
from jax import numpy as jnp
from graphufs.gefs import GEFSDeviationEmulator
from prototypes.gefs.config import BaseGEFSEmulator, _scratch



class GEFSForecastTrainer(BaseGEFSEmulator):

    local_store_path = f"{_scratch}/graph-ufs/gefs/OnlyOneDegree/forecast-training"
    num_epochs = 64
    peak_lr = 1e-3

class GEFSForecastPreprocessor(GEFSForecastTrainer):

    batch_size = 64

class GEFSForecastPreprocessed(GEFSForecastTrainer):
    input_transforms = None
    output_transforms = None

class GEFSDeviationTrainer(BaseGEFSEmulator, GEFSDeviationEmulator):

    local_store_path = f"{_scratch}/graph-ufs/gefs/OnlyOneDegree/deviation-training"
    num_epochs = 28
    peak_lr = 1e-4

class GEFSDeviationPreprocessor(GEFSDeviationTrainer):

    batch_size = 64

class GEFSDeviationPreprocessed(GEFSDeviationTrainer):
    input_transforms = None
    output_transforms = None

class GEFSForecastEvaluator(GEFSForecastTrainer):

    data_url = f"{_scratch}/gefs/one-degree/forecasts.validation.zarr"
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    evaluation_checkpoint_id = 64

class GEFSDeviationEvaluator(GEFSForecastTrainer):
    """Note that this one we want to inherit from the regular class,
    so that Dataset samples are built as we would expect...
    That may change if we want to look at evaluating deviations, but later...
    """

    data_url = f"{_scratch}/gefs/one-degree/forecasts.validation.zarr"
    local_store_path = f"{_scratch}/graph-ufs/gefs/OnlyOneDegree/deviation-training"
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    evaluation_checkpoint_id = 28


tree_util.register_pytree_node(
    GEFSForecastTrainer,
    GEFSForecastTrainer._tree_flatten,
    GEFSForecastTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSForecastPreprocessor,
    GEFSForecastPreprocessor._tree_flatten,
    GEFSForecastPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSForecastPreprocessed,
    GEFSForecastPreprocessed._tree_flatten,
    GEFSForecastPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSDeviationTrainer,
    GEFSDeviationTrainer._tree_flatten,
    GEFSDeviationTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSDeviationPreprocessor,
    GEFSDeviationPreprocessor._tree_flatten,
    GEFSDeviationPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSDeviationPreprocessed,
    GEFSDeviationPreprocessed._tree_flatten,
    GEFSDeviationPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSForecastEvaluator,
    GEFSForecastEvaluator._tree_flatten,
    GEFSForecastEvaluator._tree_unflatten
)

tree_util.register_pytree_node(
    GEFSDeviationEvaluator,
    GEFSDeviationEvaluator._tree_flatten,
    GEFSDeviationEvaluator._tree_unflatten
)
