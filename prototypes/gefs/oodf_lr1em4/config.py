from jax import tree_util
from prototypes.gefs.config import BaseGEFSEmulator, _scratch

class GEFSForecastTrainer(BaseGEFSEmulator):

    local_store_path = f"{_scratch}/graph-ufs/gefs/oodf_lr1em4"
    num_epochs = 64
    peak_lr = 1e-4

class GEFSForecastPreprocessor(GEFSForecastTrainer):

    batch_size = 64

class GEFSForecastPreprocessed(GEFSForecastTrainer):
    input_transforms = None
    output_transforms = None

class GEFSForecastEvaluator(GEFSForecastTrainer):

    data_url = f"{_scratch}/gefs/one-degree/forecasts.validation.zarr"
    wb2_obs_url = "gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    #evaluation_checkpoint_id = 64
    evaluation_checkpoint_id = 18
    batch_size = 32

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
    GEFSForecastEvaluator,
    GEFSForecastEvaluator._tree_flatten,
    GEFSForecastEvaluator._tree_unflatten
)
