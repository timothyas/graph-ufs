from jax import tree_util
from graphufs.gefs import GEFSDeviationEmulator
from prototypes.gefs.config import BaseGEFSEmulator, _scratch

class GEFSDeviationTrainer(BaseGEFSEmulator, GEFSDeviationEmulator):

    local_store_path = f"{_scratch}/graph-ufs/gefs/oodd_lr1em4"
    num_epochs = 32
    peak_lr = 1e-4

class GEFSDeviationPreprocessor(GEFSDeviationTrainer):

    batch_size = 64

class GEFSDeviationPreprocessed(GEFSDeviationTrainer):
    input_transforms = None
    output_transforms = None

class GEFSDeviationEvaluator(BaseGEFSEmulator):
    """Note that this one we want to inherit from the regular class,
    so that Dataset samples are built as we would expect...
    That may change if we want to look at evaluating deviations, but later...
    """

    local_store_path = f"{_scratch}/graph-ufs/gefs/oodd_lr1em4"
    num_epochs = 32
    peak_lr = 1e-4

    data_url = f"{_scratch}/gefs/one-degree/forecasts.validation.zarr"
    wb2_obs_url = "gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    #evaluation_checkpoint_id = 32
    evaluation_checkpoint_id = 22
    batch_size = 32



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
    GEFSDeviationEvaluator,
    GEFSDeviationEvaluator._tree_flatten,
    GEFSDeviationEvaluator._tree_unflatten
)
