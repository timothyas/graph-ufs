import xarray as xr
from jax import tree_util
import numpy as np

from graphufs import FVEmulator

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

class P2Emulator(FVEmulator):

    # paths
    data_url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "/p2-lustre/fv3.fvstatistics.1993-2019/mean_by_level.zarr",
        "std": "/p2-lustre/fv3.fvstatistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "/p2-lustre/fv3.fvstatistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    local_store_path = "/p2-lustre/p2"

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        # 3D Variables
        "ugrd",
        "vgrd",
        "dzdt",
        "tmp",
        "spfh",
        # Surface Variables
        "ugrd10m",
        "vgrd10m",
        "tmp2m",
        "spfh2m",
        "pressfc",
        # Forcing Variables at Input Time
        "toa_incident_solar_radiation",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
        # Static Variables
        "land_static",
        "hgtsfc_static",
    )
    target_variables = (
        # 3D Variables
        "ugrd",
        "vgrd",
        "dzdt",
        "tmp",
        "spfh",
        # Surface Variables
        "ugrd10m",
        "vgrd10m",
        "tmp2m",
        "spfh2m",
        "pressfc",
    )
    forcing_variables = (
        "toa_incident_solar_radiation",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )

    # vertical grid
    interfaces = tuple(x for x in range(200, 1001, 50))

    # time related
    delta_t = "3h"              # the model time step
    input_duration = "6h"      # time covered by initial condition(s) + delta_t (necessary for GraphCast code)
    target_lead_time = "3h"     # how long is the forecast ... at what point do we compare model to targets
    training_dates = (          # bounds of training data (inclusive)
        "1994-01-01T00",        # start
        "2019-12-31T18"         # stop
    )
    testing_dates = (        # bounds of testing data (inclusive)
        "2020-01-01T00",        # start
        "2021-12-31T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "2022-01-01T00",        # start
        "2023-10-13T03"         # stop
    )

    # training protocol
    batch_size = 16
    num_epochs = 50

    # model config options
    resolution = 1.0
    mesh_size = 5
    latent_size = 512
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_latitude = True
    weight_loss_per_level = False
    loss_weights_per_variable = dict() # weight them all equally
    input_transforms = {
        "spfh": log,
        "spfh2m": log,
    }
    output_transforms = {
        "spfh": exp,
        "spfh2m": exp,
    }

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # data loading options
    max_queue_size = 1
    num_workers = 1
    dask_threads = 16

    # hardware
    num_gpus = 1
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = False

tree_util.register_pytree_node(
    P2Emulator,
    P2Emulator._tree_flatten,
    P2Emulator._tree_unflatten
)
