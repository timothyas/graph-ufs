import os
import xarray as xr
from jax import tree_util, numpy as jnp
import numpy as np

from graphufs.gefs import GEFSEmulator as BaseGEFSEmulator


local_path = os.path.dirname(os.path.realpath(__file__))

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

class GEFSEmulator(BaseGEFSEmulator):

    data_url = "/home/tsmith/work/ufs2arco/examples/gefs/sample-gefs.zarr"
    norm_urls = {
        "mean": f"{local_path}/fv3.fvstatistics.1994/mean_by_level.zarr",
        "std": f"{local_path}/fv3.fvstatistics.1994/stddev_by_level.zarr",
        "stddiff": f"{local_path}/fv3.fvstatistics.1994/diffs_stddev_by_level.zarr",
    }

    local_store_path = "./dp0"

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        # 3D Variables
        "u",
        "v",
        "w",
        "t",
        "q",
        # Surface Variables
        "u10",
        "v10",
        "t2m",
        "sh2",
        "sp",
        # Forcing Variables at Input Time
        "toa_incident_solar_radiation",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
        # Static Variables
        "lsm",
        "orog",
    )
    target_variables = (
        # 3D Variables
        "u",
        "v",
        "w",
        "t",
        "q",
        # Surface Variables
        "u10",
        "v10",
        "t2m",
        "sh2",
        "sp",
    )
    forcing_variables = (
        "toa_incident_solar_radiation",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    pressure_levels = (500, 800, 1000)

    # time related
    delta_t = "6h"              # the model time step
    input_duration = "6h"      # time covered by initial condition(s) + delta_t (necessary for GraphCast code)
    target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets
    training_dates = (          # bounds of training data (inclusive)
        "2017-01-01T00",        # start
        "2017-01-06T18"         # stop
    )
    testing_dates = (           # bounds of testing data (inclusive)
        "2017-01-01T00",        # start
        "2017-01-05T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "2017-01-01T00",        # start
        "2017-01-05T18"         # stop
    )

    # training protocol
    batch_size = 16
    num_epochs = 10

    # model config options
    resolution = 1.0
    mesh_size = 2
    latent_size = 256
    gnn_msg_steps = 4
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_latitude = True
    weight_loss_per_level = False
    weight_loss_per_channel = True
    loss_weights_per_variable = dict()
    input_transforms = {
        "q": log,
        "sh2": log,
    }
    output_transforms = {
        "q": exp,
        "sh2": exp,
    }

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # data chunking options
    max_queue_size = 1
    num_workers = 1
    dask_threads = 8
    num_gpus = 1

tree_util.register_pytree_node(
    GEFSEmulator,
    GEFSEmulator._tree_flatten,
    GEFSEmulator._tree_unflatten
)
