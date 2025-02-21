import os
import xarray as xr
from jax import tree_util, numpy as jnp
import numpy as np

from graphufs.gefs import GEFSEmulator as BaseGEFSEmulator

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

_scratch = "/pscratch/sd/t/timothys/gefs/one-degree"

class GEFSEmulator(BaseGEFSEmulator):

    data_url = f"{_scratch}/forecasts.zarr"
    norm_urls = {
        "mean": f"{_scratch}/statistics/mean_by_level.zarr",
        "std": f"{_scratch}/statistics/stddev_by_level.zarr",
        "stddiff": f"{_scratch}/statistics/diffs_stddev_by_level.zarr",
    }

    local_store_path = None

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
    pressure_levels = (100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

    # time related
    delta_t = "6h"
    input_duration = "6h"
    target_lead_time = "6h"
    training_dates = (
        "2017-01-01T00",
        "2019-06-30T18"
    )
    validation_dates = (
        "2019-07-01T00",
        "2019-12-31T18"
    )
    testing_dates = (
        "2020-01-01T00",
        "2020-09-23T06"
    )

    # training protocol
    batch_size = 16
    num_epochs = 10

    # model config options
    resolution = 1.0
    mesh_size = 5
    latent_size = 512
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6

    # loss weighting
    weight_loss_per_channel = True
    weight_loss_per_latitude = True
    weight_loss_per_level = False
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

    # data loading options
    max_queue_size = 1
    num_workers = 1

tree_util.register_pytree_node(
    GEFSEmulator,
    GEFSEmulator._tree_flatten,
    GEFSEmulator._tree_unflatten
)
