from jax import tree_util

from graphufs import ReplayEmulator

class P1Emulator(ReplayEmulator):

    data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/mean_by_level.p0.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/stddev_by_level.p0.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/diffs_stddev_by_level.p0.zarr",
    }
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
    local_store_path = "./data"
    no_cache_data = False        # don't cache or use zarr dataset downloaded from GCS on disk

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        "delz", # height thickness (as a replacement for geopotential)
        "hgtsfc", # geopotential at surface
        "land", # land-sea mask
        "tmp2m", # 2m temperature
        "pressfc", # mean sea level pressure
        "ugrd10m", # 10m u component of wind
        "vgrd10m", # 10m v component of wind
        "prateb_ave",  # total precipitation 3hr
        "dswrf_avetoa", # toa incident solar radiation
        "tmp", # temperature
        "ugrd", # u component of wind
        "vgrd", # v component of wind
        "dzdt", # vertical velocity
        "spfh", # specific humidity
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    target_variables = (
        "delz",
        "tmp2m",
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "prateb_ave",
        "tmp",
        "ugrd",
        "vgrd",
        "dzdt",
        "spfh",
    )
    forcing_variables = (
        "dswrf_avetoa", # toa incident solar radiation
        "land", # land-sea-ice mask
        "hgtsfc", # geopotential at the surface
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    
    all_variables = tuple() # this is created in __init__

    pressure_levels = (
        50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

    # time related
    delta_t = "3h"              # the model time step
    input_duration = "6h"    # time covered by initial condition(s), note the 1s is necessary for GraphCast code
    target_lead_time = "3h"     # how long is the forecast ... at what point do we compare model to targets
    training_dates = (          # bounds of training data (inclusive)
        "1993-12-31T18",        # start
        "1994-12-31T18"         # stop, includes all of 1994
    )

    # training protocol
    batch_size = 32

    # model config options
    resolution = 1.0            # nominal spatial resolution
    
    mesh_size = 5               # how many refinements to do on the multi-mesh
    
    latent_size = 512           # how many latent features to include in various MLPs
    
    gnn_msg_steps = 16          # how many graph network message passing steps to do
    
    hidden_layers = 1           # number of hidden layers for each MLP

    radius_query_fraction_edge_length = 0.6  # Scalar that will be multiplied by the length of the longest edge of 
                                             # the finest mesh to define the radius of connectivity to use in the 
                                             # Grid2Mesh graph. Reasonable values are between 0.6 and 1. 0.6 reduces 
                                             # the number of grid points feeding into multiple mesh nodes and therefore 
                                             # reduces edge count and memory use, but gives better predictions.
    
    mesh2grid_edge_normalization_factor = 0.6180338738074472 # Allows explicitly controlling edge normalization for mesh2grid edges. 
                                                             # If None, defaults to max edge length.This supports using pre-trained 
                                                             # model weights with a different graph structure to what it was trained on. 

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100
    
    # data chunking options
    chunks_per_epoch = 26      # 1 chunk per year
    steps_per_chunk = None
    checkpoint_chunks = 1

    # others
    num_gpus = 1
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = False

tree_util.register_pytree_node(
    P1Emulator,
    P1Emulator._tree_flatten,
    P1Emulator._tree_unflatten
)
