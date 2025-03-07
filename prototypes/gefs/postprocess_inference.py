import logging
import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr
import pandas as pd

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes
from ufs2arco import Layers2Pressure

from graphufs import diagnostics
from graphufs.log import setup_simple_log
from graphufs.postprocess import interp2pressure, regrid_and_rename, get_valid_initial_conditions
from graphufs.fvemulator import fv_vertical_regrid

def open_predictions_and_truth(emulator):

    duration = emulator.target_lead_time[-1]

    # open graphufs, and targets as is
    gds = xr.open_zarr(f"{emulator.local_store_path}/inference/validation/graphufs.{duration}.zarr")
    gds = gds.isel(time=slice(4))

    truth = xr.open_zarr(emulator.wb2_obs_url, storage_options={"token":"anon"})

    # subsample in space to avoid poles
    if "with_poles" in emulator.wb2_obs_url:
        truth = truth.sel(latitude=slice(-89.5, 89.5))

    # subsample in time, based on truth we can compare to
    t0 = get_valid_initial_conditions(gds, truth)
    gds = gds.sel(time=t0)
    return gds, truth

def open_targets(emulator, predictions, truth):
    rds = emulator.open_dataset()
    rds = rds.rename({
        "t0": "time",
        "pressure": "level",
        "latitude": "lat",
        "longitude": "lon",
    })
    rds = rds.drop_vars("valid_time")
    rds = rds.swap_dims({"fhr": "lead_time"}).drop_vars("fhr")
    rds = rds.transpose("time", "member", "lead_time", "level", "lat", "lon")

    rds = rds.sel(time=predictions.time)
    rds = rds.sel(level=predictions.level)

    keep_vars = list(predictions.keys())
    rds = rds[[x for x in keep_vars if x in rds]]
    return rds

def postproc(emulator, xds, truth, name, plevels=(250, 500, 850)):

    logging.info(f"Selecting pressure levels {plevels} hPa...")
    pds = xds.sel(level=list(plevels))

    # regrid and rename variables
    pds = regrid_and_rename(pds, truth, is_gaussian=False)
    logging.info(f"Done forming regridding operations...")

    duration = emulator.target_lead_time[-1]
    path = f"{emulator.local_store_path}/inference/validation/{name}.{duration}.postprocessed.zarr"
    pds.to_zarr(path, mode="w")
    logging.info(f"Done writing to {path}")


def main(Emulator):

    setup_simple_log()
    emulator = Emulator()
    dask.config.set(scheduler="threads", num_workers=64)


    logging.info("Opening Predictions")
    gds, truth = open_predictions_and_truth(emulator)

    logging.info("Postprocessing Predictions")
    postproc(emulator, gds, truth, name="graphufs")
    logging.info("Done postprocessing predictions")

    logging.info("Opening GEFS")
    rds = open_targets(emulator, gds, truth)

    logging.info("Postprocessing GEFS")
    postproc(emulator, rds, truth, name="gefs")

    logging.info("Done with all postprocessing")
