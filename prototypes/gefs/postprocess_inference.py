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
    gds = xr.open_zarr(f"{emulator.inference_directory}/validation/graphufs.{duration}.zarr")

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

    do_mean = "member" in xds.dims
    if do_mean:
        logging.info(f"Found 'member' in dimensions, will compute ensemble mean")

    logging.info(f"Selecting pressure levels {plevels} hPa...")
    pds = xds.sel(level=list(plevels))

    if do_mean:
        mds = pds.mean("member")

    # regrid and rename variables
    pds = regrid_and_rename(pds, truth, is_gaussian=False)
    if do_mean:
        mds = regrid_and_rename(mds, truth, is_gaussian=False)
    logging.info(f"Done forming regridding operations...")

    duration = emulator.target_lead_time[-1]
    path = f"{emulator.inference_directory}/validation/{name}.{duration}.postprocessed.zarr"
    pds.to_zarr(path, mode="w")
    logging.info(f"Done writing to {path}")
    if do_mean:
        mpath = f"{emulator.inference_directory}/validation/{name}-mean.{duration}.postprocessed.zarr"
        mds.to_zarr(path, mode="w")
        logging.info(f"Done writing to {mpath}")


def compute_ensemble_mean(open_path, store_path, load_each_variable=True):

    logging.info(f"Computing ensemble mean")
    logging.info(f"\tReading from: {open_path}")
    logging.info(f"\tStoring to: {store_path}")

    xds = xr.open_zarr(open_path)

    for key in xds.data_vars:
        mds = xds[[key]]
        with xr.set_options(keep_attrs=True):
            mds = mds.mean("member") if "member" in mds.dims else mds
        logging.info(f"\tLoading {key}")
        mds.load()

        logging.info(f"\t... chunking")
        chunks = {k: val for k, val in xds[key].encoding["preferred_chunks"].items() if k in mds[key].dims}
        mds[key] = mds[key].chunk(chunks)
        mds[key].encoding["preferred_chunks"] = chunks
        mds[key].encoding["chunks"] = tuple(chunks.values())

        mds.to_zarr(
            store_path,
            mode="a",
        )
        logging.info(f"\t... done with {key}")


def main(Emulator):

    setup_simple_log()
    emulator = Emulator()
    duration = emulator.target_lead_time[-1]
    ckpt_id = emulator.evaluation_checkpoint_id if emulator.evaluation_checkpoint_id is not None else emulator.num_epochs

    logging.info(f"Postprocessing inference from checkpoint_id = {ckpt_id}")
    dask.config.set(scheduler="threads", num_workers=64)

    compute_ensemble_mean(
        f"{emulator.inference_directory}/validation/graphufs.{duration}.zarr",
        f"{emulator.inference_directory}/validation/graphufs.ensemble-mean.{duration}.zarr",
    )

    # The workflow below works if regridding is necessary
    # but since we can compare directly to a 1 degree ERA5, why do this?
    # Just compute ensemble mean (above)
    #logging.info("Opening Predictions")
    #gds, truth = open_predictions_and_truth(emulator)

    #logging.info("Postprocessing Predictions")
    #postproc(emulator, gds, truth, name="graphufs")
    #logging.info("Done postprocessing predictions")

    #logging.info("Opening GEFS")
    #rds = open_targets(emulator, gds, truth)

    #logging.info("Postprocessing GEFS")
    #postproc(emulator, rds, truth, name="gefs")

    logging.info("Done with all postprocessing")
