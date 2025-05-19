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
from graphufs.fvemulator import fv_vertical_regrid, get_new_vertical_grid

def open_predictions_and_truth(emulator):

    duration = emulator.target_lead_time[-1]

    # open graphufs, and targets as is
    gds = xr.open_zarr(f"{emulator.local_store_path}/inference/validation/graphufs.{duration}.zarr")

    # add vertical coordinate stuff
    if "ak" not in gds and "bk" not in gds:
        cds = get_new_vertical_grid(list(emulator.interfaces))
        gds = gds.assign_coords({"ak": cds["ak"], "bk": cds["bk"]})

    # add static hgtsfc for geopotential
    if "hgtsfc_static" not in gds:
        gds["hgtsfc_static"] = emulator.open_dataset()["hgtsfc_static"].rename({
            "grid_yt": "lat",
            "grid_xt": "lon",
        }).load()

    # for 3h dt, subsample to 6h
    if duration == "240h" and len(gds.lead_time) == 80:
        gds = gds.isel(lead_time=slice(1, None, 2))

        logging.info("Subsampled lead_time to 6h")
        logging.info(f"New time: {gds.lead_time.values}")

    truth = xr.open_zarr(emulator.wb2_obs_url, storage_options={"token":"anon"})

    # subsample in space to avoid poles
    if "with_poles" in emulator.wb2_obs_url:
        truth = truth.sel(latitude=slice(-89, 89))

    # subsample in time, based on truth we can compare to
    t0 = get_valid_initial_conditions(gds, truth)
    gds = gds.sel(time=t0)
    return gds, truth

def open_targets(emulator, predictions, truth):
    rds = emulator.open_dataset()
    keep_vars = list(predictions.keys()) + ["geopotential", "delz"]
    rds = rds[[x for x in keep_vars if x in rds]]
    valid_time = predictions["time"] + predictions["lead_time"]
    rds = rds.sel(time=slice(predictions.time.values[0], valid_time.isel(time=-1, lead_time=-1).values))
    time = get_valid_initial_conditions(rds, truth)
    rds = rds.sel(time=time)

    # Now FV vertical
    rds = fv_vertical_regrid(rds, interfaces=list(emulator.interfaces), keep_delz=True)
    rds = rds.rename({"pfull": "level", "grid_xt": "lon", "grid_yt": "lat"})

    # compute some diagnostics...
    diagnostic_mappings = dict()
    if emulator.diagnostics is not None:
        diagnostic_mappings = diagnostics.prepare_diagnostic_functions(emulator.diagnostics)

        for key, func in diagnostic_mappings["functions"].items():
            logging.info(f"{__name__}.open_targets: computing diagnostic {key}")
            rds[key] = func(rds)

    # Store this as is
    replay_path = f"{emulator.local_store_path}/inference/validation/replay.vertical_regrid.zarr"
    logging.info(f"Writing vertically regridded Replay to {replay_path} ...")
    rds.to_zarr(replay_path)
    logging.info(f"Done writing Replay")
    return rds

def postproc(emulator, xds, truth, name, plevels=(250, 500, 850)):

    pds = interp2pressure(
        xds,
        plevels,
        diagnose_geopotential= name!="replay" and "hydrostatic_geopotential" not in xds,
    )
    logging.info(f"Done forming interpolation operations...")

    # regrid and rename variables
    pds = regrid_and_rename(pds, truth)
    logging.info(f"Done forming regridding operations...")

    duration = emulator.target_lead_time[-1]
    path = f"{emulator.local_store_path}/inference/validation/{name}.{duration}.postprocessed.zarr"
    pds.to_zarr(path)
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

    logging.info("Opening Replay")
    rds = open_targets(emulator, gds, truth)

    logging.info("Postprocessing Replay")
    postproc(emulator, rds, truth, name="replay")

    logging.info("Done with all postprocessing")
