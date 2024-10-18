import logging
import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr
import pandas as pd

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes
from ufs2arco import Layers2Pressure

from graphufs.log import setup_simple_log
from graphufs.postprocess import interp2pressure, regrid_and_rename, get_valid_initial_conditions
from graphufs.fvemulator import fv_vertical_regrid

from config import P2EvaluationEmulator as Emulator

def open_datasets(emulator):
    """
    Returns:
        predictions, targets, original_target_dataset, independent_truth_dataset
    """

    duration = emulator.target_lead_time[-1]

    # open graphufs, and targets as is
    gds = xr.open_zarr(f"{emulator.local_store_path}/inference/validation/graphufs.{duration}.zarr")
    truth = xr.open_zarr(emulator.wb2_obs_url, storage_options={"token":"anon"})

    # TODO: remove this
    gds = gds.isel(time=slice(80))

    nds = xr.open_zarr(emulator.norm_urls["mean"], storage_options={"token": "anon"})
    gds["ak"] = nds["ak"]
    gds["bk"] = nds["bk"]
    gds = gds.set_coords(["ak", "bk"])

    if "hgtsfc_static" not in gds:
        gds["hgtsfc_static"] = emulator.open_dataset()["hgtsfc_static"].rename({
            "grid_yt": "lat",
            "grid_xt": "lon",
        })

    # for 3h dt, subsample to 6h
    if duration == "240h" and len(gds.lead_time) == 80:
        gds = gds.isel(lead_time=slice(1, None, 2))

        logging.info("Subsampled lead_time to 6h")
        logging.info(f"New time: {gds.lead_time.values}")

    # subsample in space to avoid poles
    if "with_poles" in emulator.wb2_obs_url:
        truth = truth.sel(latitude=slice(-89, 89))

    # subsample in time, based on truth we can compare to
    t0 = get_valid_initial_conditions(gds, truth)
    gds = gds.sel(time=t0)

    # for replay we also want to get the original data, in order to avoid
    # excessive interpolation error due to subsampled vertical levels
    rds = emulator.open_dataset()
    keep_vars = list(gds.keys()) + ["geopotential", "delz"]
    rds = rds[keep_vars]
    valid_time = gds["time"] + gds["lead_time"]
    rds = rds.sel(time=slice(gds.time.values[0], valid_time.isel(time=-1, lead_time=-1).values))
    time = get_valid_initial_conditions(rds, truth)
    rds = rds.sel(time=time)

    # Now FV vertical
    rds = fv_vertical_regrid(rds, interfaces=list(emulator.interfaces))
    rds = rds.rename({"pfull": "level", "grid_xt": "lon", "grid_yt": "lat"})
    return gds, rds, truth


if __name__ == "__main__":

    setup_simple_log()
    emulator = Emulator()
    dask.config.set(scheduler="threads", num_workers=32)

    duration = emulator.target_lead_time[-1]
    gds, rds, truth = open_datasets(emulator)

    for ds, name in zip(
        [gds, rds],
        ["graphufs", "replay"],
    ):
        ds = interp2pressure(ds, [250, 500, 850], diagnose_geopotential=name!="replay")
        logging.info(f"Done forming interpolation operations...")

        # regrid and rename variables
        ds = regrid_and_rename(ds, truth)
        logging.info(f"Done forming regridding operations...")

        path = f"{emulator.local_store_path}/inference/validation/{name}.{duration}.postprocessed.zarr"
        ds.to_zarr(path)
        logging.info(f"Done writing to {path}")
