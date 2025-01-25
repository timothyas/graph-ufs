import logging
import numpy as np
import dask
import xarray as xr
from ufs2arco import Layers2Pressure

from graphufs.log import setup_simple_log
from graphufs.postprocess import interp2pressure, calc_diagnostics

from config import P2EvaluationEmulator as Emulator

def get_diagnostics(experiment):

    path_in = f"{Emulator.local_store_path}/perturbation/inference.{experiment}.zarr"
    path_out= f"{Emulator.local_store_path}/perturbation/inference.{experiment}.diagnostics.zarr"

    logging.info(f"Reading input from {path_in}")
    xds = xr.open_zarr(path_in)
    xds = xds[["pressfc", "tmp", "spfh"]]

    cds = xr.open_zarr("inputs.common.zarr")
    cds = cds.rename({"grid_yt": "lat", "grid_xt": "lon", "pfull": "level"})
    for key in ["phalf", "ak", "bk", "hgtsfc_static"]:
        xds[key] = cds[key].copy()
    xds = xds.set_coords(["ak", "bk"])

    xds.load();
    logging.info(f"Loaded dataset\n{xds}\n")

    gds = calc_diagnostics(xds, ["geopotential", "delz", "prsl"])

    logging.info(f"Computed diagnostics\n{gds}\n")
    logging.info(f"Writing result to {path_out}")
    gds.to_zarr(path_out, mode="w")


def interpolate_diagnostics(experiment):

    path_in = f"{Emulator.local_store_path}/perturbation/inference.{experiment}.diagnostics.zarr"
    path_out= f"{Emulator.local_store_path}/perturbation/inference.{experiment}.diagnostics.interpolated.zarr"


    logging.info(f"Reading input from {path_in}")
    xds = xr.open_zarr(path_in)
    xds.load();

    plevels = np.concatenate(
        [
            np.arange(250, 850, 50),
            np.arange(850, 1001, 25),
        ],
    )
    pds = interp2pressure(xds, plevels=plevels)
    logging.info(f"Interpolated result\n{pds}\n")
    pds.to_zarr(path_out, mode="w")

if __name__ == "__main__":


    setup_simple_log()
    for experiment in ["noobs", "singleobs"]:
        #get_diagnostics(experiment)
        interpolate_diagnostics(experiment)
