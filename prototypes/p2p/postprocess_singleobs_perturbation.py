import logging
import numpy as np
import dask
import xarray as xr
from ufs2arco import Layers2Pressure

from graphufs.log import setup_simple_log
from graphufs.postprocess import interp2pressure, calc_diagnostics

def get_diagnostics(Emulator, experiment):

    pdir = f"{Emulator.local_store_path}/singleobs-perturbation"
    path_in = f"{pdir}/inference.{experiment}.zarr"
    path_out= f"{pdir}/inference.{experiment}.diagnostics.zarr"

    logging.info(f"Reading input from {path_in}")
    xds = xr.open_zarr(path_in)
    xds = xds[["pressfc", "tmp", "spfh"]]

    cds = xr.open_zarr(f"{pdir}/inputs.common.zarr")
    cds = cds.rename({"grid_yt": "lat", "grid_xt": "lon", "pfull": "level"})
    for key in ["phalf", "ak", "bk", "hgtsfc_static"]:
        xds[key] = cds[key].copy()
    xds = xds.set_coords(["ak", "bk"])

    xds.load();
    logging.info(f"Loaded dataset\n{xds}\n")

    gds = calc_diagnostics(xds, ["geopotential", "hydrostatic_layer_thickness", "prsl"])

    logging.info(f"Computed diagnostics\n{gds}\n")
    logging.info(f"Writing result to {path_out}")
    gds.to_zarr(path_out, mode="w")
    logging.info(f" ... done")


def interpolate_diagnostics(Emulator, experiment):

    path_in = f"{Emulator.local_store_path}/singleobs-perturbation/inference.{experiment}.diagnostics.zarr"
    path_out= f"{Emulator.local_store_path}/singleobs-perturbation/inference.{experiment}.diagnostics.interpolated.zarr"


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
    logging.info("... done")

def main(Emulator):


    setup_simple_log()
    for experiment in ["noobs", "singleobs"]:
        get_diagnostics(Emulator, experiment)
        interpolate_diagnostics(Emulator, experiment)

    logging.info("Done postprocessing perturbation experiment results")
