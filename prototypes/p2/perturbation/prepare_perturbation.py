import os
from datetime import datetime
import logging

import numpy as np
import dask
import xarray as xr
import pandas as pd
import cf_xarray as cfxr
import xesmf

from graphufs.fvemulator import fv_vertical_regrid
from graphufs.postprocess import get_bounds
from graphufs.log import setup_simple_log

from config import P2EvaluationEmulator as Emulator

_outer_read_path = "/work2/noaa/gsienkf/whitaker/C96L127ufs_psonlynoiau_gfsv16_oneobtest"

def open_parent_dataset():
    # get the parent, quarter degree grid
    rds = xr.open_dataset("/work2/noaa/gsienkf/timsmith/replay-grid/0.25-degree/fv3.nc")
    rename = {"grid_yt": "lat", "grid_xt": "lon"}
    for key, val in rename.items():
        if key in rds:
            rds = rds.rename({key: val})
    rds = get_bounds(rds, is_gaussian=True)
    return rds

def _cftime2datetime(cf_time):

    time = np.array([
        pd.Timestamp(
            datetime(
                int(t.dt.year),
                int(t.dt.month),
                int(t.dt.day),
                int(t.dt.hour),
                int(t.dt.minute),
                int(t.dt.second),
            )
        )
        for t in cf_time
    ])
    return xr.DataArray(
        time,
        coords=cf_time.coords,
        attrs={
            "description": "valid time",
        },
    )

def regrid_and_subsample(xds, rds):
    """regrid xds to quarter degree grid
    """

    ods = xr.Dataset({
        key: rds[key] for key in ["lat", "lon", "lat_b", "lon_b"]
    })
    ods = ods.set_coords(["lat_b", "lon_b"])

    # regrid to quarter degree
    weights_filename = "regrid_weights.nc"
    exist = os.path.isfile(weights_filename)
    regridder = xesmf.Regridder(
        ds_in=xds,
        ds_out=ods,
        method="conservative",
        reuse_weights=exist,
        filename=weights_filename,
    )
    if not exist:
        regridder.to_netcdf(filename=weights_filename)
        logging.info(f"Stored weights to {weights_filename}")

    ods = regridder(xds, keep_attrs=True)

    # now subsample in lat/lon
    ods = ods.isel(
        lat=slice(None,None,4),
        lon=slice(None,None,4)
    )
    ods = ods.rename({"lat": "grid_yt", "lon": "grid_xt"})
    return ods

def preprocess(xds, rds):
    """

    Args:
        xds (xr.Dataset): dataset from a single file, could be sanl, sfg, bfg
    """

    # select variables
    all_variables = tuple(set(Emulator.input_variables + Emulator.target_variables + Emulator.forcing_variables))
    all_variables = all_variables + ("delz", "land", "hgtsfc")
    xds = xds[[x for x in all_variables if x in xds]]

    # regrid lat/lon
    xds = regrid_and_subsample(xds, rds)

    # regrid vertical
    if "pfull" in xds.dims:
        xds = fv_vertical_regrid(xds, interfaces=list(Emulator.interfaces))

    if "land" in xds.data_vars:
        xds["land_static"] = xr.where(xds["land"].isel(time=0, drop=True) == 1, 1, 0).astype(np.float32)
        xds = xds.drop_vars("land")

    if "hgtsfc" in xds.data_vars:
        xds["hgtsfc_static"] = xds["hgtsfc"].isel(time=0, drop=True)
        xds = xds.drop_vars("hgtsfc")

    for key in ["ak", "bk"]:
        if key in xds:
            xds = xds.set_coords(key)

    # convert datetime format from cftime to np.datetime64
    xds = xds.rename({"time": "cftime"})
    xds["datetime"] = _cftime2datetime(xds["cftime"])
    xds = xds.swap_dims({"cftime": "datetime"})
    xds = xds.drop_vars("cftime")

    # finally, load this into memory
    xds = xds.load();
    return xds


def open_single_timestamp(cycle, fhr, member, is_sanl, rds):

    justcycle = cycle.split("_")[0]
    if "noobs" in cycle:
        bfg = xr.open_dataset(
            f"{_outer_read_path}/{justcycle}/bfg_{justcycle}_fhr{fhr:02d}_mem{member:03d}"
        )
    else:
        bfg = xr.open_dataset(
            f"{_outer_read_path}/{cycle}/bfg_{justcycle}_fhr{fhr:02d}_mem{member:03d}"
        )
    if "pressfc" in bfg:
        bfg = bfg.drop_vars("pressfc")
    bfg = preprocess(bfg, rds)

    prefix = "sanl" if is_sanl else "sfg"
    sanlsfg = xr.open_dataset(
        f"{_outer_read_path}/{cycle}/{prefix}_{justcycle}_fhr{fhr:02d}_mem{member:03d}",
    )
    sanlsfg = preprocess(sanlsfg, rds)
    xds = xr.merge([sanlsfg, bfg])
    return xds

def store_container(path, xds, members, **kwargs):

    output_chunks = {
        "member": 1,
        "datetime": 1,
        "pfull": 1,
        "grid_yt": -1,
        "grid_xt": -1,
    }
    xds = xds.isel(member=0, drop=True)

    container = xr.Dataset()
    for key in xds.coords:
        container[key] = xds[key].copy()

    for key in xds.data_vars:
        dims = ("member",) + xds[key].dims
        coords = {"member": members, **dict(xds[key].coords)}
        shape = (len(members),) + xds[key].shape
        chunks = (1,) + tuple(output_chunks[d] for d in xds[key].dims)
        container[key] = xr.DataArray(
            data=dask.array.zeros(
                shape=shape,
                chunks=chunks,
                dtype=xds[key].dtype,
            ),
            coords=coords,
            dims=dims,
            attrs=xds[key].attrs.copy(),
        )
    container.to_zarr(path, compute=False, **kwargs)
    logging.info(f"Stored container at {path}")


if __name__ == "__main__":

    setup_simple_log()
    n_members = 80
    output_path = "inputs.singleobs.zarr"
    rds = open_parent_dataset()
    sds = xr.open_dataset("/work2/noaa/gsienkf/timsmith/replay-grid/0.25-degree-subsampled/fv3.nc")
    for member in range(17, n_members):
        ic0 = open_single_timestamp(
            cycle="2021100100",
            fhr=9,
            member=member+1,
            is_sanl=False,
            rds=rds,
        )

        ic1 = open_single_timestamp(
            cycle="2021100106",
            fhr=6,
            member=member+1,
            is_sanl=True,
            rds=rds,
        )

        # handle static variables
        static = ic0[[x for x in ic0 if "_static" in x]]
        ic0 = ic0[[x for x in ic0 if "_static" not in x]]
        ic1 = ic1[[x for x in ic0 if "_static" not in x]]

        xds = xr.concat([ic0, ic1], dim="datetime")
        xds = xds.expand_dims({"member": [member]})
        for key in static.data_vars:
            xds[key] = static[key]

        # check with subsampled grid
        np.testing.assert_allclose(xds.grid_yt.values, sds.grid_yt.values)
        np.testing.assert_allclose(xds.grid_xt.values, sds.grid_xt.values)
        if member == 0:
            store_container(output_path, xds, members=np.arange(n_members))

        region = {k: slice(None, None) for k in xds.dims}
        region["member"] = slice(member, member+1)
        xds.to_zarr(output_path, region=region)
        logging.info(f"Done with {member} / {n_members}")
