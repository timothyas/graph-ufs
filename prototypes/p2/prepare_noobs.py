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
from prepare_perturbation import open_single_timestamp, store_container, open_parent_dataset

_outer_read_path = "/work2/noaa/gsienkf/whitaker/C96L127ufs_psonlynoiau_gfsv16_oneobtest"

if __name__ == "__main__":

    setup_simple_log()
    n_members = 80
    output_path = "perturbation.noobs.p2.zarr"
    rds = open_parent_dataset()
    sds = xr.open_dataset("/work2/noaa/gsienkf/timsmith/replay-grid/0.25-degree-subsampled/fv3.nc")
    for member in range(2, n_members):
        xds = open_single_timestamp(
            cycle="2021100106_noobs",
            fhr=6,
            member=member+1,
            is_sanl=False,
            rds=rds,
        )

        xds = xds.expand_dims({"member": [member]})

        # check with subsampled grid
        np.testing.assert_allclose(xds.grid_yt.values, sds.grid_yt.values)
        np.testing.assert_allclose(xds.grid_xt.values, sds.grid_xt.values)
        if member == 0:
            store_container(output_path, xds, members=np.arange(n_members))

        region = {k: slice(None, None) for k in xds.dims}
        region["member"] = slice(member, member+1)
        xds.to_zarr(output_path, region=region)
        logging.info(f"Done with {member} / {n_members}")
