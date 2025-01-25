from functools import partial
import logging
import os
import sys
import jax
import haiku as hk
import numpy as np
import dask
import pandas as pd
import xarray as xr
from tqdm import tqdm

from graphcast import rollout, solar_radiation, data_utils

from graphufs.training import construct_wrapped_graphcast
from graphufs.log import setup_simple_log

from config import P2EvaluationEmulator as Emulator

_lead_times = ["3h", "6h"]
_grid_ncfile = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
_rename = {
    "pfull": "level",
    "grid_yt": "lat",
    "grid_xt": "lon",
}


def store_container(path, xds, chunked_dim_values, chunked_dim_name="time", **kwargs):

    if chunked_dim_name in xds:
        xds = xds.isel({chunked_dim_name:0}, drop=True)

    container = xr.Dataset()
    for key in xds.coords:
        container[key] = xds[key].copy()

    for key in xds.data_vars:
        dims = (chunked_dim_name,) + xds[key].dims
        coords = {chunked_dim_name: chunked_dim_values, **dict(xds[key].coords)}
        shape = (len(chunked_dim_values),) + xds[key].shape
        chunks = (1,) + tuple(-1 for _ in xds[key].dims)

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

def rename2graphcast(xds):
    for key, val in _rename.items():
        if key in xds:
            xds = xds.rename({key: val})
    return xds

def datetime2time(xds, last_ic_datetime):
    time = pd.Timestamp(str(xds.datetime.values.squeeze())) - pd.Timestamp(last_ic_datetime)
    xds["time"] = xr.DataArray(time, coords=xds.datetime.coords)
    xds = xds.set_coords("time")
    return xds.swap_dims({"datetime": "time"})


def get_forecast_time(last_ic_datetime, lead_times):
    time = [pd.Timedelta(t) for t in lead_times]
    datetime = [pd.Timestamp(last_ic_datetime) + pd.Timedelta(t) for t in lead_times]
    tds = xr.Dataset()
    tds["time"] = xr.DataArray(
        time,
        coords={"time": time},
        dims="time",
    )
    tds["datetime"] = xr.DataArray(
        datetime,
        coords=tds.time.coords,
    )
    tds = tds.set_coords("datetime")
    return tds["time"], tds["datetime"]


def get_forcings(last_ic_datetime, lead_times):
    """

    Args:
        last_ic_datetime (str or pandas.Timestamp)
        lead_times (str or pandas.Timedelta)
    """

    # open the grid
    gds = xr.open_zarr(_grid_ncfile)
    gds = rename2graphcast(gds)

    # setup container for the forcings
    fds = xr.Dataset()
    for key in ["lat", "lon"]:
        fds[key] = gds[key]

    fds["time"], fds["datetime"] = get_forecast_time(last_ic_datetime, lead_times)
    fds = fds.set_coords("datetime")

    # First, use code in GraphCast repo to compute downward shortwave radiation
    # note that this is numerically identical to Replay, especially given that
    # this will be converted to half precision
    fds["dswrf_avetoa"] = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
        fds,
        integration_period="3h",
    )
    # GraphCast computes temporal accumulation of solar radiation
    # whereas our model expects the instantaneous flux, averaged over the same time period
    fds["dswrf_avetoa"] = fds["dswrf_avetoa"] / (3600 * 3)

    # If clock variables are desired, add them here
    data_utils.add_derived_vars(fds)
    logging.info(f"get_forcings: created forcings\n{fds}\n")
    return fds

def get_targets_template(last_ic_datetime, lead_times, template):
    """Creates array of zeros at target lead times, basd on template dataset"""

    tds = xr.Dataset()
    tds["time"], tds["datetime"] = get_forecast_time(last_ic_datetime, lead_times)
    tds = tds.set_coords("datetime")

    for key in template.data_vars:
        # Note: if template has dask arrays, these operations are dask friendly - they preserve the chunking
        spatial_var = xr.zeros_like(template[key].isel(time=0, drop=True))
        with_time = spatial_var.broadcast_like(tds["time"]).chunk({"time": 1})
        tds[key] = with_time.transpose(*template[key].dims)

    logging.info(f"get_targets_template: created targets template\n{tds}\n")
    return tds

def predict(
    params,
    state,
    fds,
    tds,
    ic0,
    emulator,
    prefix,
) -> xr.Dataset:

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    # open latest initial condition
    ic1 = xr.open_zarr(f"inputs.{prefix}.zarr")
    ic1 = rename2graphcast(ic1)
    ic1 = datetime2time(ic1, pd.Timestamp(str(ic1.datetime.values.squeeze())))

    # Concatenate common and perturbed inputs
    all_inputs = xr.concat(
        [ic0, ic1],
        dim="time",
        data_vars=[k for k in ic0.data_vars if "_static" not in k],
    )
    # Add clock variables if desired
    data_utils.add_derived_vars(all_inputs)
    logging.info(f"Opened {prefix} IC dataset\n{all_inputs}\n")

    # drop datetime
    fds = fds.drop_vars("datetime")
    tds = tds.drop_vars("datetime")
    all_inputs.drop_vars("datetime")

    # this is necessary for static and clock variables
    # note we don't dneed to do it for forcings because we "expand_dims" later inside the for loop
    for key in all_inputs.data_vars:
        if "member" not in all_inputs[key].dims:
            all_inputs[key] = all_inputs[key].expand_dims({"member": all_inputs["member"]})

    # rename member->batch here
    # because the add_derived_vars can't handle batch dim, but graphcast prediction code needs it...
    m2b = {"member": "batch"}
    tds = tds.rename(m2b)
    all_inputs = all_inputs.rename(m2b)

    pname = f"{emulator.local_store_path}/perturbation/inference.{prefix}.zarr"
    progress_bar = tqdm(total=len(all_inputs["batch"]), ncols=80, desc="Processing")
    for member in all_inputs["batch"].values:

        logging.debug(f"Forecasting member {member:02d}")

        # prepare dims for graphcast and load
        logging.debug(f"\tLoading inputs")
        inputs = all_inputs[list(emulator.input_variables)].sel(batch=[member])
        targets = tds[list(emulator.target_variables)].sel(batch=[member])
        forcings = fds[list(emulator.forcing_variables)].expand_dims({"batch": [member]})
        inputs.load();
        forcings.load();

        # perform input transform, e.g. log(spfh)
        logging.debug(f"\tTransforming inputs")
        for key, mapping in emulator.input_transforms.items():
            with xr.set_options(keep_attrs=True):
                inputs[key] = mapping(inputs[key])

        logging.debug(f"\tinputs\n{inputs}\n")
        logging.debug(f"\ttargets\n{targets}\n")
        logging.debug(f"\tforcings\n{forcings}\n")

        # predictions have dims [batch, time (aka forecast_time), level, lat, lon]
        if not hasattr(predict, "jitted"):
            logging.debug(f"\tJIT'ing the prediction function")
            predict.jitted = drop_state(with_params(jax.jit(run_forward.apply)))

        logging.debug(f"\tCalling rollout.chunked_prediction")
        predictions = rollout.chunked_prediction(
            predict.jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets,
            forcings=forcings,
        )

        # perform output transform, e.g. exp( log_spfh )
        logging.debug(f"\tTransforming outputs")
        for key, mapping in emulator.output_transforms.items():
            with xr.set_options(keep_attrs=True):
                predictions[key] = mapping(predictions[key])

        # Handle output dimensions
        predictions = predictions.rename({"batch": "member"})

        # Store to zarr one batch at a time
        if member == 0:
            store_container(
                pname,
                predictions,
                chunked_dim_values=all_inputs["batch"].values,
                chunked_dim_name="member",
            )

        # Store to zarr
        region = {k: slice(None, None) for k in predictions.dims}
        region["member"] = slice(member, member+1)
        predictions.to_zarr(pname, region=region)

        progress_bar.update()


if __name__ == "__main__":

    setup_simple_log()
    emulator = Emulator()
    dask.config.set(scheduler="threads", num_workers=emulator.dask_threads)

    # read weights
    ckpt_id = emulator.evaluation_checkpoint_id if emulator.evaluation_checkpoint_id is not None else emulator.num_epochs
    params, state = emulator.load_checkpoint(id=ckpt_id)

    # read the common IC
    ic0 = xr.open_zarr("inputs.common.zarr")
    ic0 = rename2graphcast(ic0)
    last_ic_datetime = pd.Timestamp(ic0.datetime.values[0]) + emulator.delta_t
    logging.info(f"last_ic_datetime: {last_ic_datetime}")

    ic0 = datetime2time(ic0, last_ic_datetime)
    logging.info(f"ic0 dataset\n{ic0}\n")

    # create forcing
    fds = get_forcings(
        last_ic_datetime=last_ic_datetime,
        lead_times=_lead_times,
    )

    # create targets template
    tds = get_targets_template(
        last_ic_datetime=last_ic_datetime,
        lead_times=_lead_times,
        template=ic0[[x for x in emulator.target_variables]],
    )

    for prefix in ["noobs", "singleobs"]:
        predict(
            params=params,
            state=state,
            fds=fds,
            tds=tds,
            ic0=ic0,
            emulator=emulator,
            prefix=prefix,
        )
