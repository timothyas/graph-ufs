"""Standalone script to run GraphUIFS forecast

Example usage:

    $ python forecast.py \
            --ensemble_member=10 \
            --previous_cycle=2021120106 \
            --experiment_directory=/work2/noaa/gsienkf/whitaker/C96L127_graphufs_psonlyiau \
            --fhr_ics=6,9

For help with the inputs:

    $ python forecast.py --help

Note:
    For the first cycle, we want to read fhr06 and fhr09 (--fhr_ics=6,9)
    After that we will just make 6 hour predictions with the ML models, so we just want fhr03 and fhr06 (--fhr_ics=3,6)
"""
import os
import sys
from typing import Optional
from functools import partial
import logging
import numpy as np
import xarray as xr
import pandas as pd
import jax
import haiku as hk

from absl import app, flags

from cftime import DatetimeJulian
from datetime import datetime

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization as graphcastnorm,
    rollout,
    solar_radiation,
)

ENSEMBLE_MEMBER = flags.DEFINE_integer(
    "ensemble_member",
    None,
    help="Ensemble member ID to propagate",
)
PREVIOUS_CYCLE = flags.DEFINE_string(
    "previous_cycle",
    None,
    help="Previous DA cycle to use for forecast Initial Conditions, e.g. '2021120106'",
)
EXPERIMENT_DIRECTORY = flags.DEFINE_string(
    "experiment_directory",
    None,
    help="Outermost directory, containing all the DA cycles.",
)
FHR_ICS = flags.DEFINE_list(
    "fhr_ics",
    None,
    help="Forecast hours to grab for initial conditions. It will be --fhr_ics=6,9 when initializing from a single-IC model (e.g. UFS), and --fhr_ics=3,6 when initializing from a 2-IC ML model",
)
MODEL_WEIGHTS = flags.DEFINE_string(
    "model_weights",
    "/work2/noaa/gsienkf/timsmith/model-weights/graphufs_p1v1.npz",
    help="Path to model weights to be used with GraphCast code base",
)


_norm_paths = {
    "mean": "/work2/noaa/gsienkf/timsmith/replay-normalization-statistics/0.25-degree-subsampled/03h-freq/fv3.statistics.1993-2019/mean_by_level.nc",
    "std": "/work2/noaa/gsienkf/timsmith/replay-normalization-statistics/0.25-degree-subsampled/03h-freq/fv3.statistics.1993-2019/stddev_by_level.nc",
    "stddiff": "/work2/noaa/gsienkf/timsmith/replay-normalization-statistics/0.25-degree-subsampled/03h-freq/fv3.statistics.1993-2019/diffs_stddev_by_level.nc",
}

class SimpleFormatter(logging.Formatter):
    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def setup_log(level=logging.INFO):

    logging.basicConfig(
        stream=sys.stdout,
        level=level,
    )
    logger = logging.getLogger()
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)

def _cycle2timestamp(cycle):
    return pd.Timestamp(f"{cycle[:4]}-{cycle[4:6]}-{cycle[6:8]}T{cycle[8:10]}")

def _timestamp2cycle(timestamp):
    cycle = str(timestamp)
    return cycle.replace("-","").replace("T","").replace(" ","").replace(":","")[:10]

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

def _datetime2cftime(datetime):
    cf_time = np.array([
        DatetimeJulian(
            int(t.dt.year),
            int(t.dt.month),
            int(t.dt.day),
            int(t.dt.hour),
            int(t.dt.minute),
            int(t.dt.second),
            has_year_zero=False,
        )
        for t in datetime
    ])
    return xr.DataArray(
        cf_time,
        coords=datetime.coords,
        attrs={
            "long_name": "time",
            "axis": "T",
            "calendar_type": "JULIAN",
        },
    )


def postprocess(xds, last_ic_datetime):
    """Take prediction from GraphCast and prepare it for FV3/DA"""

    rename = {
        "level": "pfull",
        "lat": "grid_yt",
        "lon": "grid_xt",
    }
    for key, val in rename.items():
        if key in xds:
            xds = xds.rename({key: val})

    xds = xds.squeeze().drop_vars("batch")
    xds = xds.rename({"time": "ftime"})

    datetime = last_ic_datetime + xds["ftime"]
    xds["time"] = _datetime2cftime(datetime)

    # keep ftime as the main dimension to make storing the result easier
    xds = xds.set_coords("time")

    return xds

def store_result(xds, prefix_mapper, cycle, member, experiment_directory):

    outer_dir = f"{experiment_directory}/{cycle}"
    if not os.path.isdir(outer_dir):
        os.makedirs(outer_dir)

    for ftime in xds.ftime.values:
        fhr = int(ftime / 3600 / 1e9)
        result = xds.sel(ftime=[ftime])
        result = result.swap_dims({"ftime": "time"}).drop_vars("ftime")
        for prefix in ["sfg", "bfg"]:
            fname = f"{outer_dir}/{prefix}_{cycle}_fhr{fhr:02d}_mem{member:03d}"
            tmpds = xr.Dataset({
                key: result[key]
                for key in result.data_vars if prefix_mapper[key] == prefix or key == "pressfc"
            },
            attrs=xds.attrs["file_attrs"][prefix.replace("sfg","sanl")]
            )
            if "ak" in xds.attrs and prefix == "sfg":
                tmpds.attrs["ak"] = xds.ak
                tmpds.attrs["bk"] = xds.bk
            tmpds.to_netcdf(fname)
            logging.info(f"Stored {fname}")


def get_all_variables(task_config):
    return tuple(set(
        task_config.input_variables + task_config.target_variables + task_config.forcing_variables
    ))


def get_empty_targets(lead_times, template):
    """Creates array of zeros at target lead times, basd on template dataset"""

    lead_times, duration = data_utils._process_target_lead_times_and_get_duration(lead_times)
    container = xr.concat(
        list(
            xr.zeros_like(
                template.isel(time=0, drop=True).expand_dims(
                    {"time": [t]}
                )
            )
            for t in lead_times
        ),
        dim="time",
    )
    container["datetime"] = xr.DataArray(
        [template.datetime.values[-1] + t for t in container.time.values],
        coords=container["time"].coords,
    )
    container = container.set_coords("datetime")
    return container

def get_forcings(lead_times, ids):
    """This is a hack for now, since the forcings component is a bit wonky"""

    # setup container for the forcings

    fds = xr.Dataset()
    for key in ["lat", "lon"]:
        fds[key] = ids[key]

    time = [pd.Timedelta(t) for t in lead_times]
    datetime = [pd.Timestamp(ids.datetime.values[-1]) + pd.Timedelta(t) for t in lead_times]
    fds["time"] = xr.DataArray(
        time,
        coords={"time": time},
        dims="time",
    )
    fds["datetime"] = xr.DataArray(
        datetime,
        coords=fds.time.coords,
    )
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

    # Now get hgtsfc
    fds["hgtsfc"] = ids["hgtsfc"].isel(time=0, drop=True)
    fds["hgtsfc"] = fds["hgtsfc"].broadcast_like(fds["time"])

    # Now for the real hack:
    # land is a time varying quantity, and we falsely prescribe it as forcings
    # during prediction time
    # for now, just take the last "land" value from the initial condition dataset
    # and keep it static going forward
    fds["land"] = ids["land"].isel(time=-1, drop=True)
    fds["land"] = fds["land"].broadcast_like(fds["time"])
    return fds



def get_model_inputs(xds, member, lead_times, task_config):
    """Transform initial conditions from sanl/bfg files to inputs, targets_template, forcings
    so they're ready to be passed to the model
    """

    # append template and forcings for the target lead times
    targets_template = get_empty_targets(lead_times, xds[[x for x in task_config.target_variables if x in xds]])
    forcings = get_forcings(lead_times, xds[[x for x in task_config.forcing_variables if x in xds]])
    future = xr.merge([targets_template, forcings])
    xds = xr.concat([xds, future], dim="time")

    for key in xds.data_vars:
        isnans = np.isnan(xds[key]).any(["lat", "lon"])
        if isnans.any().values:
            logging.warning(f"{key} found NaNs: \n{isnans}\n")

    # transform into inputs, targets_template, forcings
    levels = list(
        xds["level"].sel(
            level=list(task_config.pressure_levels),
            method="nearest",
        ).values
    )
    inputs, targets_template, forcings = data_utils.extract_inputs_targets_forcings(
        xds,
        input_variables=task_config.input_variables,
        target_variables=task_config.target_variables,
        forcing_variables=task_config.forcing_variables,
        pressure_levels=levels,
        input_duration=task_config.input_duration,
        target_lead_times=lead_times,
    )
    inputs = inputs.expand_dims({"batch": [member]})
    targets_template = targets_template.expand_dims({"batch": [member]})
    forcings = forcings.expand_dims({"batch": [member]})
    return inputs, targets_template, forcings

def preprocess(xds, cycle, relative_time, task_config, prefix):
    """

    Args:
        xds (xr.Dataset): dataset from a single file, could be sanl, sfg, bfg
        cycle (str): DA cycle
        relative_hours (float, int): time relative to t0, t0 should be 0
        task_config (TaskConfig): for GraphCast
    """

    # select variables
    is_from_ufs = "o3mr" in xds
    xds = xds[[x for x in get_all_variables(task_config) if x in xds]]

    # select vertical levels
    if "pfull" in xds.dims:
        pfull = xds.pfull.values
        xds = xds.sel(pfull=list(task_config.pressure_levels), method="nearest")

        # get the ak/bk coefficients, above each level
        if is_from_ufs:

            # instead of complicated estimate below, use indices suggested by Jeff
            indices = [ 28, 40, 47, 53, 56, 61, 65, 73, 79, 83, 94, 104,115,127]
            #indices = [list(pfull).index(x) for x in xds.pfull.values]
            #indices.sort()
            ## now append the very bottom (surface)
            #indices.append(indices[-1]+1)

            ## now for the interfaces in the middle, get an interface somewhere between the
            ## subsampled cell centers
            ## do this while preserving the top coefficient, and the bottom two coefficents
            ## closest to the surface
            #middle = list(int(np.mean([x,y])) for x, y in zip(indices[:-3], indices[1:-2]))
            #indices = [indices[0]] + middle + indices[-2:]

            xds.attrs["ak"] = xds.ak[indices]
            xds.attrs["bk"] = xds.bk[indices]

        else:
            xds.attrs["ak"] = xds.ak
            xds.attrs["bk"] = xds.bk

    # convert datetime format from cftime to np.datetime64
    xds = xds.rename({"time": "cftime"})
    xds["datetime"] = _cftime2datetime(xds["cftime"])
    xds = xds.swap_dims({"cftime": "datetime"})
    xds = xds.drop_vars("cftime")

    # set time as time relative to initial conditions
    xds["time"] = xr.DataArray(
        [pd.Timedelta(hours=relative_time)],
        coords=xds["datetime"].coords,
        attrs={"description": "time relative to last initial condition"},
    )
    xds = xds.swap_dims({"datetime": "time"})
    xds = xds.set_coords("datetime")

    # rename these before passing to GraphCast code
    rename = {
        "pfull": "level",
        "grid_yt": "lat",
        "grid_xt": "lon",
    }
    for k, v in rename.items():
        if k in xds:
            xds = xds.rename({k: v})

    # add file prefix to these variables in order to store result in separate netcdf files
    for key in xds.data_vars:
        xds[key].attrs["file_prefix"] = prefix.replace("sanl", "sfg")

    # finally, load this into memory
    xds = xds.load();
    return xds


def open_dataset(
    cycle: str,
    member: int,
    task_config: graphcast.TaskConfig,
    fhrs: Optional[tuple[int]] = (3, 6),
    experiment_directory: Optional[str] = "./",
) -> xr.Dataset:
    """
    Args:
        cycle (str): DA cycle in YYYYMMDDHH
        member (int): ensemble member number
        task_config (graphcast.TaskConfig): for GraphCast, defines variables, pressure levels etc used
        fhrs (tuple[int], optional): forecast hours from the previous forecast to use
        experiment_directory (str, optional): where to look for the data

    Returns:
        xds (xr.Dataset): with sanl/bfg combined, taking surface pressure from sanl since that's updated by the DA
    """

    attrs = {}
    dslist1 = []
    for fhr in fhrs:
        dslist2 = []
        for prefix in ["sanl", "bfg"]:
            tmp = xr.open_dataset(
                f"{experiment_directory}/{cycle}/{prefix}_{cycle}_fhr{fhr:02d}_mem{member:03d}",
            )
            if prefix == "bfg" and "pressfc" in tmp:
                tmp = tmp.drop_vars("pressfc")
            if prefix not in attrs:
                attrs[prefix] = tmp.attrs.copy()

            tmp = preprocess(
                tmp,
                cycle=cycle,
                relative_time=fhr-fhrs[-1],
                task_config=task_config,
                prefix=prefix,
            )

            dslist2.append(tmp)
        dslist1.append(xr.merge(dslist2))

    xds = xr.concat(dslist1, dim="time")
    xds.attrs["file_attrs"] = attrs.copy()
    return xds


def load_normalization(task_config):
    """Assumes we want normalization from the 1/4 degree subsampled to ~1 degree data"""

    normalization = dict()
    for key in ["mean", "std", "stddiff"]:
        this_norm = xr.open_dataset(_norm_paths[key])
        this_norm = this_norm[[x for x in get_all_variables(task_config) if x in this_norm]]
        this_norm = this_norm.sel(pfull=list(task_config.pressure_levels), method="nearest")
        this_norm = this_norm.rename({"pfull": "level"})
        normalization[key] = this_norm.load()

    return normalization


def load_checkpoint(path):

    with open(path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    return params, model_config, task_config


def construct_wrapped_graphcast(model_config, task_config, normalization):

    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = graphcastnorm.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=normalization["stddiff"],
        mean_by_level=normalization["mean"],
        stddev_by_level=normalization["std"],
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(inputs, targets_template, forcings, model_config, task_config, normalization):
    predictor = construct_wrapped_graphcast(model_config, task_config, normalization)
    return predictor(inputs, targets_template, forcings)


def predict(
    params: dict,
    inputs: xr.DataArray,
    targets_template: xr.DataArray,
    forcings: xr.DataArray,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    normalization: dict,
) -> xr.Dataset:

    state = dict()

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    def with_configs(fn):
        return partial(
            fn,
            model_config=model_config,
            task_config=task_config,
            normalization=normalization,
        )

    if not hasattr(predict, "gc_forecast"):
        logging.info("JIT compiling the predictor")
        predict.gc_forecast = drop_state( with_params( jax.jit( with_configs( run_forward.apply ) ) ) )

    predictions = rollout.chunked_prediction(
        predict.gc_forecast,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )
    return predictions

def main(argv):

    # Inputs and some checking
    previous_cycle = PREVIOUS_CYCLE.value
    member = ENSEMBLE_MEMBER.value
    experiment_directory = EXPERIMENT_DIRECTORY.value
    fhr_ics = tuple(int(x) for x in FHR_ICS.value)
    model_weights = MODEL_WEIGHTS.value

    assert len(previous_cycle) == 10, "previous_cycle must be a 10 character long string in format YYYYMMDDHH"
    expected = _timestamp2cycle(_cycle2timestamp(previous_cycle))
    assert previous_cycle == expected, "previous_cycle must be a 10 character long string in format YYYYMMDDHH"

    next_cycle = _timestamp2cycle( _cycle2timestamp(previous_cycle) + pd.Timedelta(hours=6) )

    setup_log()
    logging.info(f"Reading weights from: {model_weights}")
    params, model_config, task_config = load_checkpoint(model_weights)

    logging.info("Reading normalization")
    normalization = load_normalization(task_config)

    # Note that all of the code below, including postprocess, assumes
    # we are working one ensemble member at a time

    logging.info(f"Reading initial conditions from cycle: {previous_cycle}, fhrs: {fhr_ics}, member: {member}")
    xds = open_dataset(
        cycle=previous_cycle,
        member=member,
        task_config=task_config,
        fhrs=fhr_ics,
        experiment_directory=experiment_directory,
    )
    prefix_mapper = {key: xds[key].attrs["file_prefix"] for key in xds.data_vars}

    logging.info("Preparing initial conditions for GraphCast code")
    inputs, targets_template, forcings = get_model_inputs(
        xds,
        member=member,
        lead_times=["3h", "6h"],
        task_config=task_config,
    )

    logging.info(f"Starting GraphUFS Prediction for cycle: {next_cycle}, member: {member}")
    prediction = predict(
        params=params,
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
        model_config=model_config,
        task_config=task_config,
        normalization=normalization,
    )

    logging.info("Postprocessing Prediction for DA")
    prediction = postprocess(prediction, last_ic_datetime=xds["datetime"].values[1])
    prediction.attrs = xds.attrs.copy()

    # take the forcing and stick it in the prediction dataset as well
    postforcing = postprocess(forcings, last_ic_datetime=xds["datetime"].values[1])
    for key in postforcing.data_vars:
        if key in xds.data_vars:
            logging.info(f"Storing forcing[{key}] -> prediction dataset")
            prediction[key] = postforcing[key]
            prediction[key].attrs = xds[key].attrs.copy()


    store_result(
        prediction,
        prefix_mapper=prefix_mapper,
        cycle=next_cycle,
        member=member,
        experiment_directory=experiment_directory,
    )

if __name__ == "__main__":
    app.run(main)
