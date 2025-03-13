from functools import partial
import logging
import os
import sys
import jax
import haiku as hk
import numpy as np
import dask
import xarray as xr
from tqdm import tqdm

from graphcast import rollout

from graphufs.training import construct_wrapped_graphcast
from graphufs.batchloader import MPIExpandedBatchLoader
from graphufs.datasets import Dataset
from graphufs.inference import swap_batch_time_dims, store_container
from graphufs.mpi import MPITopology
from graphufs import diagnostics
from graphufs import fvemulator
_static_vars = ("lsm", "orog")

def handle_member(xds):
    if "member" in xds:
        xds = xds.swap_dims({"member": "original_member"})
        xds = xds.drop_vars("member")
        xds = xds.rename({"original_member": "member"})
    if "t0" in xds:
        xds = xds.drop_vars("t0")
    return xds

def predict(
    emulator,
    batchloader,
    mpi_topo,
) -> xr.Dataset:

    # setup weights
    ckpt_id = emulator.evaluation_checkpoint_id if emulator.evaluation_checkpoint_id is not None else emulator.num_epochs
    logging.info(f"Running inference with checkpoint_id = {ckpt_id}")
    logging.info(f"Reading weights ...")
    params, state = emulator.load_checkpoint(id=ckpt_id)

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    logging.info("JIT Compiling Predict")
    gc = drop_state(with_params(jax.jit(run_forward.apply)))
    logging.info("Done Compiling Predict")

    hours = int(emulator.forecast_duration.value / 1e9 / 3600)
    pname = f"{emulator.inference_directory}/{batchloader.dataset.mode}/graphufs.{hours}h.zarr"

    n_steps = len(batchloader)
    with open(mpi_topo.progress_file, "a") as f:
        progress_bar = tqdm(total=n_steps, ncols=80, desc="Processing", file=f)
        for k, (inputs, targets, forcings) in enumerate(batchloader):

            if inputs is not None:
                # retrieve and drop t0
                inittimes = inputs.datetime.isel(time=-1).values
                inputs = inputs.drop_vars("datetime")
                targets = targets.drop_vars("datetime")
                forcings = forcings.drop_vars("datetime")

                inputs = handle_member(inputs)
                targets = handle_member(targets)
                forcings = handle_member(forcings)

                # predictions have dims [batch, time (aka forecast_time), level, lat, lon]
                predictions = rollout.chunked_prediction(
                    gc,
                    rng=jax.random.PRNGKey(0),
                    inputs=inputs,
                    targets_template=np.nan * targets,
                    forcings=forcings,
                )

                # perform output transform, e.g. exp( log_spfh )
                for key, mapping in emulator.output_transforms.items():
                    with xr.set_options(keep_attrs=True):
                        predictions[key] = mapping(predictions[key])

                # now check for static variables, we don't want these in the resulting datasets
                predictions = predictions[[key for key in list(predictions.coords)+list(predictions.data_vars) if key not in _static_vars]]
                # Add t0 as new variable, and swap out for logical sample/batch index
                # swap dims to be [time (aka initial condition time), lead_time (aka forecast_time), level, lat, lon]
                predictions = swap_batch_time_dims(predictions, inittimes)

                # Grab attributes
                for key in predictions.data_vars:
                    predictions[key].attrs = targets[key].attrs.copy()

                # Store to zarr one batch at a time
                if k == 0:
                    if mpi_topo.is_root:
                        store_container(pname, predictions, loader=batchloader, mode="w")
                    mpi_topo.comm.Barrier()

                # Store to zarr
                region = batchloader.find_my_region(predictions)
                predictions.to_zarr(pname, region=region)

            progress_bar.update()
        progress_bar.close()


def inference(Emulator):

    topo = MPITopology(log_dir=f"{Emulator.local_store_path}/logs/inference")
    emulator = Emulator(mpi_rank=topo.rank, mpi_size=topo.size)

    vds = Dataset(
        emulator,
        mode="validation",
        preload_batch=False,
    )

    validator = MPIExpandedBatchLoader(
        vds,
        batch_size=emulator.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        max_queue_size=1,
        initial_condition_stride=emulator.initial_condition_stride,
        mpi_topo=topo,
    )
    assert validator.data_per_device == 1

    predict(
        emulator=emulator,
        batchloader=validator,
        mpi_topo=topo,
    )

    validator.shutdown()
    logging.info("Done running inference")
