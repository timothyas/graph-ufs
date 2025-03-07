import logging
import xarray as xr
import dask

def swap_batch_time_dims(xds, inittimes):

    xds = xds.rename({"time": "lead_time"})

    # create "time" dimension = t0
    xds["time"] = xr.DataArray(
        inittimes,
        coords=xds["batch"].coords,
        dims=xds["batch"].dims,
        attrs={
            "description": "Forecast initialization time, last timestep of initial conditions",
        },
    )

    # swap logical batch for t0
    xds = xds.swap_dims({"batch": "time"}).drop_vars("batch")
    return xds


def store_container(path, xds, loader, **kwargs):

    time = loader.initial_times

    # we're just getting a single sample, which may have only a slice of
    # other dimensions (e.g. just one ensemble member)
    # need to keep track of these so that we create the whole array just like time

    full_arrays_of_sample_dims = {
        "time": loader.initial_times # this one is special, because of how we subsample it with initial_condition_stride
    }
    # others we can pull directly from the dataset
    original_dims = {key: xds[key].dims for key in xds.data_vars}
    for key in loader.sample_dims:
        if key != "time":
            full_arrays_of_sample_dims[key] = loader.xds[key]
        if key in xds:
            xds = xds.isel({key: 0}, drop=True)

    container = xr.Dataset()
    for key in xds.coords:
        container[key] = xds[key].copy()

    for key in xds.data_vars:
        og = original_dims[key]
        local_sample_dims = tuple(dim for dim in loader.sample_dims if dim in og)
        dims = local_sample_dims + xds[key].dims

        local_sample_coords = {dim: full_arrays_of_sample_dims[dim] for dim in local_sample_dims}
        coords = {**local_sample_coords, **dict(xds[key].coords)}
        shape = tuple(len(x) for x in local_sample_coords.values()) + xds[key].shape
        chunks = tuple(1 for _ in local_sample_dims) + tuple(-1 for _ in xds[key].dims)

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
