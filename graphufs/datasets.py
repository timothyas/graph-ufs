"""
Implementations of Torch Dataset and DataLoader
"""
import logging
import itertools
from os.path import join
from typing import Optional
import numpy as np
import xarray as xr
import dask.array

from xbatcher import BatchGenerator

from graphcast.model_utils import dataset_to_stacked

from .emulator import ReplayEmulator

class Dataset():
    """
    Dataset for Replay Data, in the style of pytorch, but does not require torch
    """

    def __init__(
        self,
        emulator: ReplayEmulator,
        mode: str,
        preload_batch: bool = False,
        input_chunks: Optional[dict | None] = None,
        target_chunks: Optional[dict | None] = None,
    ):
        """
        Initializes the Dataset object.

        Args:
            emulator (ReplayEmulator): The emulator object.
            mode (str): "training", "validation", or "testing"
            preload_batch (bool, optional): If True, preload a sample before doing any processing, usually a good idea
            input_chunks, target_chunks (dict, optional): chunks used to store a local dataset
        """
        self.emulator = emulator
        self.mode = mode
        self.preload_batch = preload_batch
        self.input_chunks = input_chunks
        self.target_chunks = target_chunks
        xds = self._open_dataset()

        self.stacked_dims = tuple(x for x in self.possible_stacked_dims if x in xds.dims or x in ("batch", "channels"))
        self.preserved_dims = self.stacked_dims[:-1]

        self.sample_generator = BatchGenerator(
            ds=xds,
            input_dims=self.input_dims,
            input_overlap=self.input_overlap,
            preload_batch=preload_batch,
        )

        # Now, create "sample_indices", which has a dict mapping to the dimensions that define each sample
        # these will be equivalent to the dimensions in the sample generator that are not "full"
        # for example, if each sample has len(time)=2 and len(member) = 1, then
        # self.sample_generator[10] will have the same time and member dimensions as sample_indices[10]
        # but the sample_generator version will (of course) have all the other dimensions
        # like latitude, longitude etc, whereas sample_indices just has the dimensions we're looping over
        #
        # Ok, and note also that "sample_indices" just has the first item of each dimension, so when using
        # it we'll need to take a slice that is self.input_dims large, starting with that initial index
        for key, val in self.input_overlap.items():
            if self.input_dims[key] - val > 1:
                raise NotImplementedError(f"Dataset.__init__: will neeed to rework sample_indices creation when input_dim['{key}'] - input_overlap['{key}'] > 1. Right now it's {self.input_dims[key]} and {val}, respectively.")
        self.sample_dims = tuple(key for key, val in self.input_dims.items() if val < len(xds[key]))
        self.sample_sizes = {key: len(xds[key]) - self.input_overlap.get(key, 0) for key in self.sample_dims}


    def __len__(self) -> int:
        """
        Returns the number of sample forecasts in the dataset

        Returns:
            length (int): The length of the dataset.
        """
        return len(self.sample_generator)

    def __getitem__(self, idx) -> tuple[xr.DataArray]:
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            X, y (np.ndarray): with inputs and targets
        """
        if isinstance(idx, int):
            sample_input, sample_target, sample_forcing = self.get_xarrays(idx)
        else:
            sample_input, sample_target, sample_forcing = self.get_batch_of_xarrays(idx)

        x = self._stack(sample_input, sample_forcing)
        y = self._stack(sample_target)
        return x, y


    @property
    def xds(self) -> xr.Dataset:
        """
        Returns the xarray dataset.

        Returns:
            xds (xarray.Dataset): The xarray dataset.
        """
        return self.sample_generator.ds

    @property
    def local_inputs_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "inputs.zarr")

    @property
    def local_targets_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "targets.zarr")

    @property
    def initial_times(self) -> list[np.datetime64]:
        """Returns unique list of all initial conditions, i.e. not repeating it for each sample if multiple data samples have the sample initial condition timestamp"""
        return [self.xds["time"].values[i + self.emulator.n_input - 1] for i in range(self.sample_sizes["time"])]

    @property
    def possible_stacked_dims(self) -> tuple[str]:
        return self.emulator.possible_stacked_dims

    @property
    def input_dims(self) -> tuple[str]:
        return self.emulator.input_dims

    @property
    def input_overlap(self) -> tuple[str]:
        return self.emulator.input_overlap


    def _stack(self, a: xr.DataArray, b: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Stack xarrays to form input tensors.

        Args:
            a (xarray.DataArray): First xarray.
            b (xarray.DataArray, optional): Second xarray.

        Returns:
            result (xarray.DataArray): Stacked xarray.
        """
        result = dataset_to_stacked(a, preserved_dims=tuple(d for d in self.preserved_dims if d in a))
        if b is not None:
            result = xr.concat(
                [result, dataset_to_stacked(b, preserved_dims=tuple(d for d in self.preserved_dims if d in b))],
                dim="channels",
            )
        result = result.transpose(*self.stacked_dims)
        return result

    def _open_dataset(self) -> xr.Dataset:
        """
        Open, subsample, and rename variables in the dataset.

        Returns:
            xds (xarray.Dataset): Preprocessed xarray dataset.
        """
        xds = self.emulator.open_dataset()
        time = self.emulator.get_time(mode=self.mode)
        xds = self.emulator.subsample_dataset(xds, new_time=time)
        xds = self.emulator.check_for_ints(xds)
        xds = xds.rename({val: key for key, val in self.emulator.dim_names.items() if val in xds})
        for key in ["cftime", "ftime"]:
            if key in xds:
                xds = xds.drop_vars(key)
        return xds

    def get_xarrays(self, idx: int) -> tuple:
        """
        Get input, target, and forcing xarrays.

        Args:
            idx (int): Index of the sample.

        Returns:
            xinput, xtarget, xforcing (xarray.DataArray): as from graphcast.data_utils.extract_inputs_targets_forcings
        """
        sample = self.sample_generator[idx]

        xinput, xtarget, xforcing = self.emulator.extract_inputs_targets_forcings(
            sample,
            drop_datetime=False,
        )
        xinput = xinput.expand_dims({"batch": [idx]})
        xtarget = xtarget.expand_dims({"batch": [idx]})
        xforcing = xforcing.expand_dims({"batch": [idx]})
        return xinput, xtarget, xforcing

    def get_batch_of_xarrays(self, indices: list[int]) -> tuple:
        """
        Get batches of input, target, and forcing xarrays,
        convenience method to compare the "StackedGraphCast" and "GraphCast" implementations.

        Args:
            indices (list[int]): List of sample indices.

        Returns:
            Tuple of input, target, and forcing xarrays.
        """
        xinputs = []
        xtargets = []
        xforcings = []
        for idx in indices:
            xi, xt, xf = self.get_xarrays(idx)
            xinputs.append(xi)
            xtargets.append(xt)
            xforcings.append(xf)

        xinputs = xr.concat(xinputs, dim="batch")
        xtargets = xr.concat(xtargets, dim="batch")
        xforcings = xr.concat(xforcings, dim="batch")
        return xinputs, xtargets, xforcings

    def store_sample(self, idx: int) -> None:
        x,y = self[idx]

        x = x.load()
        y = y.load()
        x = x.expand_dims("batch").rename({"batch": "sample"})
        y = y.expand_dims("batch").rename({"batch": "sample"})

        x = x.chunk(self.input_chunks)
        y = y.chunk(self.target_chunks)
        spatial_region = {k : slice(None, None) for k in x.dims if k != "sample"}
        region = {"sample": slice(idx, idx+1), **spatial_region}
        for name, array, path in zip(
            ["inputs", "targets"],
            [x, y],
            [self.local_inputs_path, self.local_targets_path],
        ):
            if "batch" in array.coords:
                array = array.drop_vars("batch")
            array.to_dataset(name=name).to_zarr(
                path,
                region=region,
            )

    def get_container(self, template: xr.Dataset, name: str, chunks: dict):

        if "batch" in template.dims:
            template = template.isel(batch=0, drop=True)

        xds = xr.Dataset()
        xds["sample"] = np.arange(len(self))
        for key in self.stacked_dims[1:]:
            xds[key] = template[key].copy()

        dims = ("sample",) + template.dims
        shape = (len(self),) + template.shape
        xds[name] = xr.DataArray(
            data=dask.array.zeros(
                shape=shape,
                chunks=tuple(chunks[k] for k in dims),
                dtype=template.dtype,
            ),
            dims=dims,
        )
        return xds

    def store_containers(self):

        # get templates
        x, y = self[0]
        for name, template, chunks, path in zip(
            ["inputs", "targets"],
            [x, y],
            [self.input_chunks, self.target_chunks],
            [self.local_inputs_path, self.local_targets_path],
        ):
            xds = self.get_container(template=template, name=name, chunks=chunks)
            if "batch" in xds:
                xds = xds.drop_vars("batch")
            xds.to_zarr(path, compute=False, mode="w", consolidated=True)


class PackedDataset():
    """Similar in style to the Dataset class, and to PyTorch, but no torch dependency
    and relies on the dataset being local and ready to go for training.

    Note that this still returns xarray.DataArray with __getitem__, and this is so
    that BatchLoader can pull a full batch in a single dask/zarr call
    """

    def __init__(self, emulator, mode, missing_samples=None, **kwargs):
        self.emulator = emulator
        self.mode = mode
        self.inputs = xr.open_zarr(self.local_inputs_path, **kwargs)
        self.targets = xr.open_zarr(self.local_targets_path, **kwargs)

        self.drop_missing(missing_samples)

    def __len__(self):
        return len(self.inputs["sample"])

    def __getitem__(self, idx):
        x = self.inputs["inputs"].isel(sample=idx, drop=True)
        y = self.targets["targets"].isel(sample=idx, drop=True)
        return x, y

    def drop_missing(self, missing_samples=None):
        if missing_samples is not None:
            if isinstance(missing_samples, int):
                missing_samples = [missing_samples]
            for idx in missing_samples:
                logging.info(f"PackedDataset: dropping missing sample at idx = {idx}")
                self.inputs = self.inputs.drop_sel(sample=idx)
                self.targets = self.targets.drop_sel(sample=idx)

    @property
    def local_inputs_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "inputs.zarr")

    @property
    def local_targets_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "targets.zarr")
