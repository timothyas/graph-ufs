"""
Same as the LocalDataset and BatchLoader, but using xarray_tensorstore
"""
import numpy as np
import logging
import xarray_tensorstore

from .datasets import PackedDataset as BaseDataset
from .batchloader import BatchLoader as BaseBatchLoader, MPIBatchLoader as BaseMPIBatchLoader
from .mpi import MPITopology, _has_mpi

class PackedDataset(BaseDataset):
    """Same as the other PackedDatset, but use xarray_tensorstore instead of xarray/dask/zarr
    """

    def __init__(self, emulator, mode):
        self.emulator = emulator
        self.mode = mode
        self.inputs = xarray_tensorstore.open_zarr(self.local_inputs_path)
        self.targets = xarray_tensorstore.open_zarr(self.local_targets_path)

    def __getitem__(self, idx):
        if isinstance(idx, int):

            x = xarray_tensorstore.read(self.inputs["inputs"].sel(sample=idx))
            y = xarray_tensorstore.read(self.targets["targets"].sel(sample=idx))

        else:
            x = [xarray_tensorstore.read(self.inputs["inputs"].sel(sample=i)) for i in idx]
            y = [xarray_tensorstore.read(self.targets["targets"].sel(sample=i)) for i in idx]

        return x, y


class BatchLoader(BaseBatchLoader):

    def _next_data(self):

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            x, y = self.dataset[batch_indices]
            x = np.vstack([xi.values[None] for xi in x])
            y = np.vstack([yi.values[None] for yi in y])
            return x, y
        else:
            raise StopIteration


class ExpandedBatchLoader(BaseBatchLoader):
    def _next_data(self):

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            data = self.dataset.get_batch_of_xarrays(batch_indices)
            return tuple(d.compute() for d in data)
        else:
            raise StopIteration


class MPIBatchLoader(BaseMPIBatchLoader):

    def _next_data(self):
        if self.data_counter < len(self):
            st = (self.data_counter * self.batch_size) + self.local_batch_index
            ed = st + self.data_per_device
            batch_indices = self.sample_indices[st:ed]

            x, y = self.dataset[batch_indices]
            x = np.vstack([xi.values[None] for xi in x])
            y = np.vstack([yi.values[None] for yi in y])
            return x, y
        else:
            raise StopIteration
