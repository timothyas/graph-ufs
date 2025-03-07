from mpi4py import MPI
import logging
import xarray as xr
import numpy as np

from config import GEFSForecastPreprocessed, GEFSDeviationPreprocessed
from graphufs.tensorstore import PackedDataset, MPIBatchLoader
from graphufs.mpi import MPITopology

if __name__ == "__main__":

    topo = MPITopology(log_dir="/global/homes/t/timothys/debug-nans")
    emulator = GEFSDeviationPreprocessed(mpi_rank=topo.rank, mpi_size=topo.size)
    for mode in ["training", "validation"]:
        ds = PackedDataset(emulator, mode=mode)
        bl = MPIBatchLoader(ds, batch_size=topo.size, shuffle=False, num_workers=1, mpi_topo=topo)
        logging.info(f" --- Starting {mode} --- ")
        for k, (x,y) in enumerate(bl):
            for array, name in zip([x,y], ["inputs", "targets"]):
                array = np.isnan(array)
                any_nans = np.any(array)
                all_nans = np.all(array)
                if any_nans or all_nans:
                    idx = k*bl.batch_size + bl.local_batch_index
                    logging.info(f"{name}: Batch {k:04d}, Sample {idx:05d}: all = {all_nans}, any = {any_nans}")
        logging.info(f" --- Done with {mode} --- ")
