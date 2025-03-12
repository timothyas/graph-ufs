from mpi4py import MPI
from config import (
    GEFSForecastTrainer as RemoteTrainer,
    GEFSForecastPreprocessed as PackedTrainer,
)

from prototypes.gefs.train import train

if __name__ == "__main__":
    train(RemoteTrainer, PackedTrainer, missing_samples=[59419])
