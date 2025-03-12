from mpi4py import MPI
from config import (
    GEFSDeviationTrainer as RemoteTrainer,
    GEFSDeviationPreprocessed as PackedTrainer,
)

from prototypes.gefs.train import train

if __name__ == "__main__":
    train(RemoteTrainer, PackedTrainer, missing_samples=[28295])
