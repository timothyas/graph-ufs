from mpi4py import MPI
from config import (
    GEFSMSETrainer,
    GEFSDeviationTrainer,
)

from prototypes.gefs.train import train

if __name__ == "__main__":
    cfg = train(GEFSMSETrainer, peak_lr=1e-3)
    #cfg = train(GEFSDeviationTrainer, peak_lr=1e-4, cfg=cfg)

