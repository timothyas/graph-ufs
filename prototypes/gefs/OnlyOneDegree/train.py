from mpi4py import MPI
from config import (
    GEFSMSETrainer as RemoteMSETrainer,
    GEFSMSEPreprocessed as PackedMSETrainer,
    GEFSDeviationTrainer as RemoteDeviationTrainer,
    GEFSDeviationPreprocessed as PackedDeviationTrainer,
)

from prototypes.gefs.train import train

if __name__ == "__main__":
    cfg = train(RemoteMSETrainer, PackedMSETrainer, peak_lr=1e-3)
    #cfg = train(RemoteDeviationTrainer, PackedDeviationTrainer, peak_lr=1e-4, cfg=cfg)

