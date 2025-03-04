from mpi4py import MPI
from config import (
    GEFSForecastTrainer as RemoteForecastTrainer,
    GEFSForecastPreprocessed as PackedForecastTrainer,
    GEFSDeviationTrainer as RemoteDeviationTrainer,
    GEFSDeviationPreprocessed as PackedDeviationTrainer,
)

from prototypes.gefs.train import train

if __name__ == "__main__":
    cfg = train(RemoteForecastTrainer, PackedForecastTrainer, peak_lr=1e-3)
    #cfg = train(RemoteDeviationTrainer, PackedDeviationTrainer, peak_lr=1e-4, cfg=cfg)

