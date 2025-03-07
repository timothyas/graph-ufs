from mpi4py import MPI
from config import (
    GEFSForecastTrainer as RemoteForecastTrainer,
    GEFSForecastPreprocessed as PackedForecastTrainer,
    GEFSDeviationTrainer as RemoteDeviationTrainer,
    GEFSDeviationPreprocessed as PackedDeviationTrainer,
)

from prototypes.gefs.train import train

if __name__ == "__main__":
    cfg = train(RemoteForecastTrainer, PackedForecastTrainer, missing_samples=[59419])
    cfg = train(RemoteDeviationTrainer, PackedDeviationTrainer, missing_samples=[28295], cfg=cfg)

