import logging
import os
import sys

from graphufs import (
    DataGenerator,
    init_model,
    save_checkpoint,
)

from p1 import P1Emulator


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )

    # parse arguments
    p1, args = P1Emulator.from_parser()

    # 1. Read remote normalization, store locally, and set to p1
    p1.set_normalization()

    # 2. Pull the training data and store to data/data.zarr
    p1.get_the_data(mode="training")
