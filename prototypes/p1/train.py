import logging
import os
import sys

from graphufs import DataGenerator, init_model, init_devices
from p1 import P1Emulator


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )

    # parse arguments
    p1, args = P1Emulator.from_parser()
    init_devices(p1)

    # 1. Read remote normalization, store locally, and set to p1
    p1.set_normalization()

    # 2. Pull the training and validation data and store to data/data.zarr
    logging.info("Downloading Training Data")
    p1.get_the_data(mode="training")
    logging.info("Downloading Validation Data")
    p1.get_the_data(mode="validation")

    # 3. Initialize the model and store
    logging.info("Initializing Model & Storing Weights")
    gen = p1.get_batches(
        n_optim_steps=1,
        mode="training",
    )
    inputs, targets, forcings, inittimes = next(gen)
    data = {
        "inputs": inputs.load(),
        "targets": targets.load(),
        "forcings": forcings.load(),
        "inittimes": inittimes.load(),
    }
    params, state = init_model(p1, data)

    # TODO: make sure "state" is actually empty
    p1.save_checkpoint(params, id=0)
