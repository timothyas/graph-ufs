import argparse
import logging
import os
import shutil
from functools import partial

import optax
from graphufs import (
    optimize,
    predict,
    DataGenerator,
    init_model,
    convert_wb2_format,
    compute_rmse_bias,
    init_devices,
)

from simple_emulator import P0Emulator


"""
Script to train and test graphufs over multiple chunks and epochs

Usage:

    python3 -W ignore train.py --num-epochs 2 --chunks-per-epoch 2 --latent-size 32 --training-dates "1994-01-01T00" "1994-01-31T18"

    This will train networks for 2 epochs with 2 chunks per epoch with training dataset of first month of 1994.
    You should get 4 models (checkpoints) after training completes.

    Later, you can evaluate a specific model by specifying model id, and testing dataset range i.e. first month of 1995

    python3 -W ignore train.py --chunks-per-epoch 2 --latent-size 32 --test --id 3 --testing-dates  "1995-01-01T00"  "1995-01-31T18"
"""


if __name__ == "__main__":

    # parse arguments
    gufs, args = P0Emulator.from_parser()

    # for multi-gpu training
    init_devices(gufs)

    # explicitly get normalization
    gufs.set_normalization()

    # data generators
    generator = DataGenerator(
        emulator=gufs,
        n_optim_steps=gufs.steps_per_chunk,
        mode="testing" if args.test else "training",
    )

    # validation
    if not args.test:
        generator_valid = DataGenerator(
            emulator=gufs,
            n_optim_steps=gufs.steps_per_chunk,
            mode="validation",
        )

    # load weights or initialize a random model
    if gufs.checkpoint_exists(args.id) and args.id >= 0:
        logging.info(f"Loading weights: {args.id}")
        params, state = gufs.load_checkpoint(args.id)
    else:
        logging.info("Initializing Optimizer and Parameters")
        data = generator.get_data()  # just to figure out shapes
        params, state = init_model(gufs, data)
        loss_name = f"{gufs.local_store_path}/loss.nc"
        if os.path.exists(loss_name):
            os.remove(loss_name)

    # training
    if not args.test:
        logging.info("Starting Training")

        optimizer = optax.adam(learning_rate=1e-4)

        # training loop
        for e in range(gufs.num_epochs):
            for c in range(gufs.chunks_per_epoch):
                logging.info(f"Training on epoch {e} and chunk {c}")

                # get chunk of data in parallel with NN optimization
                if gufs.chunks_per_epoch > 1:
                    generator.generate()
                    generator_valid.generate()
                data = generator.get_data()
                data_valid = generator_valid.get_data()

                # optimize
                params, loss = optimize(
                    params=params,
                    state=state,
                    optimizer=optimizer,
                    emulator=gufs,
                    training_data=data,
                    validation_data=data_valid,
                )

                # save weights
                if c % gufs.checkpoint_chunks == 0:
                    ckpt_id = (e * gufs.chunks_per_epoch + c) // gufs.checkpoint_chunks
                    gufs.save_checkpoint(params, ckpt_id)

    # testing
    else:
        logging.info("Starting Testing")

        # create predictions and targets zarr file for WB2
        predictions_zarr_name = f"{gufs.local_store_path}/graphufs_predictions.zarr"
        targets_zarr_name = f"{gufs.local_store_path}/graphufs_targets.zarr"
        if os.path.exists(predictions_zarr_name):
            shutil.rmtree(predictions_zarr_name)
        if os.path.exists(targets_zarr_name):
            shutil.rmtree(targets_zarr_name)

        stats = {}
        for c in range(gufs.chunks_per_epoch):
            logging.info(f"Testing on chunk {c}")

            # get chunk of data in parallel with inference
            generator.generate()
            data = generator.get_data()

            # run predictions
            predictions = predict(
                params=params,
                state=state,
                emulator=gufs,
                input_batches=data["inputs"],
                target_batches=data["targets"],
                forcing_batches=data["forcings"],
            )

            targets = data["targets"]
            inittimes = data["inittimes"]

            # Compute rmse and bias comparing targets and predictions
            compute_rmse_bias(predictions, targets, stats, c)

            # write predictions chunk by chunk to avoid storing all of it in memory
            predictions = convert_wb2_format(gufs, predictions, inittimes)
            predictions = predictions.dropna("time")
            predictions.to_zarr(predictions_zarr_name, append_dim="time" if c else None)

            # write also targets to compute metrics against it with wb2
            targets = convert_wb2_format(gufs, targets, inittimes)
            targets = targets.dropna("time")
            targets.to_zarr(targets_zarr_name, append_dim="time" if c else None)

        logging.info("--------- Statistiscs ---------")
        for k, v in stats.items():
            logging.info(f"{k:32s}: RMSE: {v[0]} BIAS: {v[1]}")
