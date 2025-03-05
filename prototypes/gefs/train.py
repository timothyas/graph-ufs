import os
import logging

from mpi4py import MPI

from graphufs.datasets import Dataset
from graphufs.tensorstore import PackedDataset as TSPackedDataset, MPIBatchLoader as TSBatchLoader
from graphufs.mpi import MPITopology

from graphufs.stacked_mpi_training import (
    init_model,
    optimize,
)

from graphufs.optim import clipped_cosine_adamw
from graphufs import utils

def train(RemoteEmulator, PackedEmulator, cfg=None):
    """

    Args:
        RemoteEmulator, PackedEmulator (graphufs.Emulator)
        cfg (dict, optional): if provided, a dict with
            topo, params, state, opt_state
    """

    # initial setup
    is_pickup = True
    if cfg is None:
        is_pickup = False
        cfg = dict()
        cfg["opt_state"] = None
        cfg["topo"] = MPITopology(log_dir=f"{RemoteEmulator.local_store_path}/logs/training")

    emulator = PackedEmulator(mpi_rank=cfg["topo"].rank, mpi_size=cfg["topo"].size)
    remote_emulator = RemoteEmulator(mpi_rank=cfg["topo"].rank, mpi_size=cfg["topo"].size)

    # data generators
    tds = Dataset(remote_emulator, mode="training")
    training_data = TSPackedDataset(emulator, mode="training")
    validation_data = TSPackedDataset(emulator, mode="validation")

    trainer = TSBatchLoader(
        training_data,
        batch_size=emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=emulator.num_workers,
        max_queue_size=emulator.max_queue_size,
        mpi_topo=cfg["topo"],
        rng_seed=10,
    )
    validator = TSBatchLoader(
        validation_data,
        batch_size=emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=emulator.num_workers,
        max_queue_size=emulator.max_queue_size,
        mpi_topo=cfg["topo"],
        rng_seed=11,
    )

    logging.info("Initializing Loss Function Weights and Stacked Mappings")
    # compute loss function weights once
    loss_weights = emulator.calc_loss_weights(tds)
    last_input_channel_mapping = utils.get_last_input_mapping(tds)

    # initialize a random model
    if not is_pickup:
        logging.info("Initializing Optimizer and Parameters")
        inputs, _ = trainer.get_data()
        cfg["params"], cfg["state"] = init_model(
            emulator=emulator,
            inputs=inputs,
            last_input_channel_mapping=last_input_channel_mapping,
            mpi_topo=cfg["topo"],
        )

    loss_name = f"{emulator.local_store_path}/loss.nc"
    if cfg["topo"].is_root:
        emulator.save_checkpoint(cfg["params"], id=0)
        if os.path.exists(loss_name):
            os.remove(loss_name)

    # setup optimizer
    steps_in_epoch = len(trainer)
    n_total = emulator.num_epochs * steps_in_epoch
    n_linear = emulator.n_linear_warmup_steps
    n_cosine = n_total - n_linear
    optimizer = clipped_cosine_adamw(
        n_linear=n_linear,
        n_total=n_total,
        peak_value=emulator.peak_lr,
    )

    logging.info(f"Starting Training with:")
    logging.info(f"\t batch_size = {emulator.batch_size}")
    logging.info(f"\t {len(trainer)} training steps per epoch")
    logging.info(f"\t {len(validator)} validation steps per epoch")
    logging.info(f"\t ---")
    logging.info(f"\t {n_linear} linearly increasing LR steps")
    logging.info(f"\t {n_cosine} cosine decay LR steps")
    logging.info(f"\t {n_total} total training steps")

    # training
    for e in range(emulator.num_epochs):
        logging.info(f"Starting epoch {e+1}")

        # optimize
        cfg["params"], loss, cfg["opt_state"] = optimize(
            params=cfg["params"],
            state=cfg["state"],
            optimizer=optimizer,
            emulator=emulator,
            trainer=trainer,
            validator=validator,
            loss_weights=loss_weights,
            last_input_channel_mapping=last_input_channel_mapping,
            opt_state=cfg["opt_state"],
            mpi_topo=cfg["topo"],
        )

        # save weights
        logging.info(f"Done with epoch {e+1}")
        if cfg["topo"].is_root:
            emulator.save_checkpoint(cfg["params"], id=e+1)

    logging.info("Done Training")
    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)

    return cfg
