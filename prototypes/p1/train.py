import os
import logging

import optax
from graphufs import (
    optimize,
    DataGenerator,
    init_devices,
)

from p1 import P1Emulator

def graphufs_optimizer(
    n_linear,
    n_total,
    peak_value=1e-3,
    ):

    # define learning rate schedules
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_value,
        warmup_steps=n_linear,
        decay_steps=n_total,
        end_value=0.0,
    )

    # Adam optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(32),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=0.1,
        ),
    )
    return optimizer



if __name__ == "__main__":

    # parse arguments
    p1, args = P1Emulator.from_parser()

    # for multi-gpu training
    init_devices(p1)

    # data generators
    trainer = DataGenerator(
        emulator=p1,
        n_optim_steps=p1.steps_per_chunk,
        mode="testing" if args.test else "training",
    )

    validator = DataGenerator(
        emulator=p1,
        n_optim_steps=p1.steps_per_chunk,
        mode="validation",
    )

    # load weights or initialize a random model
    logging.info(f"Loading weights: {0}")
    params, state = p1.load_checkpoint(0)
    loss_name = f"{p1.local_store_path}/loss.nc"
    if os.path.exists(loss_name):
        os.remove(loss_name)

    n_linear = 1_000
    n_total = p1.num_epochs * p1.chunks_per_epoch * p1.steps_per_chunk
    n_cosine = n_total - n_linear
    optimizer = graphufs_optimizer(
        n_linear=n_linear,
        n_total=n_total,
        peak_value=1e-3,
    )

    # training loop
    logging.info(f"Starting Training with:")
    logging.info(f"\t {n_linear} linearly increasing LR steps")
    logging.info(f"\t {n_cosine} cosine decay LR steps")
    opt_state = None
    for e in range(p1.num_epochs):
        for c in range(p1.chunks_per_epoch):
            logging.info(f"Training on epoch {e} and chunk {c}")

            # get chunk of data in parallel with NN optimization
            if p1.chunks_per_epoch > 1:
                trainer.generate()
                validator.generate()
            data = trainer.get_data()
            data_valid = validator.get_data()

            # optimize
            params, loss, opt_state = optimize(
                params=params,
                state=state,
                optimizer=optimizer,
                emulator=p1,
                training_data=data,
                validation_data=data_valid,
                opt_state=opt_state,
            )

            # save weights
            if c % p1.checkpoint_chunks == 0:
                ckpt_id = (e * p1.chunks_per_epoch + c) // p1.checkpoint_chunks
                p1.save_checkpoint(params, ckpt_id)
