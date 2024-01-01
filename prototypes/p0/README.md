# Prototype 0

Train a GraphCast-like emulator on Replay data with the following
specifications:
- Reduced variable set
- 3 vertical levels
- 6 hour time step
- 1 year of training data
- 1 year of evaluation data
- Evaluate on and with WeatherBench2
- 10 day forecasts
- data normalized based on avg/std taken over 1994-1997

The configuration is defined in `simple_emulator.py`, and the training can be
run with the script `run_training.py` as

```bash
python run_training.py
```

This code requires [this fork and branch](https://github.com/NOAA-PSL/graphcast/tree/feature/replay-integration)
of GraphCast, and uses the environment file at
[../../conda/gpu-workaround.yaml](../../conda/gpu-workaround.yaml).

## Notes

### Normalization

The normalization fields were computed using `calc_normalization.py`, and there
are many unnecessary hard coded values.
This code should be generalized in the future, and could probably be more
efficient, e.g. with a dask cluster rather than brute force slurm job
submission.
Some points of generalization include:
- storage location input/output (which would generalize resolution and model
  component)
- time frame and data frequency (timestep) used
- the averaging may need to take into account a more accurate grid cell volume
  weighted average
- the `year_progress` and `day_progress` fields are computed by graphcast, and
  these calculations should go here so that the correct normalization can be
  computed. Right now those normalization values are hard coded into
  `graphufs.ReplayEmulator.load_normalization`

### Emulator class

This holds all of the configuration details, loads the normalization data, and preprocesses the data for training. See the generic class definition `graphufs.ReplayEmulator` and this specific setup in `simple_emulator.py`. This configuration could be rewritten to read a yaml file rather than to specify everything via class inheritance.

### Training

The main training algorithm is launched from `run_training.py` which calls the
generic routines from the module `graphufs.train`.
Those routines are modified from the graphcast demo to work with the
`ReplayEmulator` class and to work with the optax adam optimizer.

This is just at the point of compiling and running, not at the point of actually
successfully training (converging).
The main issue that needs to be addressed is how to handle mini-batches in the
training.
Currently, the `ReplayEmulator.get_training_batches` uses routines from the
graphcast code to get the samples prepared to make predictions with GraphCast,
but it is unclear how to set up batches with more than one forecast sample
dataset (i.e., initial conditions and target predictions for that forecast) per
batch.
It is then unclear how this is handled with optax, although it could be that
optax can do this more automatically given a bunch of samples of data.

Once this is figured out, it is a matter of hyperparameter tuning and making the
code more efficient (e.g., the zarr2zarr transfer happens inside the graphcast
code, and we should just have this dataset ready).



