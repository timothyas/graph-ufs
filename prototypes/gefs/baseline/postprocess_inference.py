from graphufs.log import setup_simple_log
from prototypes.gefs.postprocess_inference import compute_ensemble_mean

if __name__ == "__main__":
    setup_simple_log()
    compute_ensemble_mean(
        open_path="/pscratch/sd/t/timothys/gefs/one-degree/forecasts.validation.zarr",
        store_path="/pscratch/sd/t/timothys/gefs/one-degree/ensemble-mean.validation.zarr",
    )
