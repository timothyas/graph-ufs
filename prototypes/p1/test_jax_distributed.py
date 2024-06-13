import jax
import os

import xarray as xr


jax.distributed.initialize(
    coordinator_address="192.168.122.1:8892",
    num_processes=2,
    local_device_ids=range(4),
)



if __name__ == "__main__":

    _n_local = 2

    slurm_stuff = [
        "SLURM_JOB_ID",
        "SLURM_STEP_NODELIST",
        "SLURM_NTASKS",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_STEP_NUM_NODES",
        "CUDA_VISIBLE_DEVICES",
    ]

    for key in slurm_stuff:
        print(key, os.environ.get(key, "N/A"))

    print()
    print("jax.devices ... ", jax.devices())
    print("jax.local_devices ... ", jax.local_devices())

    # sharding
    sharding = jax.sharding.PositionalSharding(jax.local_devices())
    print("sharding: ")
    print(sharding)
    print()

    my_id = int(os.environ.get("SLURM_PROCID"))
    sharding = sharding.reshape((_n_local,1,1,1))
    print("my_id = ", my_id)
    print("sharding: ")
    print(sharding)
    print()


    # simple case
    print(" --- simple test --- ")
    xs = jax.numpy.ones(jax.local_device_count())
    print("running jax.pmap ...")
    print(jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(xs))
    print()
    print("done!")

    ds = xr.open_zarr("/lustre/stacked-p1-data-1year/training/inputs.zarr")
    st = _n_local*my_id
    ed = _n_local*(my_id+1)
    myslice = slice(st,ed)
    print("myslice: ")
    print(myslice)
    x = ds.inputs.isel(sample=myslice).values
    x = jax.device_put(x, sharding)
    print("x")
    print(x)
    print()

    loss1 = jax.pmap(lambda z: jax.lax.psum(z, "i"), axis_name="i")(x)
    print("loss1: ", loss1)
    print(loss1.shape)
    print()
    #loss = jax.pmap(lambda z: jax.lax.psum(z, ["i", "j", "k", "l"]), axis_name=("i","j", "k", "l"))(x)
    #print("loss: ", loss)
    #print(loss.shape)
    #print()
