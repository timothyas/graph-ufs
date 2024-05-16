# Notes on using TensorFlow data loader

Aside from just being weird to use, these are some issues that exist with using
a tensorflow dataloader.

Current status:
* Cannot use bfloat16 casting that's done with GraphCast
* Getting this very strange error
  ```python
  Exception ignored in: <function AtomicFunction.__del__ at 0x7f9e5410d800>
  Traceback (most recent call last):
    File
  "/home/tsmith/miniconda3/envs/graphufs3/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py",
  line 291, in __del__
  TypeError: 'NoneType' object is not subscriptable
  ```
  which is an [open issue](https://github.com/tensorflow/datasets/issues/5355) on TensorFlow


## Some timing just using the data loader

* ~15 sec per batch: load after all preprocessing, at the end of `_stack` where .values is called
* ~9-10 sec per batch: load at the end of `get_xds`
* ~8-9 sec per batch: `preload_batch=True`

Now using `preload_batch=True`, `dask.config.set(scheduler="threads", num_workers=..`:
* 2, ~13-15sec per batch
* 4, ~8-10sec per batch
* 8, ~7-8sec per batch
* 16, 6-7sec per batch
* 32, 6-7 sec per batch

Now using prefetch, no dask settings
* ~7-8 sec per batch
