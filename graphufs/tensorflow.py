from typing import Optional, Tuple, Generator
import numpy as np
import tensorflow as tf

from .dataset import Dataset as BaseDataset
from .emulator import ReplayEmulator

class Dataset(BaseDataset):
    """
    TensorFlow Dataset that works with StackedGraphCast

    Example:
        >>> ds = Dataset(p0, mode="training")
        >>> tfds = ds.get_dataset()
        >>> for i, (x,y) in enumerate(tfds):
        >>>     # loop through each batch of inputs x and targets y
    """

    def __init__(
        self,
        emulator: ReplayEmulator,
        mode: str,
        preload_batch: bool = False,
        shuffle: bool = False,
        rng_seed: Optional[int] = None,
        epoch_size: Optional[int] = None,
        drop_last: Optional[bool] = False,
    ) -> None:
        """
        Initializes the Dataset.

        Args:
            emulator (ReplayEmulator): The emulator object.
            mode (str): The mode of operation, e.g., "training" or "testing".
            preload_batch (bool, optional): Whether to preload batches. Defaults to False.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            rng_seed (int, optional): Random seed for shuffling. Defaults to None.
            epoch_size (int, optional): Size of the epoch. Defaults to all batches available.
            drop_last (bool, optional): If True then drop incomplete last batch. Defaults to False.
        """
        super().__init__(emulator, mode)

        self.epoch_size = len(self) // self.batch_size if epoch_size is None else epoch_size

        self.pointer = 0
        self.stop_at = len(self) if not drop_last else len(self) - len(self)%self.batch_size
        self.drop_last = drop_last
        self.sample_indices = np.arange(len(self))
        self.shuffle = shuffle
        if shuffle:
            self.rstate = np.random.RandomState(rng_seed)
            self.rstate.shuffle(self.sample_indices)

        # get a single sample to figure out size
        x, y = self.__getitem__(0)
        self.x_shape = (self.batch_size,) + x.shape
        self.y_shape = (self.batch_size,) + y.shape

    def generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generator function for creating batches.

        Yields:
            Generator[Tuple[np.ndarray, np.ndarray], None, None]: A tuple containing input and target batches.
        """
        for b in range(self.epoch_size):
            x_batch = []
            y_batch = []
            for s in range(self.batch_size):
                idx = self.sample_indices[self.pointer]
                xb, yb = self.__getitem__(int(idx))
                x_batch += [xb]
                y_batch += [yb]
                self.pointer += 1

                if self.pointer == self.stop_at:
                    self.pointer = 0
                    if self.shuffle:
                        self.rstate.shuffle(self.sample_indices)
                    break

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            yield x_batch, y_batch

    def get_dataset(self) -> tf.data.Dataset:
        """
        Creates a TensorFlow dataset from the generator.

        Returns:
            tf.data.Dataset: A TensorFlow dataset.
        """
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=self.x_shape, dtype=tf.float32),
                tf.TensorSpec(shape=self.y_shape, dtype=tf.float32),
            ),
        )
        # return numpy arrays not tensors for jax
        # this is slightly faster for our current storage anyway
        dataset = tf.data.NumpyIterator(dataset)

        # is this a bad idea?
        dataset.drop_last = self.drop_last
        dataset.__len__ = self.epoch_size
        dataset.epoch_size = self.epoch_size
        return dataset
