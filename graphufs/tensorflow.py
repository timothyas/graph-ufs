import numpy as np
import tensorflow as tf

class Dataset(BaseDataset):
    """
    TensorFlow Dataset that works with StackedGraphCast

    Example:
        >>> ds = Dataset(p0, mode="training")
        >>> tfds = ds.get_dataset()
        >>> for i, (x,y) in enumerate(tfds):
        >>>     # loop through each batch of inputs x and targets y
    """
    def __init__(self, emulator, mode, preload_batch=False, shuffle=False, rng_seed=None, epoch_size=None):
        super().__init__(emulator, mode)

        self.epoch_size = len(self) // self.batch_size if epoch_size is None else epoch_size

        self.pointer = 0
        self.sample_indices = np.arange(len(self))
        self.shuffle = shuffle
        if shuffle:
            self.rstate = np.random.RandomState(rng_seed)
            self.rstate.shuffle(self.sample_indices)

        # get a single sample to figure out size
        x,y = self.__getitem__(0)
        self.x_shape = (self.batch_size,) + x.shape
        self.y_shape = (self.batch_size,) + y.shape

    def generator(self):
        for b in range(self.epoch_size):
            x_batch = []
            y_batch = []
            for s in range(self.batch_size):
                idx = self.sample_indices[self.pointer]
                xb, yb = self.__getitem__(int(idx))
                x_batch += [xb]
                y_batch += [yb]
                self.pointer+=1

                if self.pointer == len(self):
                    self.pointer = 0
                    if self.shuffle:
                        rstate.shuffle(self.sample_indices)
                    break

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            yield x_batch, y_batch

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=self.x_shape, dtype=tf.float32),
                tf.TensorSpec(shape=self.y_shape, dtype=tf.float32),
            ),
        )
        return dataset
