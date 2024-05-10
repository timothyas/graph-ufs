"""
Implementations of Torch Dataset and DataLoader
"""
from typing import Optional
import numpy as np

from jax.tree_util import tree_map
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import default_collate

from .dataset import Dataset as BaseDataset

class Dataset(BaseDataset, TorchDataset):
    """
    PyTorch dataset for Replay Data that should work with GraphCast
    """

class DataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = collate_fn
        super().__init__(*args, **kwargs)


def collate_fn(batch):
    return tree_map(np.asarray, default_collate(batch))
