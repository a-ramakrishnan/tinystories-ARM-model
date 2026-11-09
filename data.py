# Dataset download and batch sampling
# Uses Tiny Stories Data set -> https://huggingface.co/datasets/roneneldan/TinyStories

import os

import numpy as np
import torch
from datasets import load_dataset, load_from_disk

DATASET_NAME = "roneneldan/TinyStories"


def get_dataset(local_dir, name=DATASET_NAME):
    """Return the TinyStories dataset, downloading it only once.

    The first run downloads it from Hugging Face and saves a copy to `local_dir`
    (git-ignored). Later runs load that local copy without touching the network.
    """
    if os.path.isdir(local_dir):
        print(f"Loading dataset from local copy: {local_dir}")
        return load_from_disk(local_dir)
    print(f"Downloading {name} and saving it to {local_dir}")
    ds = load_dataset(name)
    ds.save_to_disk(local_dir)
    return ds


class BatchLoader:
    """Samples random (input, target) blocks from the tokenized train/validation .bin files.

    Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications.
    block_size = context window.
    """

    def __init__(self, data_dir, block_size, batch_size, device):
        self.paths = {
            "train": os.path.join(data_dir, "train.bin"),
            "val": os.path.join(data_dir, "validation.bin"),
        }
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else str(device)

    def get_batch(self, split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data = np.memmap(self.paths[split], dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
