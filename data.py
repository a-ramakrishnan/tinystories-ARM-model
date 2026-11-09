# Dataset download and batch sampling
# Default: Tiny Stories Data set -> https://huggingface.co/datasets/roneneldan/TinyStories
# Any Hugging Face text dataset with a train split and a text column works.

import os
import re
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_dataset_builder, load_from_disk

DEFAULT_DATASET = "roneneldan/TinyStories"
VALIDATION_SPLIT_NAMES = ("validation", "valid", "val", "test")


def dataset_slug(name, config_name=None):
    """Folder-safe name for a dataset, e.g. 'roneneldan/TinyStories' -> 'roneneldan__tinystories'."""
    full = f"{name}-{config_name}" if config_name else name
    return re.sub(r"[^A-Za-z0-9._-]+", "__", full).lower()


@dataclass
class DatasetPaths:
    """Where everything for one dataset lives: data/<slug>/..."""
    root: str

    @property
    def hf_dir(self):          # local copy of the downloaded dataset
        return os.path.join(self.root, "dataset")

    @property
    def train_bin(self):
        return os.path.join(self.root, "train.bin")

    @property
    def val_bin(self):
        return os.path.join(self.root, "validation.bin")

    @property
    def checkpoint(self):      # best weights from training on this dataset
        return os.path.join(self.root, "best_model_params.pt")

    @property
    def model_config(self):    # model config + training summary for the checkpoint
        return os.path.join(self.root, "model_config.json")


def dataset_paths(name=DEFAULT_DATASET, config_name=None, data_root="data"):
    return DatasetPaths(os.path.join(data_root, dataset_slug(name, config_name)))


def _normalize_splits(raw, text_column, val_fraction, seed):
    """Return a DatasetDict with exactly 'train' and 'validation' splits and a single 'text' column."""
    if isinstance(raw, Dataset):
        raw = DatasetDict(train=raw)
    if "train" not in raw:
        raise ValueError(f"Dataset has no 'train' split (found: {', '.join(raw)})")

    train = raw["train"]
    val_name = next((s for s in VALIDATION_SPLIT_NAMES if s in raw), None)
    if val_name:
        val = raw[val_name]
    else:
        # No validation split: hold out a small slice of train
        parts = train.train_test_split(test_size=val_fraction, seed=seed)
        train, val = parts["train"], parts["test"]

    def keep_text(d):
        if text_column not in d.column_names:
            raise ValueError(f"Column '{text_column}' not found (available: {', '.join(d.column_names)})")
        d = d.select_columns([text_column])
        return d.rename_column(text_column, "text") if text_column != "text" else d

    return DatasetDict(train=keep_text(train), validation=keep_text(val))


def get_dataset(local_dir, name=DEFAULT_DATASET, config_name=None, text_column="text", val_fraction=0.005, seed=42):
    """Return the dataset as {'train', 'validation'} splits with a 'text' column, downloading it only once.

    The first run downloads it from Hugging Face and saves a copy to `local_dir`
    (git-ignored). Later runs load that local copy without touching the network.
    `config_name` is the dataset's subset, if it has one (e.g. 'wikitext-2-raw-v1').
    `text_column` is renamed to 'text'. If there is no validation split, `val_fraction` of train is held out.
    """
    if os.path.isdir(local_dir):
        print(f"Loading dataset from local copy: {local_dir}")
        return load_from_disk(local_dir)
    # Check the text column from the dataset's metadata before downloading everything
    features = load_dataset_builder(name, config_name).info.features
    if features is not None and text_column not in features:
        raise ValueError(f"Column '{text_column}' not found (available: {', '.join(features)})")
    print(f"Downloading {name}{f' ({config_name})' if config_name else ''} and saving it to {local_dir}")
    raw = load_dataset(name, config_name)
    ds = _normalize_splits(raw, text_column, val_fraction, seed)
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
