# Training data: the same tokenized files the autoregressive models use (data/<dataset>/train.bin and
# validation.bin, GPT-2 token ids, documents separated by <|endoftext|>). A batch is random windows of seq_len
# tokens, as in pretraining; there's no target shifted by one, the noising makes the targets.
#
# v1 tokenized every story in Python as it trained; reading the token files is much faster and works for any
# dataset the project can tokenize (TinyStories now, Mini FineWeb-Edu next).

import os
import shutil

import torch

from data import BatchLoader, dataset_paths, get_dataset
from tokenization import bin_files_exist, tokenize_and_save

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
TINYSTORIES = ("roneneldan/TinyStories", None)
FINEWEB_EDU_SMALL = ("BaoMinhTran/mini-fineweb-edu", "small")


def ensure_tokenized(name=TINYSTORIES[0], config=None, data_root=DATA_ROOT, cache_dir=None, say=print):
    """The dataset's folder with train.bin and validation.bin, downloading and tokenizing it if needed.

    cache_dir (e.g. a Google Drive folder): if it already holds the two token files, copy them instead of
    tokenizing again; after tokenizing, save a copy there for next time.
    """
    paths = dataset_paths(name, config, data_root)
    files = ("train.bin", "validation.bin")
    cache = os.path.join(cache_dir, os.path.basename(paths.root)) if cache_dir else None
    if not bin_files_exist(paths.root) and cache and all(os.path.exists(os.path.join(cache, f)) for f in files):
        os.makedirs(paths.root, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(cache, f), os.path.join(paths.root, f))
        say(f"Copied the token files from {cache}")
    if not bin_files_exist(paths.root):
        ds = get_dataset(paths.hf_dir, name, config_name=config, text_column="text")
        tokenize_and_save(ds, paths.root, num_proc=os.cpu_count())
        if cache:
            os.makedirs(cache, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(paths.root, f), os.path.join(cache, f))
            say(f"Saved a copy of the token files to {cache}")
    return paths.root


class BlockLoader:
    """Random windows of seq_len tokens from one or more tokenized datasets."""

    def __init__(self, roots, seq_len, batch_size, device):
        self.loader = BatchLoader(roots, seq_len, batch_size, device)
        self.device = device

    def batch(self, split="train"):
        x, _ = self.loader.get_batch("train" if split == "train" else "val")
        return x

    def fixed_batches(self, n, split="val", seed=1234):
        """The same n batches every time (for evaluation that can be compared across checkpoints)."""
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        out = [self.batch(split) for _ in range(n)]
        torch.set_rng_state(state)
        return out
