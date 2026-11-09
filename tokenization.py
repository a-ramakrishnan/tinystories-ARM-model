# Tokenization: GPT-2 BPE via tiktoken, written to train.bin / validation.bin
# Some functions from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

import os

import numpy as np
import tiktoken
from tqdm.auto import tqdm

# Module-level so worker processes (num_proc > 1) can rebuild it on import
_ENC = tiktoken.get_encoding("gpt2")


def get_tokenizer():
    """Return the GPT-2 tokenizer (vocab size 50257)."""
    return _ENC


def bin_files_exist(data_dir):
    """True if both tokenized splits are already on disk."""
    return all(os.path.exists(os.path.join(data_dir, f)) for f in ("train.bin", "validation.bin"))


def process(example):
    ids = _ENC.encode_ordinary(example["text"])  # encode_ordinary ignores any special tokens
    return {"ids": ids, "len": len(ids)}


def tokenize_and_save(ds, data_dir, num_proc=8, total_batches=1024, progress=None):
    """Tokenize every split of `ds` and write each to <data_dir>/<split>.bin as uint16 token ids.

    progress: optional callback(split, done, total) called while each .bin file is written.
    Files are written as .tmp and renamed when complete, so an interrupted run is redone next time.
    """
    os.makedirs(data_dir, exist_ok=True)
    tokenized = ds.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(data_dir, f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename + ".tmp", dtype=dtype, mode="w+", shape=(arr_len,))

        n_batches = max(1, min(total_batches, len(dset)))  # small splits: avoid empty shards
        idx = 0
        for batch_idx in tqdm(range(n_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=n_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
            if progress is not None:
                progress(split, batch_idx + 1, n_batches)
        arr.flush()
        del arr
        os.replace(filename + ".tmp", filename)
