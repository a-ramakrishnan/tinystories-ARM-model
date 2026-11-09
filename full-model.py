# Building a Small Language Model from Scratch, made for Apple Silicon Macs
# Default dataset: Tiny Stories -> https://huggingface.co/datasets/roneneldan/TinyStories
# Ramakrishnan A
# 9th Nov 2026
#
# Entry point: runs the whole pipeline.
#   data.py          - download a dataset once (stored locally) and sample training batches
#   tokenization.py  - GPT-2 tokenization into train.bin / validation.bin
#   model.py         - model architecture and loss estimation
#   train_config.py  - training hyperparameters and device setup (Apple GPU via MPS)
#   pretrain.py      - pretraining loop
#   plots.py         - loss curves
#   inference.py     - load the best checkpoint and generate text
#
# Run:  python3 full-model.py
#       python3 full-model.py --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1
# Web front end: python3 app/server.py

import argparse
import os
import time
from dataclasses import asdict

import torch

from data import DEFAULT_DATASET, BatchLoader, dataset_paths, get_dataset
from inference import generate_text, load_model, save_model_config
from model import GPT, GPTConfig
from plots import plot_losses
from pretrain import pretrain
from tokenization import bin_files_exist, get_tokenizer, tokenize_and_save
from train_config import TrainConfig, setup_device

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")  # one folder per dataset (git-ignored)

# Model config
config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=128,       # context window; also the length of each training sample
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

PROMPTS = [
    "Once upon a time there was a lady who .",
    "A monster lay lurking in the woods ",
]


def main(train_cfg=None, dataset=DEFAULT_DATASET, dataset_config=None, text_column="text"):
    paths = dataset_paths(dataset, dataset_config, DATA_ROOT)
    if train_cfg is None:
        train_cfg = TrainConfig(best_model_path=paths.checkpoint)

    # 1. Dataset: downloaded once, then loaded from the local copy
    ds = get_dataset(paths.hf_dir, dataset, config_name=dataset_config, text_column=text_column)

    # 2. Tokenize (only if the .bin files are missing)
    enc = get_tokenizer()
    if not bin_files_exist(paths.root):
        tokenize_and_save(ds, paths.root)

    # 3. Device and batches (sampled fresh every run)
    dev = setup_device()
    print(f"Training on device: {dev.device} ({dev.dtype})")
    torch.manual_seed(train_cfg.seed)
    loader = BatchLoader(paths.root, config.block_size, train_cfg.batch_size, dev.device)

    # 4. Model + pretraining
    model = GPT(config).to(dev.device)
    run_info = {"dataset": dataset, "dataset_config": dataset_config}
    save_model_config(paths.model_config, config, run_info)
    train_losses, val_losses = pretrain(model, loader, train_cfg, dev)
    save_model_config(paths.model_config, config, {
        **run_info, "steps": train_cfg.max_iters, "best_val": min(val_losses) if val_losses else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"), "train_config": asdict(train_cfg)})

    # 5. Plots
    plot_losses(train_losses, val_losses, train_cfg.eval_interval)

    # 6. Inference with the best checkpoint
    model = load_model(config, train_cfg.best_model_path, dev.device)
    for prompt in PROMPTS:
        print(generate_text(model, enc, prompt, dev.device))


def parse_args():
    p = argparse.ArgumentParser(description="Train a small GPT on a text dataset. Made for Apple Silicon.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="Hugging Face dataset name or link, or a link to a .txt/.jsonl/.json/.csv/.parquet file "
                        f"(default: {DEFAULT_DATASET})")
    p.add_argument("--dataset-config", default=None,
                   help="Dataset subset/config name, if it has one (e.g. wikitext-2-raw-v1)")
    p.add_argument("--text-column", default="text",
                   help="Column that holds the text (default: text)")
    p.add_argument("--list-datasets", action="store_true",
                   help="List public text datasets that work out of the box, then exit")
    return p.parse_args()


def list_datasets():
    from catalog import DATASET_CATALOG
    for c in DATASET_CATALOG:
        args = f"--dataset {c['dataset']}"
        if c["config"]:
            args += f" --dataset-config {c['config']}"
        if c["text_column"] != "text":
            args += f" --text-column {c['text_column']}"
        print(f"{c['title']}  ({c['rows']}, {c['size']})\n  {c['description']}\n  python3 full-model.py {args}\n")


if __name__ == "__main__":
    args = parse_args()
    if args.list_datasets:
        list_datasets()
    else:
        main(dataset=args.dataset, dataset_config=args.dataset_config, text_column=args.text_column)
