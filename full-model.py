# Building a Small Language Model from Scratch
# Uses Tiny Stories Data set -> https://huggingface.co/datasets/roneneldan/TinyStories
# Ramakrishnan A
# 9th Nov 2026
#
# Entry point: runs the whole pipeline.
#   data.py          - download the dataset once (stored locally) and sample training batches
#   tokenization.py  - GPT-2 tokenization into train.bin / validation.bin
#   model.py         - model architecture and loss estimation
#   train_config.py  - training hyperparameters and device setup
#   pretrain.py      - pretraining loop
#   plots.py         - loss curves
#   inference.py     - load the best checkpoint and generate text
#
# Run: python3 full-model.py

import os

import torch

from data import BatchLoader, get_dataset
from inference import generate_text, load_model
from model import GPT, GPTConfig
from plots import plot_losses
from pretrain import pretrain
from tokenization import bin_files_exist, get_tokenizer, tokenize_and_save
from train_config import TrainConfig, setup_device

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "data", "tinystories")  # local dataset copy (git-ignored)

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


def main(train_cfg=None):
    if train_cfg is None:
        train_cfg = TrainConfig(best_model_path=os.path.join(PROJECT_DIR, "best_model_params.pt"))

    # 1. Dataset: downloaded once, then loaded from the local copy
    ds = get_dataset(DATASET_DIR)

    # 2. Tokenize (only if the .bin files are missing)
    enc = get_tokenizer()
    if not bin_files_exist(PROJECT_DIR):
        tokenize_and_save(ds, PROJECT_DIR)

    # 3. Device and batches (sampled fresh every run)
    dev = setup_device()
    print(f"Training on device: {dev.device} ({dev.dtype})")
    torch.manual_seed(train_cfg.seed)
    loader = BatchLoader(PROJECT_DIR, config.block_size, train_cfg.batch_size, dev.device)

    # 4. Model + pretraining
    model = GPT(config).to(dev.device)
    train_losses, val_losses = pretrain(model, loader, train_cfg, dev)

    # 5. Plots
    plot_losses(train_losses, val_losses, train_cfg.eval_interval)

    # 6. Inference with the best checkpoint
    model = load_model(config, train_cfg.best_model_path, dev.device)
    for prompt in PROMPTS:
        print(generate_text(model, enc, prompt, dev.device))


if __name__ == "__main__":
    main()
