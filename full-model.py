# Building a Small Language Model from Scratch, made for Apple Silicon Macs
# Default dataset: Mini FineWeb-Edu ('small' subset) -> https://huggingface.co/datasets/BaoMinhTran/mini-fineweb-edu
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
import json
import math
import os
import time
from dataclasses import asdict

import torch

from data import DEFAULT_CONFIG, DEFAULT_DATASET, BatchLoader, dataset_paths, get_dataset, mix_paths, parse_dataset_spec
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
    block_size=256,       # context window; also the length of each training sample (fits most stories)
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,          # no dropout: with this much data the model underfits rather than memorizes
    bias=True
)

PROMPTS = [
    "The water cycle is",
    "In the history of science,",
]


def main(train_cfg=None, dataset=DEFAULT_DATASET, dataset_config=DEFAULT_CONFIG, text_column="text", datasets=None, mix="size"):
    """Run the pipeline. `datasets` is an optional list of (name, subset, text_column) to train on a mix;
    otherwise the single `dataset` is used. `mix` is "size" (in proportion to tokens) or "equal"."""
    specs = datasets or [(dataset, dataset_config, text_column)]
    enc = get_tokenizer()

    # 1-2. Each dataset: downloaded once and kept locally, then tokenized once (skipped if the .bin files exist)
    roots = []
    for name, subset, column in specs:
        dpaths = dataset_paths(name, subset, DATA_ROOT)
        ds = get_dataset(dpaths.hf_dir, name, config_name=subset, text_column=column)
        if not bin_files_exist(dpaths.root):
            tokenize_and_save(ds, dpaths.root)
        roots.append(dpaths.root)
    paths = mix_paths(roots, DATA_ROOT)  # one dataset: its own folder; several: data/mix-.../
    os.makedirs(paths.root, exist_ok=True)
    if train_cfg is None:
        train_cfg = TrainConfig(best_model_path=paths.checkpoint)

    # 3. Device and batches (sampled fresh every run)
    dev = setup_device()
    print(f"Training on device: {dev.device} ({dev.dtype})")
    if len(roots) > 1:
        print(f"Training on {len(roots)} datasets, mixed {'equally' if mix == 'equal' else 'by size'}")
    torch.manual_seed(train_cfg.seed)
    loader = BatchLoader(roots, config.block_size, train_cfg.batch_size, dev.device, mix=mix)

    # 4. Model + pretraining
    model = GPT(config).to(dev.device)
    run_info = {"datasets": [s[0] for s in specs], "mix": mix if len(specs) > 1 else None}
    save_model_config(paths.model_config, config, run_info)
    t0 = time.time()
    train_losses, val_losses = pretrain(model, loader, train_cfg, dev)
    updates = math.ceil(train_cfg.max_iters / train_cfg.gradient_accumulation_steps)
    n_params = sum(p.numel() for p in model.parameters())
    summary = {**run_info, "steps": train_cfg.max_iters, "updates": updates, "params": n_params,
               "param_updates": n_params * updates, "tokens": train_cfg.max_iters * train_cfg.batch_size * config.block_size,
               "best_val": min(val_losses) if val_losses else None, "seconds": round(time.time() - t0, 1),
               "trained_at": time.strftime("%Y-%m-%d %H:%M"), "device": dev.device}
    save_model_config(paths.model_config, config, {**summary, "train_config": asdict(train_cfg)})
    with open(os.path.join(paths.root, "runs.jsonl"), "a") as f:
        f.write(json.dumps({**summary, "model": {k: getattr(config, k) for k in ("n_layer", "n_head", "n_embd", "block_size")},
                            "max_iters": train_cfg.max_iters}) + "\n")
    print(f"{updates:,} optimizer updates × {n_params:,} parameters = {n_params * updates:,} parameter updates")

    # 5. Plots
    plot_losses(train_losses, val_losses, train_cfg.eval_interval)

    # 6. Inference with the best checkpoint
    model = load_model(config, train_cfg.best_model_path, dev.device)
    for prompt in PROMPTS:
        print(generate_text(model, enc, prompt, dev.device))


def parse_args():
    p = argparse.ArgumentParser(description="Train a small GPT on a text dataset. Made for Apple Silicon.")
    p.add_argument("--dataset", action="append",
                   help="Hugging Face dataset name or link, or a link to a .txt/.jsonl/.json/.csv/.parquet file "
                        f"(default: {DEFAULT_DATASET}@{DEFAULT_CONFIG}). Repeat to train on several; each may be written "
                        "name@subset#column, e.g. Salesforce/wikitext@wikitext-2-raw-v1")
    p.add_argument("--dataset-config", default=None,
                   help="Dataset subset/config name, if it has one (e.g. wikitext-2-raw-v1)")
    p.add_argument("--text-column", default="text",
                   help="Column that holds the text (default: text)")
    p.add_argument("--mix", choices=["size", "equal"], default="size",
                   help="With several datasets: draw batches in proportion to their size (default) or equally")
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
    elif args.dataset and len(args.dataset) > 1:
        main(datasets=[parse_dataset_spec(s) for s in args.dataset], mix=args.mix)
    else:
        name, subset, column = parse_dataset_spec(args.dataset[0] if args.dataset else f"{DEFAULT_DATASET}@{DEFAULT_CONFIG}")
        main(dataset=name, dataset_config=args.dataset_config or subset,
             text_column=args.text_column if args.text_column != "text" else column)
