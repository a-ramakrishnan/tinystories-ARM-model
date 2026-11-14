# Stage 2 (E2 in docs/plan.md): continue the base model's training on much more data.
#
# The base model saw 500M tokens of Mini FineWeb-Edu (small) + five Cosmopedia subsets. Stage 2 carries on from its
# saved training state (weights, AdamW moments, step 30,500) with the *full* Mini FineWeb-Edu (3.5M documents,
# ~6x small) in place of small, the same Cosmopedia subsets, to ~2.5B tokens in all (~20 tokens per parameter).
# pretrain() extends a run when asked for more steps than it was planned for: the cosine schedule is stretched to
# the new end and the learning rate re-warms onto it, so nothing jumps.
#
# Two things keep the numbers honest:
#  - The held-out FineWeb-Edu score (evaluate.py) uses the small subset's validation split. The full subset holds
#    the same documents, so they're removed from its training data before tokenizing.
#  - The saved best validation loss (2.903) was measured on the old mix; it's reset, so the first evaluation on the
#    new mix sets the bar and checkpoints keep being saved.
#
#   python3 stage2.py --base-run <folder with training_state.pt> --out <run folder> [--tokens 2.5e9]
#                     [--data-cache <folder>] [--prepare-only]
#
# Run it again with the same --out to resume after a disconnect. Token files are copied to --data-cache (Google
# Drive on Colab), so a new session skips the tokenizing.

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, fields

import torch

from data import BatchLoader, dataset_paths, get_dataset, parse_dataset_spec
from inference import save_model_config
from model import GPT, GPTConfig
from pretrain import STATE_FILE, STATE_META, pretrain
from tokenization import bin_files_exist, tokenize_and_save
from train_config import TrainConfig, setup_device

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
FINEWEB = "BaoMinhTran/mini-fineweb-edu"
BASE_CONFIG = os.path.join(PROJECT_DIR, "sft_configs", "base-14ed26.json")


def specs_from_base(base_info, fineweb_config="full"):
    """The base mix with FineWeb-Edu's subset swapped: [(spec, name, config, column, max_rows)]."""
    out = []
    for spec in base_info["datasets"]:
        rows = None
        s = spec
        if "%" in s:
            s, n = s.rsplit("%", 1)
            rows = int(n)
        name, config, column = parse_dataset_spec(s)
        if name == FINEWEB:
            config = fineweb_config
            spec = f"{FINEWEB}@{fineweb_config}"
        out.append((spec, name, config, column, rows))
    if not any(o[1] == FINEWEB for o in out):
        out.insert(0, (f"{FINEWEB}@{fineweb_config}", FINEWEB, fineweb_config, "text", None))
    return out


def text_hashes(ds):
    return {hashlib.md5(t.encode("utf-8")).hexdigest() for t in ds["text"]}


def prepare(specs, cache=None, num_proc=None, say=print):
    """Tokenize every dataset in the mix (or copy its token files from the cache). Returns the token folders."""
    num_proc = num_proc or os.cpu_count()
    roots = []
    for spec, name, config, column, rows in specs:
        p = dataset_paths(name, config, DATA_ROOT, max_rows=rows)
        roots.append(p.root)
        cached = os.path.join(cache, os.path.basename(p.root)) if cache else None
        if bin_files_exist(p.root):
            say(f"{spec}: token files already here")
            continue
        if cached and bin_files_exist(cached):
            t0 = time.time()
            os.makedirs(p.root, exist_ok=True)
            for f in ("train.bin", "validation.bin"):
                shutil.copy(os.path.join(cached, f), os.path.join(p.root, f))
            say(f"{spec}: token files copied from {cache} in {time.time() - t0:.0f} s")
            continue
        t0 = time.time()
        ds = get_dataset(p.hf_dir, name, config_name=config, text_column=column, max_rows=rows)
        if name == FINEWEB and config != "small":
            # keep the held-out set held out: drop small's validation documents from this subset
            small = get_dataset(dataset_paths(FINEWEB, "small", DATA_ROOT).hf_dir, FINEWEB, config_name="small")
            held = text_hashes(small["validation"])
            before = {k: len(v) for k, v in ds.items()}
            ds = ds.filter(lambda b: [hashlib.md5(t.encode("utf-8")).hexdigest() not in held for t in b["text"]],
                           batched=True, num_proc=num_proc, desc="removing the held-out documents")
            say(f"{spec}: removed {sum(before.values()) - sum(len(v) for v in ds.values()):,} of small's "
                f"{len(held):,} held-out documents ({before} -> { {k: len(v) for k, v in ds.items()} })")
        tokenize_and_save(ds, p.root, num_proc=num_proc)
        say(f"{spec}: tokenized in {(time.time() - t0) / 60:.1f} min")
        if cached:
            try:                       # a full Drive shouldn't stop the run: the cache only saves time later
                os.makedirs(cached, exist_ok=True)
                for f in ("train.bin", "validation.bin"):
                    shutil.copy(os.path.join(p.root, f), os.path.join(cached, f))
            except OSError as e:
                say(f"  (couldn't keep a copy in {cache}: {e})")
                shutil.rmtree(cached, ignore_errors=True)
    for spec, r in zip([s[0] for s in specs], roots):
        n = os.path.getsize(os.path.join(r, "train.bin")) // 2
        say(f"  {spec}: {n / 1e6:,.0f}M training tokens")
    return roots


def start_from_base(base_run, out, say=print):
    """Copy the base run's training state into the new run folder, with the best validation loss reset."""
    os.makedirs(out, exist_ok=True)
    st = torch.load(os.path.join(base_run, STATE_FILE), map_location="cpu", weights_only=False)
    say(f"Base training state: step {st['step']:,}, best validation loss {st['best_val']:.4f} (on the old mix)")
    st["base_best_val"], st["best_val"] = st["best_val"], float("inf")
    torch.save(st, os.path.join(out, STATE_FILE))
    meta = {"step": st["step"], "best_val": None, "continued_from": base_run, "base_best_val": st["base_best_val"]}
    json.dump(meta, open(os.path.join(out, STATE_META), "w"), indent=1)
    shutil.copy(os.path.join(base_run, "best_model_params.pt"), os.path.join(out, "base_model_params.pt"))
    return st


def main():
    ap = argparse.ArgumentParser(description="Stage 2: continue the base model on ~2.5B tokens.")
    ap.add_argument("--base-run", required=True, help="the base model's run folder (training_state.pt)")
    ap.add_argument("--out", required=True, help="the stage 2 run folder (resumes if it has a training state)")
    ap.add_argument("--tokens", type=float, default=2.5e9, help="total tokens, the base's 500M included")
    ap.add_argument("--fineweb", default="full", help="Mini FineWeb-Edu subset in place of small")
    ap.add_argument("--base-config", default=None, help="model_config.json of the base (default: the run's, else the repo copy)")
    ap.add_argument("--data-cache", default=None, help="folder to keep token files in (e.g. on Google Drive)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--num-proc", type=int, default=None)
    a = ap.parse_args()

    cfg_path = a.base_config or next(p for p in (os.path.join(a.base_run, "model_config.json"), BASE_CONFIG) if os.path.exists(p))
    base_info = json.load(open(cfg_path))
    specs = specs_from_base(base_info, a.fineweb)
    print("Mix:", ", ".join(s[0] for s in specs), flush=True)
    roots = prepare(specs, a.data_cache, a.num_proc)
    if a.prepare_only:
        return

    dev = setup_device()
    config = GPTConfig(**base_info["model"])
    tc = base_info.get("train_config") or {}
    keep = {f.name for f in fields(TrainConfig)}
    train_cfg = TrainConfig(**{k: (tuple(v) if k == "betas" else v) for k, v in tc.items() if k in keep})
    steps = int(round(a.tokens / (train_cfg.batch_size * config.block_size), -2))
    train_cfg.max_iters = steps
    train_cfg.best_model_path = os.path.join(a.out, "best_model_params.pt")
    train_cfg.compile = dev.device == "cuda"
    if not os.path.exists(os.path.join(a.out, STATE_FILE)):
        start_from_base(a.base_run, a.out)
    print(f"Training to step {steps:,} ({steps * train_cfg.batch_size * config.block_size / 1e9:.2f}B tokens), "
          f"peak lr {train_cfg.learning_rate:g}, run folder {a.out}", flush=True)
    model = GPT(config).to(dev.device)
    loader = BatchLoader(roots, config.block_size, train_cfg.batch_size, dev.device)
    start = json.load(open(os.path.join(a.out, STATE_META)))["step"]
    t0 = time.time()
    train_losses, val_losses = pretrain(model, loader, train_cfg, dev, resume=True, model_config=asdict(config))
    secs = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    session_tokens = (steps - start) * train_cfg.batch_size * config.block_size
    summary = {k: base_info[k] for k in ("dataset_config",) if k in base_info}
    summary.update({
        "datasets": [s[0] for s in specs], "mix": "size", "steps": steps, "updates": steps,
        "tokens": steps * train_cfg.batch_size * config.block_size, "params": n_params, "param_updates": n_params * steps,
        "best_val": min(val_losses) if val_losses else None, "seconds": round(secs, 1),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"), "stopped_early": False, "device": dev.device,
        "chip": torch.cuda.get_device_name(0) if dev.device == "cuda" else dev.device,
        "platform": "Colab" if os.path.exists("/content") else "local", "dtype": dev.dtype,
        "tokens_per_sec": round(session_tokens / secs) if secs else None,
        "stage": 2, "continued_from": base_info.get("library_id") or a.base_run,
        "base_tokens": base_info.get("tokens"), "base_best_val": base_info.get("best_val"),
        "held_out_removed": "Mini FineWeb-Edu small's validation documents are not in the training data",
        "train_config": asdict(train_cfg),
    })
    save_model_config(os.path.join(a.out, "model_config.json"), config, summary)
    print(f"Done: best validation loss {summary['best_val']:.4f} (on the new mix). Saved in {a.out}")


if __name__ == "__main__":
    main()
