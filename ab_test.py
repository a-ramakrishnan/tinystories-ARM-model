# A/B tests for architecture changes: train several variants of the model the same way (same data, seed, batches
# and number of updates), evaluate each with the evaluation suite, and compare.
#
#   python3 ab_test.py --steps 1000 \
#       --variant "baseline: rope swiglu no-bias" \
#       --variant "qk-norm: rope swiglu no-bias qk-norm"
#
# A variant is "name: options", the options being any of
#   learned | rope          position encoding           layernorm | rmsnorm     normalization
#   gelu | swiglu           feed-forward layer          bias | no-bias          bias terms
#   qk-norm                 normalize queries and keys  lr=1.8e-3               peak learning rate (default 6e-4)
#   size=tiny|small|medium|large (default small)
# Anything not given is the GPT-2 default (learned, layernorm, gelu, bias, no QK-norm).
#
# Each update uses 32 sequences of 256 tokens. On a Mac they're split into micro-batches of 8 (gradient
# accumulation: the same gradients in a quarter of the memory); on an Nvidia GPU they're one batch.
#
# Results go to outputs/ab/<stamp>/: one folder per variant (weights, config, evaluation), results.json, a
# comparison table (results.md), a loss chart (losses.png) and, with --zip, one zip to bring home:
#   python3 import_run.py ab-<stamp>.zip      adds every variant to the model library, with its evaluation
# Runs on the same machine are comparable; across machines (Mac vs GPU), compare each to its own baseline.

import argparse
import json
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import evaluate  # noqa: E402
import registry  # noqa: E402
from data import BatchLoader, dataset_paths, get_dataset  # noqa: E402
from inference import benchmark_generation, save_model_config  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from pretrain import pretrain  # noqa: E402
from tokenization import bin_files_exist, get_tokenizer, tokenize_and_save  # noqa: E402
from train_config import TrainConfig, setup_device  # noqa: E402

DATA_ROOT = os.path.join(PROJECT_DIR, "data")
SIZES = {"tiny": (4, 4, 256), "small": (6, 6, 384), "medium": (8, 8, 512), "large": (12, 12, 768)}
DEFAULT_LR = 6e-4
BATCH, BLOCK, SEED = 32, 256, 42
# small datasets for the held-out part of the evaluation suite (WikiText-2 and Tiny Shakespeare, a few MB each)
EVAL_SETS = [("Salesforce/wikitext", "wikitext-2-raw-v1", "text"),
             ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", None, "text")]
BENCH_PROMPT = "Once upon a time there was a little girl who"


def parse_variant(spec):
    """'name: rope swiglu lr=1e-3' -> (name, model options, learning rate, size)."""
    name, _, opts = spec.partition(":")
    name = name.strip()
    if not opts.strip():  # no colon: the name is the options too
        opts, name = name, re.sub(r"\s+", "-", name.strip())
    if not re.fullmatch(r"[A-Za-z0-9._+-]+", name):
        raise SystemExit(f"Variant name '{name}': use letters, digits and . _ + - only")
    cfg, lr, size = {}, DEFAULT_LR, "small"
    for tok in opts.split():
        t = tok.lower()
        if t in ("learned", "rope"):
            cfg["pos_encoding"] = t
        elif t in ("layernorm", "rmsnorm"):
            cfg["norm"] = t
        elif t in ("gelu", "swiglu"):
            cfg["mlp"] = t
        elif t in ("bias", "no-bias"):
            cfg["bias"] = t == "bias"
        elif t == "qk-norm":
            cfg["qk_norm"] = True
        elif t.startswith("lr="):
            lr = float(t[3:])
        elif t.startswith("size=") and t[5:] in SIZES:
            size = t[5:]
        else:
            raise SystemExit(f"Variant '{name}': unknown option '{tok}' (see the top of ab_test.py)")
    return name, cfg, lr, size


def describe(cfg, lr, size):
    parts = [size, cfg.get("pos_encoding", "learned"), cfg.get("norm", "layernorm"), cfg.get("mlp", "gelu")]
    parts += [] if cfg.get("bias", True) else ["no-bias"]
    parts += ["qk-norm"] if cfg.get("qk_norm") else []
    return " ".join(parts) + f" lr={lr:g}"


def ensure_dataset(name, config_name, column, say=print):
    """Download and tokenize a dataset once (skipped if its .bin files exist); returns its folder."""
    paths = dataset_paths(name, config_name, DATA_ROOT)
    if not bin_files_exist(paths.root):
        say(f"Preparing {name}{' / ' + config_name if config_name else ''} (downloaded and tokenized once)")
        ds = get_dataset(paths.hf_dir, name, config_name=config_name, text_column=column)
        tokenize_and_save(ds, paths.root, num_proc=min(8, os.cpu_count() or 1))
    return paths.root


def chip_name(dev):
    if dev.device == "cuda":
        return torch.cuda.get_device_name(0)
    return "Apple Silicon" if dev.device == "mps" else "CPU"


def train_variant(name, cfg, lr, size, steps, data_root, dataset_name, out_dir, dev, say=print):
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    n_layer, n_head, n_embd = SIZES[size]
    config = GPTConfig(vocab_size=50257, block_size=BLOCK, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                       dropout=0.0, **{"bias": True, **cfg})
    torch.manual_seed(SEED)
    model = GPT(config).to(dev.device)
    micro = 8 if dev.device != "cuda" else BATCH       # the Mac's 16 GB can't hold a batch of 32 at once
    accum = BATCH // micro
    # pretrain counts micro-batches, so every step count is multiplied by accum
    tcfg = TrainConfig(learning_rate=lr, min_lr=lr / 10, max_iters=steps * accum,
                       warmup_steps=max(1, steps // 10) * accum, eval_interval=max(1, steps // 4) * accum,
                       eval_iters=50 * accum, batch_size=micro, gradient_accumulation_steps=accum, seed=SEED,
                       best_model_path=os.path.join(folder, "best_model_params.pt"))
    loader = BatchLoader(data_root, BLOCK, micro, dev.device)
    step_losses = []

    def cb(e):
        if e["type"] == "step" and e["step"] % accum == 0:
            step_losses.append(round(e["loss"], 4))

    torch.manual_seed(SEED)
    say(f"\n=== {name}: {describe(cfg, lr, size)}, {steps:,} updates of {BATCH} × {BLOCK} tokens"
        + (f" ({accum} micro-batches of {micro})" if accum > 1 else ""))
    t0 = time.time()
    _, val = pretrain(model, loader, tcfg, dev, callback=cb, save_state=False)
    secs = time.time() - t0
    n = sum(p.numel() for p in model.parameters())
    finite = [v for v in val if math.isfinite(v)]
    save_model_config(os.path.join(folder, "model_config.json"), config,
                      {"datasets": [dataset_name], "steps": steps, "updates": steps, "params": n,
                       "best_val": min(finite) if finite else None, "seconds": round(secs, 1),
                       "trained_at": time.strftime("%Y-%m-%d %H:%M"), "device": dev.device, "chip": chip_name(dev),
                       "platform": "ab_test", "tokens": steps * BATCH * BLOCK})
    del model
    return {"name": name, "steps": steps, "options": cfg, "lr": lr, "size": size, "description": describe(cfg, lr, size),
            "params": n, "val_curve": [round(v, 4) for v in val], "eval_every": max(1, steps // 4),
            "best_val": round(min(finite), 4) if finite else None, "seconds": round(secs, 1),
            "updates_per_sec": round(steps / secs, 2), "step_losses": step_losses,
            "loss_spikes": spikes(step_losses), "config": asdict(config)}


def spikes(losses, window=50, factor=1.5):
    """Updates whose loss jumped above `factor` × the median of the previous `window` updates: a rough
    measure of how steady training was."""
    out = 0
    for i in range(window, len(losses)):
        prev = sorted(losses[i - window:i])
        if not math.isfinite(losses[i]) or losses[i] > factor * prev[window // 2]:
            out += 1
    return out


def evaluate_variant(r, out_dir, dataset_name, dev, eval_sets, title=None, say=print):
    folder = os.path.join(out_dir, r["name"])
    meta, _ = registry.add_model(os.path.join(folder, "best_model_params.pt"), os.path.join(folder, "model_config.json"),
                                 source="trained", name=f"{title or 'A/B'} · {r['name']} · {r['description']} · {r['steps']:,} updates")
    say(f"Evaluating {r['name']}")
    ev = evaluate.evaluate_model(meta["id"], dev.device, datasets=eval_sets, progress=lambda m: None)
    with open(os.path.join(folder, "eval.json"), "w") as f:
        json.dump(ev, f, indent=1)
    from inference import load_model, load_model_config
    config, _ = load_model_config(os.path.join(folder, "model_config.json"))
    model = load_model(config, os.path.join(folder, "best_model_params.pt"), dev.device)
    ids = get_tokenizer().encode_ordinary(BENCH_PROMPT)
    bench = benchmark_generation(model, ids, dev.device, n_tokens=BLOCK - len(ids) + 80)
    del model
    r.update({"model_id": meta["id"], "heldout": {k: v["loss"] for k, v in ev["heldout"].items()},
              "heldout_titles": {k: v["title"] for k, v in ev["heldout"].items()}, "generation": ev["generation"],
              "kv_speedup": bench["speedup"], "kv_recomputes": bench["kv_cache"]["recomputes"]})
    if dev.device == "mps":
        torch.mps.empty_cache()
    elif dev.device == "cuda":
        torch.cuda.empty_cache()


def _pct(x):
    return f"{x * 100:.1f}%" if x is not None else "–"


def table(results):
    """A markdown comparison table: one column per variant."""
    if not results:
        return ""
    keys = list(results[0].get("heldout", {}))
    titles = results[0].get("heldout_titles", {})
    rows = [("Options", lambda r: r["description"]), ("Parameters", lambda r: f"{r['params']:,}"),
            ("Validation loss (each quarter)", lambda r: " / ".join(f"{v:.3f}" for v in r["val_curve"])),
            ("Best validation loss", lambda r: f"{r['best_val']:.3f}" if r["best_val"] is not None else "diverged")]
    rows += [(f"Held-out: {titles.get(k, k)}", lambda r, k=k: f"{r['heldout'][k]:.3f}" if k in r.get("heldout", {}) else "–")
             for k in keys]
    rows += [("Greedy / Balanced repetition", lambda r: f"{_pct(r['generation']['greedy_repeated_4grams'])} / "
              f"{_pct(r['generation']['balanced_repeated_4grams'])}" if "generation" in r else "–"),
             ("Loss spikes", lambda r: str(r["loss_spikes"])),
             ("Updates / s", lambda r: f"{r['updates_per_sec']:.2f}")]
    out = ["| | " + " | ".join(r["name"] for r in results) + " |", "|---|" + "---|" * len(results)]
    out += ["| " + label + " | " + " | ".join(f(r) for r in results) + " |" for label, f in rows]
    return "\n".join(out)


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        xs = [r["eval_every"] * (i + 1) for i in range(len(r["val_curve"]))]
        a.plot(xs, r["val_curve"], marker="o", label=r["name"])
        sl = r["step_losses"]
        k = 25
        smooth = [sum(sl[max(0, i - k):i + 1]) / len(sl[max(0, i - k):i + 1]) for i in range(len(sl))]
        b.plot(range(1, len(sl) + 1), smooth, label=r["name"], linewidth=1)
    a.set(title="Validation loss", xlabel="update", ylabel="loss")
    b.set(title="Training loss (moving average)", xlabel="update", ylabel="loss")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_zip(out_dir, stamp, dest_dir):
    path = os.path.join(dest_dir, f"ab-{stamp}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, out_dir))
    return path


def run(variants, steps=1000, dataset=("roneneldan/TinyStories", None, "text"), eval_sets=True, zip_dir=None,
        out_root=os.path.join(PROJECT_DIR, "outputs", "ab"), title=None, say=print):
    dev = setup_device()
    parsed = [parse_variant(v) for v in variants]
    names = [p[0] for p in parsed]
    if len(set(names)) != len(names):
        raise SystemExit("Variant names must be different")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, stamp)
    os.makedirs(out_dir, exist_ok=True)
    say(f"A/B test on {chip_name(dev)} ({dev.device}, {dev.dtype}): {len(parsed)} variants × {steps:,} updates")

    data_root = ensure_dataset(*dataset, say=say)
    sets = [{"key": os.path.basename(data_root), "title": evaluate.DATASET_TITLES.get(os.path.basename(data_root),
             dataset[0]), "path": os.path.join(data_root, "validation.bin")}]
    if eval_sets:
        for spec in EVAL_SETS:
            root = ensure_dataset(*spec, say=say)
            key = os.path.basename(root)
            sets.append({"key": key, "title": evaluate.DATASET_TITLES.get(key, "Tiny Shakespeare" if "input.txt" in key
                         else spec[0]), "path": os.path.join(root, "validation.bin")})

    results = []
    summary = {"title": title, "stamp": stamp, "steps": steps, "dataset": dataset[0], "device": dev.device,
               "chip": chip_name(dev), "batch": BATCH, "block": BLOCK, "seed": SEED, "variants": results}
    for name, cfg, lr, size in parsed:
        r = train_variant(name, cfg, lr, size, steps, data_root, dataset[0], out_dir, dev, say)
        results.append(r)
        if r["best_val"] is not None:
            evaluate_variant(r, out_dir, dataset[0], dev, sets, title, say)
        with open(os.path.join(out_dir, "ab.json"), "w") as f:  # after each variant, so nothing is lost
            json.dump(summary, f, indent=1)
        say(f"{name}: best validation loss {r['best_val']}, {r['loss_spikes']} loss spikes, {r['updates_per_sec']} updates/s")

    z = report(summary, out_dir, zip_dir, say)
    return summary, out_dir, z


def report(summary, out_dir, zip_dir=None, say=print):
    """Write results.md and losses.png from the results, and the zip if asked; returns the zip's path."""
    results, title = summary["variants"], summary.get("title")
    md = table(results)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(f"# A/B test{': ' + title if title else ''}\n\n{summary['steps']:,} updates of {BATCH} × {BLOCK} tokens "
                f"on {summary['dataset']}, seed {SEED}, on {summary['chip']}.\n\n{md}\n")
    plot(results, os.path.join(out_dir, "losses.png"))
    say("\n" + md)
    z = make_zip(out_dir, summary["stamp"], zip_dir) if zip_dir else None
    if z:
        say(f"\nResults zip: {z}  (bring it home with  python3 import_run.py {os.path.basename(z)})")
    return z


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and compare model variants (see the top of this file).")
    ap.add_argument("--variant", action="append", required=True, help='"name: options", e.g. "qk: rope swiglu qk-norm"')
    ap.add_argument("--steps", type=int, default=1000, help="weight updates per variant (default 1,000)")
    ap.add_argument("--dataset", default="roneneldan/TinyStories", help="Hugging Face dataset to train on")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--no-eval-sets", action="store_true", help="skip WikiText-2 and Tiny Shakespeare in the evaluation")
    ap.add_argument("--zip", metavar="DIR", help="also write ab-<stamp>.zip to this folder")
    ap.add_argument("--title", help="a name for this experiment, e.g. 'A5 QK-norm'")
    a = ap.parse_args()
    run(a.variant, a.steps, (a.dataset, a.dataset_config, a.text_column), not a.no_eval_sets, a.zip, title=a.title)
