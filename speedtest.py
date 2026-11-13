"""Training speed test: tokens per second for the 124M best-design model under different settings.

Each setting builds the model fresh, runs a few warm-up steps (torch.compile compiles here), then times real
training steps (forward, backward, AdamW) on real batches, and reports tokens/s and peak GPU memory. The loss is
shown too, as a sanity check that a setting trains the same way.

    python3 speedtest.py --data data/baominhtran__mini-fineweb-edu-small
    python3 speedtest.py --data ... --batches 32,64,128 --steps 60

Settings tried: the current recipe (batch 32, no compile, plain AdamW) as the baseline, then fused AdamW + TF32,
then torch.compile, then bigger micro-batches (more tokens per step keep a large GPU busy).
"""

import argparse
import time

import torch

from data import BatchLoader
from model import GPT, GPTConfig
from train_config import TrainConfig, setup_device


def best_design(block=256):
    return GPTConfig(block_size=block, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=False,
                     pos_encoding="rope", mlp="swiglu", qk_norm=True)


def measure(data_dir, batch, compile_, fused, tf32, steps, warmup, dev, block=256):
    torch.manual_seed(42)
    if dev.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = GPT(best_design(block)).to(dev.device)
    cfg = TrainConfig()
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, betas=cfg.betas, weight_decay=cfg.weight_decay,
                            eps=cfg.eps, **({"fused": True} if fused and dev.device == "cuda" else {}))
    run = torch.compile(model) if compile_ and dev.device == "cuda" else model
    loader = BatchLoader(data_dir, block, batch, dev.device)
    sync = torch.cuda.synchronize if dev.device == "cuda" else (torch.mps.synchronize if dev.device == "mps" else (lambda: None))
    losses = []
    t_compile = time.time()
    for i in range(warmup + steps):
        if i == warmup:
            sync()
            t_compile = time.time() - t_compile
            t0 = time.time()
        X, y = loader.get_batch("train")
        with dev.ctx:
            _, loss = run(X, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i >= warmup and i % 10 == 0:
            losses.append(loss.item())
    sync()
    secs = time.time() - t0
    tps = steps * batch * block / secs
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev.device == "cuda" else None
    del model, opt, run
    return {"tokens_per_s": round(tps), "ms_per_step": round(1000 * secs / steps, 1), "warmup_s": round(t_compile, 1),
            "peak_gb": round(mem, 1) if mem else None, "loss": round(sum(losses) / len(losses), 3) if losses else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="a tokenized data folder (with train.bin)")
    ap.add_argument("--batches", default="32,64,128")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--block", type=int, default=256)
    a = ap.parse_args()
    dev = setup_device()
    gpu = torch.cuda.get_device_name(0) if dev.device == "cuda" else dev.device
    print(f"Speed test on {gpu} ({dev.dtype}), 124M best design, context {a.block}")
    batches = [int(b) for b in a.batches.split(",")]
    b0 = batches[0]           # the smallest batch is the baseline (32 at 256 tokens, 16 at 512: the same tokens)
    settings = [(f"baseline: batch {b0}, plain AdamW, no compile", b0, False, False, False),
                ("+ fused AdamW + TF32", b0, False, True, True),
                ("+ torch.compile", b0, True, True, True)]
    settings += [(f"+ compile, batch {b}", b, True, True, True) for b in batches[1:]]
    rows = []
    for name, b, comp, fused, tf32 in settings:
        try:
            r = measure(a.data, b, comp, fused, tf32, a.steps, a.warmup if comp else 5, dev, a.block)
        except torch.cuda.OutOfMemoryError:
            r = {"tokens_per_s": None, "note": "out of memory"}
            torch.cuda.empty_cache()
        rows.append((name, r))
        print(f"{name:42s} {r}", flush=True)
    base = rows[0][1]["tokens_per_s"]
    print("\nSummary (tokens/s, × the baseline):")
    for name, r in rows:
        t = r.get("tokens_per_s")
        print(f"  {name:42s} {t if t else '–':>9}  {f'{t / base:.2f}×' if t and base else ''}  "
              f"{'' if r.get('peak_gb') is None else str(r['peak_gb']) + ' GB'}")
