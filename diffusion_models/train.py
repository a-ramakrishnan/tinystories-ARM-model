# Training loop for the masked diffusion model (v1's loop, without Accelerate):
# AdamW, linear warmup then cosine decay, gradient clipping, mixed precision on Nvidia GPUs (bfloat16 on L4/A100,
# float16 + loss scaling on a T4), torch.compile on Nvidia GPUs.
#
# A run folder holds: model.pt (the best weights by the validation bound), config.json, metrics.json (loss
# history, speed, samples) and state.pt (the latest weights + optimizer, to resume after a disconnect).
#
#   python3 -m diffusion_models.train --mode smoke                 # a few minutes, checks everything works
#   python3 -m diffusion_models.train --mode colab --out <folder>  # the comparison run (on Colab)

import argparse
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from train_config import setup_device

from . import model as M
from .config import TrainSettings, run_mode
from .data import TINYSTORIES, BlockLoader, ensure_tokenized
from .noising import diffusion_loss, stratified_t
from .sample import diffusion_generate

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_DIR, "diffusion_models", "runs")
SAMPLE_PROMPTS = ["Once upon a time", "Lily wanted to", "The little dog was sad because"]


def lr_at(step, ts: TrainSettings):
    if step < ts.warmup_steps:
        return ts.lr * (step + 1) / ts.warmup_steps
    p = min(1.0, (step - ts.warmup_steps) / max(1, ts.steps - ts.warmup_steps))
    return ts.min_lr + 0.5 * (ts.lr - ts.min_lr) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, batches, T, mask_id, ctx):
    """Masked-token loss and the likelihood bound on fixed batches, noise levels spread evenly over 1..T and a
    fixed masking seed, so checkpoints are compared on exactly the same problem."""
    model.eval()
    g = torch.Generator(device=batches[0].device).manual_seed(0)
    losses, bounds = [], []
    for x in batches:
        loss, bound = diffusion_loss(model, x, T, mask_id, t=stratified_t(x.size(0), T, x.device), generator=g, ctx=ctx)
        losses.append(loss.item())
        bounds.append(bound.item())
    model.train()
    return sum(losses) / len(losses), sum(bounds) / len(bounds)


def train(cfg, ts, roots, out_dir, dev=None, resume=False, say=print, dataset_names=None):
    dev = dev or setup_device()
    device, ctx = dev.device, (dev.ctx if dev.device == "cuda" else nullcontext())
    cuda = device == "cuda"
    torch.manual_seed(ts.seed)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(out_dir, exist_ok=True)

    model = M.DiffusionTransformerLM(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": ts.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=ts.lr, betas=(0.9, 0.95), **({"fused": True} if cuda else {}))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and dev.dtype == "float16")
    start, history, best = 0, {"train": [], "val": []}, float("inf")
    state_path = os.path.join(out_dir, "state.pt")
    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        start, history, best = st["step"], st["history"], st["best"]
        say(f"Resuming at update {start:,} of {ts.steps:,} (best bound so far {best:.4f})")
    run = torch.compile(model) if cuda and ts.compile else model

    loader = BlockLoader(roots, cfg.seq_len, ts.batch_size, device)
    val_batches = loader.fixed_batches(ts.eval_batches, "val")
    n_params = model.n_params()
    say(f"Model: {n_params / 1e6:.1f}M parameters ({cfg.n_layers} layers x {cfg.d_model}, {cfg.n_heads} heads, "
        f"T = {cfg.diffusion_steps}) on {device} ({dev.dtype})")
    say(f"Training: {ts.steps:,} updates x {ts.batch_size * ts.grad_accum} x {cfg.seq_len} tokens = {ts.tokens * cfg.seq_len / 1e6:.0f}M tokens")

    extra = {"train": asdict(ts), "datasets": dataset_names or [os.path.basename(r) for r in roots]}
    t_start, t_log, tokens_since = time.time(), time.time(), 0
    for step in range(start, ts.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, ts)
        for _ in range(ts.grad_accum):
            loss, _ = diffusion_loss(run, loader.batch("train"), cfg.diffusion_steps, cfg.mask_id, ctx=ctx)
            scaler.scale(loss / ts.grad_accum).backward()
            tokens_since += ts.batch_size * cfg.seq_len
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), ts.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        u = step + 1
        if u % 50 == 0 or u == ts.steps:
            secs = time.time() - t_log
            tps = tokens_since / secs if secs > 0 else 0
            history["train"].append([u, round(loss.item(), 4), round(tps)])
            t_log, tokens_since = time.time(), 0
            if u % max(50, ts.eval_interval // 4) == 0:
                say(f"update {u:,}/{ts.steps:,}  loss {loss.item():.4f}  lr {opt.param_groups[0]['lr']:.2e}  {tps:,.0f} tokens/s")
        if u % ts.eval_interval == 0 or u == ts.steps:
            vl, vb = evaluate(run, val_batches, cfg.diffusion_steps, cfg.mask_id, ctx)
            history["val"].append([u, round(vl, 4), round(vb, 4)])
            is_best = vb < best
            if is_best:
                best = vb
                M.save(model, out_dir, extra=extra)
            say(f"  eval @ {u:,}: masked-token loss {vl:.4f} | likelihood bound {vb:.4f} nats/token"
                f"{'  (best, saved)' if is_best else ''}")
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": u, "history": history,
                        "best": best}, state_path)
        if ts.sample_interval and u % ts.sample_interval == 0 and u < ts.steps:
            r = diffusion_generate(model, SAMPLE_PROMPTS[0], 64, top_k=50, record_steps=False, seed=0)
            say(f"  sample @ {u:,}: {r['text']!r}")
            model.train()

    secs = time.time() - t_start
    best_model = M.load(out_dir, device)
    samples = []
    for i, p in enumerate(SAMPLE_PROMPTS):
        r = diffusion_generate(best_model, p, 128, top_k=50, record_steps=False, seed=i)
        samples.append({"prompt": p, "text": r["text"], "seconds": r["seconds"]})
    tps = [h[2] for h in history["train"][1:]]
    summary = {"model_type": "masked-diffusion", "params": n_params, "model": asdict(cfg), "train": asdict(ts),
               "datasets": extra["datasets"], "device": device, "dtype": dev.dtype,
               "gpu": torch.cuda.get_device_name(0) if cuda else platform.processor() or platform.machine(),
               "seconds": round(secs), "updates": ts.steps, "tokens": ts.tokens * cfg.seq_len,
               "tokens_per_sec": round(sum(tps) / len(tps)) if tps else None,
               "best_bound": round(best, 4), "final_val": history["val"][-1] if history["val"] else None,
               "history": history, "samples": samples, "finished_at": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    say(f"Done in {secs / 60:.1f} min. Best likelihood bound {best:.4f} nats/token. Saved in {out_dir}")
    return summary


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = summary["history"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot([x[0] for x in h["train"]], [x[1] for x in h["train"]], label="train (masked tokens)", alpha=.6)
    ax[0].plot([x[0] for x in h["val"]], [x[1] for x in h["val"]], "o-", label="validation (masked tokens)")
    ax[1].plot([x[0] for x in h["val"]], [x[2] for x in h["val"]], "o-", color="C3", label="validation bound")
    ax[0].set_title("Loss on the masked tokens")
    ax[1].set_title("Likelihood bound (nats per token; compare with AR loss)")
    for a in ax:
        a.set_xlabel("update")
        a.legend()
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Train the masked diffusion language model.")
    ap.add_argument("--mode", default="smoke", help="smoke, quick or colab (see config.RUN_MODES)")
    ap.add_argument("--dataset", default=TINYSTORIES[0])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", default=None, help="run folder (default diffusion_models/runs/<mode>-<time>)")
    ap.add_argument("--steps", type=int, default=None, help="override the preset's number of updates")
    ap.add_argument("--lr", type=float, default=None, help="override the peak learning rate (min_lr = lr / 10)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg, ts = run_mode(a.mode)
    if a.steps:
        ts.steps = a.steps
    if a.lr:
        ts.lr, ts.min_lr = a.lr, a.lr / 10
    root = ensure_tokenized(a.dataset, a.subset)
    out = a.out or os.path.join(RUNS_DIR, f"{a.mode}-{time.strftime('%Y%m%d-%H%M')}")
    s = train(cfg, ts, [root], out, resume=a.resume, dataset_names=[a.dataset + (f"@{a.subset}" if a.subset else "")])
    plot(s, os.path.join(out, "loss.png"))
    for x in s["samples"]:
        print(f"\n--- {x['prompt']!r}\n{x['text']}")


if __name__ == "__main__":
    main()
