# Pretraining loop: AdamW + linear warmup -> cosine decay, gradient accumulation, best-checkpoint saving,
# and a resumable training state.
#
# Two files are written next to the best checkpoint:
#   best_model_params.pt   the weights with the best validation loss (what generation uses)
#   training_state.pt      everything needed to carry on: the latest weights, AdamW's moment estimates,
#                          the mixed-precision scaler, the step, the loss history and the random-number state
#   training_state.json    a small summary of training_state.pt (step, losses), quick to read
#
# Resuming with the same max_iters continues an interrupted run exactly where it stopped. Resuming with a
# larger max_iters *extends* a run: the cosine schedule is stretched to the new end, and the learning rate
# ramps from where it was back up to the stretched curve over a short re-warmup, so it doesn't jump.

import json
import math
import os
from dataclasses import asdict

import torch
from tqdm.auto import tqdm

from model import estimate_loss

STATE_FILE = "training_state.pt"
STATE_META = "training_state.json"
REWARM_STEPS = 200


def state_paths(cfg):
    folder = os.path.dirname(os.path.abspath(cfg.best_model_path))
    return os.path.join(folder, STATE_FILE), os.path.join(folder, STATE_META)


def read_state_meta(folder):
    """The summary of a saved training state in `folder`, or None if there's none."""
    path = os.path.join(folder, STATE_META)
    if not (os.path.exists(path) and os.path.exists(os.path.join(folder, STATE_FILE))):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def lr_at(step, cfg, rewarm=None):
    """Learning rate for a step: linear warmup from a third of the peak, then cosine decay to min_lr at
    max_iters (the same shape as torch's LinearLR + CosineAnnealingLR). `rewarm` = (start step, lr there,
    length): after an extended resume, blend from that lr to the curve over `length` steps."""
    if step < cfg.warmup_steps:
        lr = cfg.learning_rate * (1 / 3 + (2 / 3) * step / cfg.warmup_steps)
    else:
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_iters - cfg.warmup_steps))
        lr = cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    if rewarm and step < rewarm[0] + rewarm[2]:
        lr = rewarm[1] + (lr - rewarm[1]) * (step - rewarm[0]) / rewarm[2]
    return lr


def _save_state(cfg, model, optimizer, scaler, step, best_val, train_losses, val_losses, lr, extra):
    path, meta_path = state_paths(cfg)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "step": step, "best_val": best_val, "train_losses": train_losses, "val_losses": val_losses,
             "lr": lr, "train_config": asdict(cfg), "rng": torch.get_rng_state(), **(extra or {})}
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # never leave a half-written state behind
    meta = {"step": step, "max_iters": cfg.max_iters, "best_val": None if best_val == float("inf") else best_val,
            "last_val": val_losses[-1] if val_losses else None, "lr": lr, "eval_interval": cfg.eval_interval,
            "train_losses": [round(x, 4) for x in train_losses], "val_losses": [round(x, 4) for x in val_losses],
            "train_config": asdict(cfg), **{k: v for k, v in (extra or {}).items() if k != "model_config"},
            "model_config": (extra or {}).get("model_config")}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def pretrain(model, loader, cfg, dev, callback=None, resume=False, save_state=True, model_config=None):
    """Train `model` on batches from `loader`.

    cfg: TrainConfig, dev: DeviceSetup.
    Saves the weights with the best validation loss to cfg.best_model_path, and (with save_state) the full
    training state after every evaluation and at the end, so the run can be resumed.
    resume: continue from the saved training state next to cfg.best_model_path (see the top of this file).
    model_config: optional dict stored with the state, so a resume can rebuild the same model.
    Returns (train_losses, val_losses), one entry per evaluation (including those before a resume).

    callback: optional function receiving progress events, used by the web front end:
      {"type": "eval", "step", "train", "val", "lr"} after each evaluation
      {"type": "step", "step", "loss", "lr"} after each training step
    If it returns False, training stops early (and the state is saved, so it can be resumed).
    """
    cuda = dev.device == "cuda"
    if cuda and getattr(cfg, "tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    fused = cuda and getattr(cfg, "fused_adamw", True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=cfg.betas,
                                  weight_decay=cfg.weight_decay, eps=cfg.eps, **({"fused": True} if fused else {}))
    # torch.compile on an Nvidia GPU: the compiled module runs the steps; `model` itself (same weights) is what's
    # saved, so checkpoints keep their usual names
    run = torch.compile(model) if cuda and getattr(cfg, "compile", False) else model
    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.dtype == "float16"))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    start, rewarm = 0, None
    if resume:
        path, _ = state_paths(cfg)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No training state to resume from at {path}")
        st = torch.load(path, map_location=dev.device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["rng"].cpu() if hasattr(st["rng"], "cpu") else st["rng"])
        start, best_val_loss = st["step"], st["best_val"]
        train_losses, val_losses = list(st["train_losses"]), list(st["val_losses"])
        if start >= cfg.max_iters:
            raise ValueError(f"The saved run is already at step {start:,}; ask for more than {start:,} steps.")
        if cfg.max_iters != st["train_config"]["max_iters"]:  # extending: re-warm onto the stretched schedule
            rewarm = (start, st["lr"], max(1, min(REWARM_STEPS, (cfg.max_iters - start) // 5)))
        print(f"Resuming at step {start:,} of {cfg.max_iters:,} (best validation loss so far {best_val_loss:.4f})"
              + (f"; the schedule now ends at step {cfg.max_iters:,}" if rewarm else ""))
        del st
    extra = {"model_config": model_config}

    model.train()
    lr = lr_at(start, cfg, rewarm)
    step = start
    stopped = False

    def evaluate(at_step):
        nonlocal best_val_loss
        losses = estimate_loss(run, loader, cfg.eval_iters, dev.ctx)
        print(f"Step {at_step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"The current learning rate: {lr:.5f}")
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), cfg.best_model_path)
        if save_state:
            _save_state(cfg, model, optimizer, scaler, at_step, best_val_loss, train_losses, val_losses, lr, extra)
        return callback is None or callback({"type": "eval", "step": at_step, "train": losses["train"],
                                             "val": losses["val"], "lr": lr}) is not False

    for step in tqdm(range(start, cfg.max_iters), initial=start, total=cfg.max_iters):
        lr = lr_at(step, cfg, rewarm)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % cfg.eval_interval == 0 and step != 0 and step != start:
            if not evaluate(step):
                stopped = True
                break

        X, y = loader.get_batch("train")

        with dev.ctx:
            logits, loss = run(X, y)
            loss = loss / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if ((step + 1) % cfg.gradient_accumulation_steps == 0) or (step + 1 == cfg.max_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if callback is not None and callback({"type": "step", "step": step + 1,
                                              "loss": loss.item() * cfg.gradient_accumulation_steps,
                                              "lr": lr}) is False:
            stopped = True
            step += 1
            break
    else:
        step = cfg.max_iters

    if not stopped:
        # A final evaluation, so the last stretch of training can still give the best model (and a run shorter
        # than eval_interval still gets its checkpoint)
        evaluate(step)
    else:
        if best_val_loss == float("inf"):  # stopped before any evaluation: keep what was learned so far
            torch.save(model.state_dict(), cfg.best_model_path)
        if save_state:
            _save_state(cfg, model, optimizer, scaler, step, best_val_loss, train_losses, val_losses, lr, extra)

    return train_losses, val_losses
