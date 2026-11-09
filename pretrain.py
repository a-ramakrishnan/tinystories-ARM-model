# Pretraining loop: AdamW + linear warmup -> cosine decay, gradient accumulation, best-checkpoint saving

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from model import estimate_loss


def pretrain(model, loader, cfg, dev, callback=None):
    """Train `model` on batches from `loader`.

    cfg: TrainConfig, dev: DeviceSetup.
    Saves the weights with the best validation loss to cfg.best_model_path.
    Returns (train_losses, val_losses), one entry per evaluation.

    callback: optional function receiving progress events, used by the web front end:
      {"type": "eval", "step", "train", "val", "lr"} after each evaluation
      {"type": "step", "step", "loss", "lr"} after each training step
    If it returns False, training stops early.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=cfg.betas,
                                  weight_decay=cfg.weight_decay, eps=cfg.eps)

    scheduler_warmup = LinearLR(optimizer, total_iters=cfg.warmup_steps)  # linear warmup
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=cfg.max_iters - cfg.warmup_steps, eta_min=cfg.min_lr)  # lr decay
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay],
                             milestones=[cfg.warmup_steps])  # switch from warmup to decay

    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.dtype == "float16"))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    model.train()

    for step in tqdm(range(cfg.max_iters)):
        if step % cfg.eval_interval == 0 and step != 0:
            losses = estimate_loss(model, loader, cfg.eval_iters, dev.ctx)
            print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), cfg.best_model_path)

            if callback is not None and callback({"type": "eval", "step": step, "train": losses["train"],
                                                  "val": losses["val"],
                                                  "lr": optimizer.param_groups[0]["lr"]}) is False:
                break

        X, y = loader.get_batch("train")

        with dev.ctx:
            logits, loss = model(X, y)
            loss = loss / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if ((step + 1) % cfg.gradient_accumulation_steps == 0) or (step + 1 == cfg.max_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if callback is not None and callback({"type": "step", "step": step + 1,
                                              "loss": loss.item() * cfg.gradient_accumulation_steps,
                                              "lr": optimizer.param_groups[0]["lr"]}) is False:
            break

    return train_losses, val_losses
