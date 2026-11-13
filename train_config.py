# Training configuration: hyperparameters and device / precision setup

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    learning_rate: float = 6e-4             # peak learning rate (as in nanoGPT); 1e-4 learned far too slowly
    min_lr: float = 6e-5                    # cosine decays down to this (a tenth of the peak)
    max_iters: int = 20000                  # training steps
    warmup_steps: int = 1000                # linear warmup before the cosine decay
    eval_interval: int = 1000               # evaluate every N steps
    eval_iters: int = 100                   # batches per evaluation
    batch_size: int = 32                    # sequences per batch
    gradient_accumulation_steps: int = 1    # update the weights every step (32 meant only 625 updates in 20,000 steps)
    weight_decay: float = 0.1               # weight decay for regularization
    betas: tuple = (0.9, 0.95)              # beta2 0.95, as in GPT-style training
    eps: float = 1e-9
    grad_clip: float = 1.0
    best_model_path: str = "best_model_params.pt"
    seed: int = 42
    # speed (Nvidia GPUs; ignored on the Mac): none of these change the maths, only how fast it runs
    compile: bool = False                   # torch.compile the model: fuses small GPU operations (slow first steps)
    fused_adamw: bool = True                # AdamW's update as one GPU kernel instead of one per parameter
    tf32: bool = True                       # TensorFloat-32 for the float32 matrix multiplies left outside autocast


@dataclass
class DeviceSetup:
    device: str        # "cuda", "mps" or "cpu"
    dtype: str         # "float32", "bfloat16" or "float16"
    ctx: object        # autocast context (or nullcontext)


def setup_device():
    """Pick the fastest available device and matching precision.

    Nvidia GPU (cuda) > Apple Silicon GPU (mps) > CPU.
    Mixed precision only on CUDA; MPS and CPU train in float32 (no GradScaler needed).
    How to use autocast: https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
    """
    if torch.cuda.is_available():
        device = "cuda"
        # bfloat16 needs Ampere or newer (compute capability 8.0+: A100, L4, RTX 30xx/40xx). Older GPUs such as
        # the T4 report bfloat16 as "supported" only through slow emulation, so they use float16 + GradScaler.
        dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
    elif torch.backends.mps.is_available():
        device, dtype = "mps", "float32"
    else:
        device, dtype = "cpu", "float32"

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device, dtype=ptdtype) if device == "cuda" else nullcontext()
    return DeviceSetup(device=device, dtype=dtype, ctx=ctx)
