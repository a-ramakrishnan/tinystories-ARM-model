# Training configuration: hyperparameters and device / precision setup

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4             # more stable training, earlier 1e-4
    min_lr: float = 1e-5                    # cosine decays down to this; must be below learning_rate
    max_iters: int = 20000                  # increase from 25000
    warmup_steps: int = 1000                # smoother initial train, earlier 100
    eval_interval: int = 500                # evaluate every N iterations
    eval_iters: int = 500                   # batches per evaluation, increased from 100
    batch_size: int = 32                    # changed from 16, better gradient estimate
    gradient_accumulation_steps: int = 32   # reduced from 50
    weight_decay: float = 0.1               # weight decay for regularization
    betas: tuple = (0.9, 0.95)              # changed beta2 to 0.95
    eps: float = 1e-9
    grad_clip: float = 0.5
    best_model_path: str = "best_model_params.pt"
    seed: int = 42


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
