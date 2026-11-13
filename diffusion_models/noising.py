# Noising and the loss (as in v1_original.py).
#
# Forward process: pick a noise level t in 1..T and replace each token with [MASK] with probability r(t) = t/T.
# The model sees the masked sequence and t, and is trained to predict the original token at the masked positions.
#
# Two numbers come out of the loss:
#   - loss: cross-entropy averaged over the masked tokens (v1's training objective);
#   - nll_bound: the same errors weighted by 1/r(t) and averaged over all tokens. For the masked diffusion model
#     this is an upper bound on the true negative log-likelihood per token (the MDLM/LLaDA bound, here with T
#     discrete levels), so it's the number to put next to an autoregressive model's loss. It's noisier than
#     `loss`, so evaluate it on many batches.

import torch
import torch.nn.functional as F


def mask_ratio_schedule(t, T):
    """Linear schedule: at level t, a fraction t/T of the tokens is masked."""
    return t.float() / float(T)


@torch.no_grad()
def corrupt_with_mask(input_ids, t, T, mask_id, generator=None):
    """Mask each token with probability t/T. Returns (noisy ids, labels with -100 where unmasked, mask)."""
    ratio = mask_ratio_schedule(t, T).unsqueeze(1)                      # [B, 1]
    rand = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    masked = rand < ratio
    noisy = torch.where(masked, torch.full_like(input_ids, mask_id), input_ids)
    labels = torch.where(masked, input_ids, torch.full_like(input_ids, -100))
    return noisy, labels, masked


def diffusion_loss(model, input_ids, T, mask_id, t=None, generator=None, ctx=None):
    """One batch: random noise levels (or the given t), mask, predict. Returns (loss, nll_bound)."""
    B, L = input_ids.shape
    if t is None:
        t = torch.randint(1, T + 1, (B,), device=input_ids.device, generator=generator)
    noisy, labels, masked = corrupt_with_mask(input_ids, t, T, mask_id, generator)
    if ctx is not None:
        with ctx:
            logits = model(noisy, timesteps=t)
    else:
        logits = model(noisy, timesteps=t)
    ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
                         reduction="none").view(B, L)
    n_masked = masked.sum()
    loss = ce.sum() / n_masked.clamp(min=1)
    bound = (ce.sum(1) / mask_ratio_schedule(t, T)).mean() / L
    return loss, bound


def stratified_t(B, T, device):
    """Noise levels spread evenly over 1..T across a batch (lower-variance evaluation)."""
    return (torch.arange(B, device=device) * T // B + 1).clamp(1, T)
