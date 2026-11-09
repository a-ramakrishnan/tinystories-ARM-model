# Inference: load trained weights and generate text from a prompt

import json
import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def save_model_config(path, config, extra=None):
    """Write the model config (plus any training summary in `extra`) next to the checkpoint."""
    with open(path, "w") as f:
        json.dump({"model": asdict(config), **(extra or {})}, f, indent=2)


def load_model_config(path):
    """Read a file written by save_model_config. Returns (GPTConfig, full dict)."""
    with open(path) as f:
        info = json.load(f)
    return GPTConfig(**info["model"]), info


def load_model(config, weights_path, device):
    """Rebuild the GPT from `config`, load saved weights, and put it in eval mode on `device`."""
    model = GPT(config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    return model.to(device).eval()


def generate_text(model, enc, prompt, device, max_new_tokens=200, temperature=1.0, top_k=None):
    """Continue `prompt` with up to `max_new_tokens` tokens and return the decoded text."""
    context = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())


@torch.no_grad()
def generate_stream(model, prompt_ids, device, max_new_tokens=200, temperature=1.0, top_k=None, n_alternatives=5):
    """Like GPT.generate, but yields each new token as soon as it's chosen, with timing and probabilities.

    Yields dicts: {"id", "prob", "alternatives": [(id, prob), ...], "ms"} where `prob` is the model's own
    probability for the chosen token (before temperature / top-k), and `ms` is the time since the start.
    There is no KV cache: every step runs the model over the whole context (up to block_size tokens).
    """
    block = model.config.block_size
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block else idx[:, -block:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        raw = F.softmax(logits, dim=-1)
        scaled = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled[scaled < v[:, [-1]]] = -float("Inf")
        idx_next = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1)
        alt_p, alt_i = torch.topk(raw[0], n_alternatives)
        token = int(idx_next.item())  # .item() waits for the GPU, so the timing below is real
        yield {"id": token, "prob": float(raw[0, token]),
               "alternatives": list(zip(alt_i.tolist(), [round(p, 4) for p in alt_p.tolist()])),
               "ms": (time.perf_counter() - t0) * 1000}
        idx = torch.cat((idx, idx_next), dim=1)


@torch.no_grad()
def next_token_probs(model, prompt_ids, device, k=10):
    """The model's top-k candidates for the token after `prompt_ids`: [(id, probability), ...]."""
    idx = torch.tensor(prompt_ids[-model.config.block_size:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(idx)
    p, i = torch.topk(F.softmax(logits[0, -1], dim=-1), k)
    return list(zip(i.tolist(), p.tolist()))


@torch.no_grad()
def attention_maps(model, prompt_ids, device, layer=0):
    """Attention weights of every head in one layer, for a prompt: tensor of shape (n_head, T, T).

    Re-runs the forward pass up to `layer` and computes the attention weights explicitly (the model itself
    uses PyTorch's fused attention, which doesn't return them).
    """
    idx = torch.tensor(prompt_ids[-model.config.block_size:], dtype=torch.long, device=device).unsqueeze(0)
    T = idx.size(1)
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
    for i, block in enumerate(model.transformer.h):
        if i == layer:
            a = block.attn
            q, k, _ = a.c_attn(block.ln1(x)).split(a.n_embd, dim=2)
            hs = a.n_embd // a.n_head
            q = q.view(1, T, a.n_head, hs).transpose(1, 2)
            k = k.view(1, T, a.n_head, hs).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
            att = att.masked_fill(torch.ones(T, T, device=device).tril() == 0, float("-inf"))
            return F.softmax(att, dim=-1)[0]
        x = block(x)
    raise ValueError(f"layer {layer} out of range (model has {len(model.transformer.h)})")
