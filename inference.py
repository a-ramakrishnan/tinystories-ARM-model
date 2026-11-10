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
    ids = out[0].tolist()
    new = ids[context.size(1):]
    if enc.eot_token in new:  # the story ended: drop <|endoftext|> and anything after it
        ids = ids[:context.size(1) + new.index(enc.eot_token)]
    return enc.decode(ids)


def kv_cache_bytes(caches):
    """Memory held by a list of per-layer KV caches."""
    return sum(c[k].numel() * c[k].element_size() for c in caches for k in ("k", "v") if c.get(k) is not None)


def kv_bytes_per_token(config, dtype_bytes=4):
    """Each token stores a key and a value vector of n_embd numbers in every layer."""
    return 2 * config.n_layer * config.n_embd * dtype_bytes


def device_memory(device):
    """GPU memory in use right now (bytes), plus the peak where the backend tracks it."""
    device = str(device)
    if device.startswith("cuda"):
        return {"allocated": torch.cuda.memory_allocated(), "peak": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.memory_reserved(), "total": torch.cuda.get_device_properties(0).total_memory,
                "name": torch.cuda.get_device_name(0)}
    if device == "mps":
        return {"allocated": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory(),
                "peak": None, "name": "Apple GPU (MPS)"}
    return {"allocated": None, "peak": None, "name": "CPU"}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def generate_stream(model, prompt_ids, device, max_new_tokens=200, temperature=1.0, top_k=None, n_alternatives=5,
                    use_kv_cache=True, stats=None, stop_token=None):
    """Like GPT.generate, but yields each new token as soon as it's chosen, with timing and probabilities.

    stop_token: stop when the model picks this token (e.g. <|endoftext|>, the end of a story); it isn't yielded.

    Yields dicts: {"id", "prob", "alternatives": [(id, prob), ...], "ms"} where `prob` is the model's own
    probability for the chosen token (before temperature / top-k), and `ms` is the time since the start.

    With use_kv_cache, the keys and values of earlier tokens are kept, so each step only runs the new token
    through the model. Without it, every step re-reads the whole context (up to block_size tokens). This model
    uses learned absolute positions, so once the context window is full the positions shift and the cache has
    to be rebuilt ("recompute"). `stats`, if given, is filled with cache size and recompute counts.
    """
    block = model.config.block_size
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    caches, cache_len, recomputes = None, 0, 0
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        if use_kv_cache:
            if caches is None or cache_len + 1 > block:
                if caches is not None:
                    recomputes += 1
                caches = [{"max_len": block} for _ in range(model.config.n_layer)]
                ctx = idx[:, -block:]
                logits, _ = model(ctx, kv_caches=caches, start_pos=0)
                cache_len = ctx.size(1)
            else:
                logits, _ = model(idx[:, -1:], kv_caches=caches, start_pos=cache_len)
                cache_len += 1
        else:
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
        if stop_token is not None and token == stop_token:
            if stats is not None:
                stats["stopped"] = True
            return
        if stats is not None:
            stats.update(kv_cache_bytes=kv_cache_bytes(caches) if caches else 0, cache_tokens=cache_len if caches else 0,
                         recomputes=recomputes)
        yield {"id": token, "prob": float(raw[0, token]),
               "alternatives": list(zip(alt_i.tolist(), [round(p, 4) for p in alt_p.tolist()])),
               "ms": (time.perf_counter() - t0) * 1000}
        idx = torch.cat((idx, idx_next), dim=1)


def benchmark_generation(model, prompt_ids, device, n_tokens=100, seed=1234, repeats=2):
    """Generate the same tokens with and without the KV cache and compare speed and memory.

    Both runs use the same random seed, so they should produce the same text (tiny float differences can
    occasionally change a sampled token). Each mode runs `repeats` times and the fastest run is kept, which
    smooths out GPU timing noise.
    """
    results = {}
    for use_cache in (False, True):  # warm up both code paths (GPU kernels for each shape) so timing is fair
        for _ in generate_stream(model, prompt_ids, device, min(n_tokens, 30), use_kv_cache=use_cache):
            pass
    for use_cache in (False, True):
        best = None
        for _ in range(repeats):
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            _sync(device)
            torch.manual_seed(seed)
            stats, times, ids = {}, [], []
            for ev in generate_stream(model, prompt_ids, device, n_tokens, temperature=1.0, top_k=50,
                                      use_kv_cache=use_cache, stats=stats):
                times.append(ev["ms"])
                ids.append(ev["id"])
            _sync(device)
            if best is None or times[-1] < best[0][-1]:
                best = (times, ids, stats)
        times, ids, stats = best
        decode_ms = times[-1] - times[0]
        results["kv_cache" if use_cache else "no_cache"] = {
            "ttft_ms": round(times[0], 1), "total_ms": round(times[-1], 1),
            "tokens_per_sec": round((len(times) - 1) / (decode_ms / 1000), 1) if decode_ms > 0 else None,
            "ms_per_token": round(decode_ms / (len(times) - 1), 2) if len(times) > 1 else None,
            "kv_cache_bytes": stats.get("kv_cache_bytes", 0), "recomputes": stats.get("recomputes", 0),
            "memory": device_memory(device), "ids": ids,
        }
    a, b = results["no_cache"], results["kv_cache"]
    results["speedup"] = round(a["total_ms"] / b["total_ms"], 2) if b["total_ms"] else None
    results["same_output"] = a["ids"] == b["ids"]
    for r in (a, b):
        r.pop("ids")
    return results


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
