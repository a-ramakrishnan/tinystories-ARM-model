# Inference: load trained weights and generate text from a prompt

import json
import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, apply_rope, rope_cos_sin


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


# ---------- sampling: how the next token is picked ----------
# Order (the usual one): penalties -> temperature -> top-k -> top-p -> min-p, then sample (or argmax when greedy).

SAMPLING_DEFAULTS = {"top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0, "frequency_penalty": 0.0,
                     "presence_penalty": 0.0}


def apply_penalties(logits, history, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    """Discourage tokens that already appear in `history` (1-D tensor of token ids).

    repetition_penalty (CTRL-style): divides a seen token's logit by the penalty if positive, multiplies if
    negative, so it always becomes less likely. frequency_penalty subtracts penalty × (times seen);
    presence_penalty subtracts a flat amount from every token seen at least once (OpenAI-style).
    Returns (new logits, counts).
    """
    counts = torch.zeros_like(logits).scatter_add_(0, history, torch.ones_like(history, dtype=logits.dtype))
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits, counts
    seen = counts > 0
    if repetition_penalty != 1.0:
        logits = torch.where(seen, torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty), logits)
    return logits - frequency_penalty * counts - presence_penalty * seen.to(logits.dtype), counts


def sampling_distribution(logits, history, temperature=1.0, top_k=None, top_p=1.0, min_p=0.0,
                          repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, stages=False):
    """Turn one position's logits (1-D) into the distribution the next token is drawn from.

    temperature <= 0 means greedy: all probability on the single most likely token (after penalties).
    top_k keeps the k most likely tokens; top_p keeps the smallest set whose probability adds up to p
    (nucleus sampling); min_p drops tokens less likely than min_p × the top token's probability.
    Returns (probs, n_kept), or with stages=True a dict with every intermediate step (for the explorer).
    """
    logits = logits.float()
    penalized, counts = apply_penalties(logits, history, repetition_penalty, frequency_penalty, presence_penalty)
    greedy = temperature <= 0
    scaled = penalized / (1.0 if greedy else temperature)
    p = F.softmax(scaled, dim=-1)
    cut_k = torch.zeros_like(p, dtype=torch.bool)
    cut_p = torch.zeros_like(cut_k)
    cut_min = torch.zeros_like(cut_k)
    if greedy:
        keep = torch.zeros_like(cut_k)
        keep[torch.argmax(p)] = True
        cut_k = ~keep   # reported as "greedy" by callers
    else:
        if top_k:
            kth = torch.topk(p, min(int(top_k), p.numel())).values[-1]
            cut_k = p < kth
        if top_p < 1.0:
            q = p.masked_fill(cut_k, 0)
            sp, si = torch.sort(q, descending=True)
            before = torch.cumsum(sp, 0) - sp            # probability of the tokens ranked above this one
            drop = before >= top_p * q.sum()
            cut_p = torch.zeros_like(cut_k).scatter_(0, si, drop) & ~cut_k
        if min_p > 0.0:
            cut_min = (p < min_p * p.max()) & ~cut_k & ~cut_p
        keep = ~(cut_k | cut_p | cut_min)
    final = p.masked_fill(~keep, 0)
    final = final / final.sum()
    n_kept = int(keep.sum())
    if not stages:
        return final, n_kept
    return {"raw": F.softmax(logits, dim=-1), "penalized": F.softmax(penalized, dim=-1), "scaled": p, "final": final,
            "counts": counts, "cut_k": cut_k, "cut_p": cut_p, "cut_min": cut_min, "greedy": greedy, "n_kept": n_kept,
            "kept_mass": float(p[keep].sum())}


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
                    use_kv_cache=True, stats=None, stop_token=None, sampling=None):
    """Like GPT.generate, but yields each new token as soon as it's chosen, with timing and probabilities.

    stop_token: stop when the model picks this token (e.g. <|endoftext|>, the end of a story); it isn't yielded.
    sampling: optional top_p, min_p, repetition_penalty, frequency_penalty, presence_penalty (see
    sampling_distribution). temperature <= 0 picks the most likely token every time (greedy).

    Yields dicts: {"id", "prob", "alternatives": [(id, prob), ...], "ms"} where `prob` is the model's own
    probability for the chosen token (before temperature / top-k), and `ms` is the time since the start.

    With use_kv_cache, the keys and values of earlier tokens are kept, so each step only runs the new token
    through the model. Without it, every step re-reads the whole context (up to block_size tokens). This model
    uses learned absolute positions, so once the context window is full the positions shift and the cache has
    to be rebuilt ("recompute"). `stats`, if given, is filled with cache size and recompute counts.
    """
    block = model.config.block_size
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    sliding = getattr(model.config, "pos_encoding", "learned") == "rope"
    caches, cache_len, pos, recomputes = None, 0, 0, 0
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        if use_kv_cache:
            # Learned positions are tied to slots 0..block_size-1, so a full window means rebuilding the cache
            # with every position shifted. RoPE models just slide: the oldest key/value is overwritten.
            if caches is None or (not sliding and cache_len + 1 > block):
                if caches is not None:
                    recomputes += 1
                caches = [{"max_len": block} for _ in range(model.config.n_layer)]
                ctx = idx[:, -block:]
                logits, _ = model(ctx, kv_caches=caches, start_pos=0)
                cache_len = pos = ctx.size(1)
            else:
                logits, _ = model(idx[:, -1:], kv_caches=caches, start_pos=pos)
                pos += 1
                cache_len = min(cache_len + 1, block)
        else:
            idx_cond = idx if idx.size(1) <= block else idx[:, -block:]
            logits, _ = model(idx_cond)
        logits = logits[0, -1, :]
        raw = F.softmax(logits.float(), dim=-1).unsqueeze(0)
        probs, n_kept = sampling_distribution(logits, idx[0, -block:], temperature, top_k, **{**SAMPLING_DEFAULTS, **(sampling or {})})
        idx_next = (torch.argmax(probs) if temperature <= 0 else torch.multinomial(probs, num_samples=1)).view(1, 1)
        alt_p, alt_i = torch.topk(raw[0], n_alternatives)
        token = int(idx_next.item())  # .item() waits for the GPU, so the timing below is real
        if stop_token is not None and token == stop_token:
            if stats is not None:
                stats["stopped"] = True
            return
        if stats is not None:
            stats.update(kv_cache_bytes=kv_cache_bytes(caches) if caches else 0, cache_tokens=cache_len if caches else 0,
                         recomputes=recomputes, sliding=sliding)
        yield {"id": token, "prob": float(raw[0, token]), "kept": n_kept,
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
            # time for each token (the first includes the prompt's prefill), for per-position charts
            "token_ms": [round(b - a, 2) for a, b in zip([0.0] + times[:-1], times)],
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
    rope = None
    if getattr(model.config, "pos_encoding", "learned") == "rope":
        rope = rope_cos_sin(pos, model.config.n_embd // model.config.n_head, model.config.rope_theta)
        x = model.transformer.drop(model.transformer.wte(idx))
    else:
        x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
    for i, block in enumerate(model.transformer.h):
        if i == layer:
            a = block.attn
            q, k, _ = a.c_attn(block.ln1(x)).split(a.n_embd, dim=2)
            hs = a.n_embd // a.n_head
            q = q.view(1, T, a.n_head, hs).transpose(1, 2)
            k = k.view(1, T, a.n_head, hs).transpose(1, 2)
            if getattr(a, "qk_norm", False):
                q, k = a.q_norm(q), a.k_norm(k)
            if rope is not None:
                q, k = apply_rope(q, *rope), apply_rope(k, *rope)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
            att = att.masked_fill(torch.ones(T, T, device=device).tril() == 0, float("-inf"))
            return F.softmax(att, dim=-1)[0]
        x = block(x, None, rope)
    raise ValueError(f"layer {layer} out of range (model has {len(model.transformer.h)})")
