# Model architecture (GPT-style decoder-only transformer) and loss estimation

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction and the bias (Llama, Mistral, Qwen): each vector is divided by
    its root mean square, then scaled per feature. Fewer operations and parameters, the same job."""
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()  # the mean of squares in float32, so half precision doesn't overflow
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def make_norm(config):
    return RMSNorm(config.n_embd) if config.norm == "rmsnorm" else LayerNorm(config.n_embd, config.bias)


def rope_cos_sin(positions, head_dim, theta=10000.0):
    """RoPE angles for these token positions: each pair of dimensions in a query or key is rotated by
    position × frequency, with frequencies from 1 down to 1/theta. Returns (cos, sin), each (T, head_dim/2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    angles = positions.float()[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate x (B, n_head, T, head_dim) by the RoPE angles (the "rotate half" layout used by Llama).
    After rotation, the dot product of a query at position m and a key at position n depends only on m - n."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype)[None, None], sin.to(x.dtype)[None, None]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache=None, rope=None):
        """kv_cache: optional dict for fast generation. It holds the keys and values of earlier tokens;
        the new tokens' keys and values are added, so only the new tokens need computing.
        rope: optional (cos, sin) for these tokens' positions (RoPE models): queries and keys are rotated.

        The cache is a fixed-size buffer of max_len slots. Until it's full, tokens are written in order. Once
        it's full (RoPE models only, see GPT.forward) the newest token overwrites the oldest slot, like a ring:
        attention doesn't care about the order of keys, because each key carries its position in its rotation."""
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)

        if kv_cache is not None:
            written, size = kv_cache.get("written", 0), kv_cache["max_len"]
            if "kbuf" not in kv_cache:  # allocate the whole window once, then write into it in place
                shape = (B, self.n_head, size, C // self.n_head)
                kv_cache["kbuf"] = torch.empty(shape, dtype=k.dtype, device=k.device)
                kv_cache["vbuf"] = torch.empty(shape, dtype=v.dtype, device=v.device)
            assert written == 0 or T == 1, "with a filled cache, feed one new token at a time"
            assert written + T <= size or (T == 1 and rope is not None), "cache full: only RoPE models can slide"
            if written + T <= size:          # still filling up, in order
                kv_cache["kbuf"][:, :, written:written + T] = k
                kv_cache["vbuf"][:, :, written:written + T] = v
            else:                             # full: overwrite the oldest slot (ring buffer)
                slot = written % size
                kv_cache["kbuf"][:, :, slot:slot + 1] = k
                kv_cache["vbuf"][:, :, slot:slot + 1] = v
            kv_cache["written"] = written + T
            n = min(written + T, size)
            kv_cache["len"] = n
            k, v = kv_cache["kbuf"][:, :, :n], kv_cache["vbuf"][:, :, :n]
            kv_cache["k"], kv_cache["v"] = k, v  # views of the filled part (used to report the cache size)
            # first pass (prefill): causal over the prompt; later: the single new query may see every cached key
            y = F.scaled_dot_product_attention(q, k, v, is_causal=written == 0)
        elif self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln2 = make_norm(config)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, rope=None):
        x = x + self.attn(self.ln1(x), kv_cache, rope)
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    pos_encoding: str = "learned"   # "learned": a trained vector per position (GPT-2); "rope": rotary (Llama)
    rope_theta: float = 10000.0     # RoPE base: sets the slowest rotation frequency
    norm: str = "layernorm"         # "layernorm" (GPT-2) or "rmsnorm" (Llama): scale only, no mean or bias


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            **({"wpe": nn.Embedding(config.block_size, config.n_embd)} if config.pos_encoding == "learned" else {}),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=make_norm(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        """kv_caches: optional list of one dict per layer (see CausalSelfAttention) for fast generation;
        start_pos is then the position of the first token in idx."""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(start_pos, start_pos + t, dtype=torch.long, device=device)

        x = self.transformer.wte(idx)
        rope = None
        if self.config.pos_encoding == "rope":
            # positions enter attention as rotations of queries and keys (no position vector is added); only
            # distances between tokens matter, so start_pos may run past block_size while the cache slides
            rope = rope_cos_sin(pos, self.config.n_embd // self.config.n_head, self.config.rope_theta)
        else:
            assert start_pos + t <= self.config.block_size, "learned positions stop at block_size"
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            x = block(x, kv_caches[i] if kv_caches is not None else None, rope)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens given a conditioning sequence.
        idx: Tensor of shape (B, T)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def param_breakdown(config):
    """Count the model's parameters part by part from its config, with the formula for each part.

    The total equals sum(p.numel() for p in GPT(config).parameters()). The output layer reuses the
    token embedding matrix (weight tying), so it adds no parameters of its own.
    """
    V, T, d, L = config.vocab_size, config.block_size, config.n_embd, config.n_layer
    b = 1 if config.bias else 0
    rms = config.norm == "rmsnorm"
    ln = d if rms else (1 + b) * d           # LayerNorm: weight (+ bias); RMSNorm: weight only
    norm_name = "RMSNorm" if rms else "LayerNorm"
    ln_formula = "n_embd" if rms or not b else "2 × n_embd"
    qkv = d * 3 * d + b * 3 * d              # c_attn
    attn_out = d * d + b * d                 # attention c_proj
    mlp_in = d * 4 * d + b * 4 * d           # c_fc
    mlp_out = 4 * d * d + b * d              # MLP c_proj
    block = 2 * ln + qkv + attn_out + mlp_in + mlp_out
    linear_terms = f"{(9 if b else 0) + (2 if rms else 4 if b else 2)}·n_embd"

    def row(name, formula, expr, count, note=""):
        return {"name": name, "formula": formula, "expr": expr, "count": count, "note": note}

    parts = [
        row("Token embeddings", "vocab_size × n_embd", f"{V:,} × {d}", V * d,
            "Also used as the output layer (weight tying)"),
        row("Position embeddings", "block_size × n_embd", f"{T} × {d}", T * d) if config.pos_encoding == "learned" else
        row("Positions (RoPE)", "rotations of queries and keys", "0", 0, "No parameters: computed from each position"),
        row("Transformer blocks", f"n_layer × (12·n_embd² + {linear_terms})", f"{L} × {block:,}", L * block),
        row(f"Final {norm_name}", ln_formula, f"{ln // d} × {d}", ln),
        row("Output layer", "shares the token embeddings", "0", 0),
    ]
    per_block = [
        row("Attention: query, key, value", "n_embd × 3·n_embd" + (" + 3·n_embd" if b else ""),
            f"{d} × {3 * d}" + (f" + {3 * d}" if b else ""), qkv),
        row("Attention: output projection", "n_embd × n_embd" + (" + n_embd" if b else ""),
            f"{d} × {d}" + (f" + {d}" if b else ""), attn_out),
        row("MLP: expand", "n_embd × 4·n_embd" + (" + 4·n_embd" if b else ""),
            f"{d} × {4 * d}" + (f" + {4 * d}" if b else ""), mlp_in),
        row("MLP: project back", "4·n_embd × n_embd" + (" + n_embd" if b else ""),
            f"{4 * d} × {d}" + (f" + {d}" if b else ""), mlp_out),
        row(f"Two {norm_name}s", f"2 × {ln_formula.replace(' × ', '·')}", f"2 × {ln}", 2 * ln),
    ]
    return {"total": sum(p["count"] for p in parts), "parts": parts, "per_block": per_block, "block": block}


def estimate_loss(model, loader, eval_iters, ctx):
    """Mean loss over `eval_iters` random batches for the train and val splits."""
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = loader.get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out
