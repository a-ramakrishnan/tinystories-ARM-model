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
        # Grouped-query attention (Llama 2 70B, Mistral, Qwen): n_kv_head key/value heads, each shared by
        # n_head / n_kv_head query heads, so the keys and values (and the KV cache) are that many times smaller
        self.n_kv_head = config.n_kv_head or config.n_head
        assert config.n_head % self.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        self.kv_dim = self.n_kv_head * (config.n_embd // config.n_head)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # QK-norm (OLMo 2, Gemma 3, Qwen3): RMS-normalize each head's query and key before attention, so their
        # dot products can't grow without bound; the learned scales set how sharp attention can get
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(config.n_embd // config.n_head)
            self.k_norm = RMSNorm(config.n_embd // config.n_head)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def _expand(self, t):
        """Repeat each key/value head for the query heads that share it (a no-op without GQA)."""
        rep = self.n_head // self.n_kv_head
        return t if rep == 1 else t.repeat_interleave(rep, dim=1)

    def forward(self, x, kv_cache=None, rope=None):
        """kv_cache: optional dict for fast generation. It holds the keys and values of earlier tokens;
        the new tokens' keys and values are added, so only the new tokens need computing.
        rope: optional (cos, sin) for these tokens' positions (RoPE models): queries and keys are rotated.

        The cache is a fixed-size buffer of max_len slots. Until it's full, tokens are written in order. Once
        it's full (RoPE models only, see GPT.forward) the newest token overwrites the oldest slot, like a ring:
        attention doesn't care about the order of keys, because each key carries its position in its rotation."""
        B, T, C = x.size()
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split([self.n_embd, self.kv_dim, self.kv_dim], dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)

        if kv_cache is not None:
            written, size = kv_cache.get("written", 0), kv_cache["max_len"]
            if "kbuf" not in kv_cache:  # allocate the whole window once, then write into it in place
                shape = (B, self.n_kv_head, size, hd)  # only the shared key/value heads are cached
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
            k, v = self._expand(k), self._expand(v)
            # first pass (prefill): causal over the prompt; later: the single new query may see every cached key
            y = F.scaled_dot_product_attention(q, k, v, is_causal=written == 0)
        elif self.flash:
            k, v = self._expand(k), self._expand(v)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            k, v = self._expand(k), self._expand(v)
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


def swiglu_hidden(n_embd):
    """SwiGLU's hidden width: 8/3 · n_embd (a multiple of 32), so its three matrices hold about as many
    parameters as the GELU MLP's two matrices of width 4 · n_embd (exactly as many for 384 and 768)."""
    return 32 * round(8 * n_embd / 3 / 32)


class SwiGLU(nn.Module):
    """Gated feed-forward layer (Llama, Mistral, Qwen): silu(x W_gate) * (x W_up), then W_down.
    The gate decides, feature by feature, how much of the "up" projection passes through."""
    def __init__(self, config):
        super().__init__()
        h = swiglu_hidden(config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, 2 * h, bias=config.bias)  # gate and up projections in one matrix
        self.c_proj = nn.Linear(h, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate, up = self.c_fc(x).chunk(2, dim=-1)
        return self.dropout(self.c_proj(F.silu(gate) * up))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln2 = make_norm(config)
        self.mlp = SwiGLU(config) if config.mlp == "swiglu" else MLP(config)

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
    mlp: str = "gelu"               # "gelu" (GPT-2: 4x wide) or "swiglu" (Llama: gated, 8/3x wide)
    qk_norm: bool = False           # RMS-normalize queries and keys per head before attention (OLMo 2, Qwen3)
    n_kv_head: int = 0              # key/value heads for grouped-query attention; 0 = n_head (ordinary attention)
    block_repeat: int = 1           # layer sharing (MobileLLM): apply each block this many times in a row, with the
                                    # same weights; more depth and compute, no more parameters
    mtp_heads: int = 1              # multi-token prediction (Gloeckle et al. 2024, DeepSeek-V3): 1 = next token only;
                                    # n = n - 1 extra heads, one transformer block each, also predict tokens 2…n ahead
                                    # during training. Generation and validation loss use the next-token head only.


def effective_layers(config):
    """Layers the input actually passes through: n_layer blocks, each applied block_repeat times."""
    return config.n_layer * max(1, getattr(config, "block_repeat", 1))


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
        # multi-token prediction: extra heads (a block each) on top of the trunk, sharing ln_f and the output layer
        if config.mtp_heads > 1:
            self.mtp = nn.ModuleList([Block(config) for _ in range(config.mtp_heads - 1)])
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
        i = 0
        for block in self.transformer.h:
            for _ in range(max(1, self.config.block_repeat)):  # layer sharing: the same block again, same weights
                x = block(x, kv_caches[i] if kv_caches is not None else None, rope)  # each pass has its own cache
                i += 1
        trunk = x
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training and self.config.mtp_heads > 1:
                # head k predicts the token k steps ahead: targets shifted by k - 1, the last k - 1 positions have none
                losses = [loss]
                for k, head in enumerate(self.mtp, start=2):
                    tk = torch.full_like(targets, -1)
                    tk[:, :t - (k - 1)] = targets[:, k - 1:]
                    lk = self.lm_head(self.transformer.ln_f(head(trunk, None, rope)))
                    losses.append(F.cross_entropy(lk.view(-1, lk.size(-1)), tk.view(-1), ignore_index=-1))
                loss = torch.stack(losses).mean()  # equal weight per head (as Gloeckle et al.)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def mtp_logits(self, idx):
        """Logits of every head for every position: a list, [next token, 2 ahead, …]."""
        t = idx.size(1)
        pos = torch.arange(t, device=idx.device)
        x, rope = self.transformer.wte(idx), None
        if self.config.pos_encoding == "rope":
            rope = rope_cos_sin(pos, self.config.n_embd // self.config.n_head, self.config.rope_theta)
        else:
            x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            for _ in range(max(1, self.config.block_repeat)):
                x = block(x, None, rope)
        out = [self.lm_head(self.transformer.ln_f(x))]
        for head in getattr(self, "mtp", []):
            out.append(self.lm_head(self.transformer.ln_f(head(x, None, rope))))
        return out

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
    hd = d // config.n_head
    nkv = config.n_kv_head or config.n_head
    gqa = nkv != config.n_head
    kvd = nkv * hd                           # key/value width (n_kv_head · head_dim; = n_embd without GQA)
    qkv = d * (d + 2 * kvd) + b * (d + 2 * kvd)  # c_attn: queries, then the (shared) keys and values
    attn_out = d * d + b * d                 # attention c_proj
    sw = config.mlp == "swiglu"
    h = swiglu_hidden(d) if sw else 4 * d    # the feed-forward layer's hidden width
    hs = f"{h}" if sw else "4·n_embd"
    mlp_in = d * (2 if sw else 1) * h + b * (2 if sw else 1) * h   # c_fc (SwiGLU: gate and up together)
    mlp_out = h * d + b * d                  # MLP c_proj
    qkn = 2 * hd if config.qk_norm else 0     # QK-norm: a scale per head dimension, for queries and for keys
    block = 2 * ln + qkv + attn_out + mlp_in + mlp_out + qkn
    # weights: attention (queries + output 2·n_embd², keys and values 2·n_embd·kv), feed-forward; then biases and norms
    attn_w = "2·n_embd² + 2·n_embd·kv" if gqa else "4·n_embd²"
    ffn_w = "3·n_embd·h" if sw else "8·n_embd²"
    weights = ("10·n_embd² + 2·n_embd·kv" if gqa else "12·n_embd²") if not sw else f"{attn_w} + {ffn_w}"
    coef = b * ((2 if gqa else 4) + (1 if sw else 5)) + (2 if rms else 2 + 2 * b)
    extra = (" + 2·kv" if gqa and b else "") + (" + 2·h" if sw and b else "")
    block_formula = f"{weights} + {coef}·n_embd{extra}"
    block_formula += " + 2·head_dim" if config.qk_norm else ""
    block_formula += (f", kv = {kvd}" if gqa else "") + (f", h = {h}" if sw else "")

    def row(name, formula, expr, count, note=""):
        return {"name": name, "formula": formula, "expr": expr, "count": count, "note": note}

    parts = [
        row("Token embeddings", "vocab_size × n_embd", f"{V:,} × {d}", V * d,
            "Also used as the output layer (weight tying)"),
        row("Position embeddings", "block_size × n_embd", f"{T} × {d}", T * d) if config.pos_encoding == "learned" else
        row("Positions (RoPE)", "rotations of queries and keys", "0", 0, "No parameters: computed from each position"),
        row("Transformer blocks", f"n_layer × ({block_formula})", f"{L} × {block:,}", L * block,
            f"Each block is applied {config.block_repeat}× in a row: {L * config.block_repeat} layers deep, no extra parameters"
            if getattr(config, "block_repeat", 1) > 1 else ""),
        row(f"Final {norm_name}", ln_formula, f"{ln // d} × {d}", ln),
        row("Output layer", "shares the token embeddings", "0", 0),
        *([row("Multi-token heads", "(mtp_heads − 1) × one block", f"{config.mtp_heads - 1} × {block:,}",
                (config.mtp_heads - 1) * block, "Training only: predict tokens 2… ahead; not used to generate")]
          if getattr(config, "mtp_heads", 1) > 1 else []),
    ]
    per_block = [
        row("Attention: query, key, value", (f"n_embd × (n_embd + 2·kv)" + (" + n_embd + 2·kv" if b else "")) if gqa
            else "n_embd × 3·n_embd" + (" + 3·n_embd" if b else ""),
            f"{d} × {d + 2 * kvd}" + (f" + {d + 2 * kvd}" if b else ""), qkv,
            f"{nkv} key/value heads shared by {config.n_head} query heads (kv = {nkv} × {hd})" if gqa else ""),
        row("Attention: output projection", "n_embd × n_embd" + (" + n_embd" if b else ""),
            f"{d} × {d}" + (f" + {d}" if b else ""), attn_out),
        row("SwiGLU: gate and up" if sw else "MLP: expand", f"n_embd × {'2·' if sw else ''}{hs}" + (f" + {'2·' if sw else ''}{hs}" if b else ""),
            f"{d} × {(2 if sw else 1) * h}" + (f" + {(2 if sw else 1) * h}" if b else ""), mlp_in),
        row("SwiGLU: down" if sw else "MLP: project back", f"{hs} × n_embd" + (" + n_embd" if b else ""),
            f"{h} × {d}" + (f" + {d}" if b else ""), mlp_out),
        row(f"Two {norm_name}s", f"2 × {ln_formula.replace(' × ', '·')}", f"2 × {ln}", 2 * ln),
        *([row("QK-norm", "2 × head_dim", f"2 × {hd}", qkn, "Scales for the normalized queries and keys")] if config.qk_norm else []),
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
