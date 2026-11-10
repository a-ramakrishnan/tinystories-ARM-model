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

    def forward(self, x, kv_cache=None):
        """kv_cache: optional dict for fast generation. It holds the keys and values of earlier tokens;
        the new tokens' keys and values are appended, so only the new tokens need computing."""
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if kv_cache is not None:
            n = kv_cache.get("len", 0)
            if "kbuf" not in kv_cache:  # allocate the whole window once, then write into it in place
                shape = (B, self.n_head, kv_cache["max_len"], C // self.n_head)
                kv_cache["kbuf"] = torch.empty(shape, dtype=k.dtype, device=k.device)
                kv_cache["vbuf"] = torch.empty(shape, dtype=v.dtype, device=v.device)
            assert n == 0 or T == 1, "with a filled cache, feed one new token at a time"
            kv_cache["kbuf"][:, :, n:n + T] = k
            kv_cache["vbuf"][:, :, n:n + T] = v
            kv_cache["len"] = n + T
            k, v = kv_cache["kbuf"][:, :, :n + T], kv_cache["vbuf"][:, :, :n + T]
            kv_cache["k"], kv_cache["v"] = k, v  # views of the filled part (used to report the cache size)
            # first pass (prefill): causal over the prompt; later: the single new query may see every cached key
            y = F.scaled_dot_product_attention(q, k, v, is_causal=n == 0)
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
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache)
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


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
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
        assert start_pos + t <= self.config.block_size
        pos = torch.arange(start_pos, start_pos + t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for i, block in enumerate(self.transformer.h):
            x = block(x, kv_caches[i] if kv_caches is not None else None)
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
    ln = (1 + b) * d                         # LayerNorm: weight (+ bias)
    qkv = d * 3 * d + b * 3 * d              # c_attn
    attn_out = d * d + b * d                 # attention c_proj
    mlp_in = d * 4 * d + b * 4 * d           # c_fc
    mlp_out = 4 * d * d + b * d              # MLP c_proj
    block = 2 * ln + qkv + attn_out + mlp_in + mlp_out
    linear_terms = "13·n_embd" if config.bias else "2·n_embd"

    def row(name, formula, expr, count, note=""):
        return {"name": name, "formula": formula, "expr": expr, "count": count, "note": note}

    parts = [
        row("Token embeddings", "vocab_size × n_embd", f"{V:,} × {d}", V * d,
            "Also used as the output layer (weight tying)"),
        row("Position embeddings", "block_size × n_embd", f"{T} × {d}", T * d),
        row("Transformer blocks", f"n_layer × (12·n_embd² + {linear_terms})", f"{L} × {block:,}", L * block),
        row("Final LayerNorm", "2 × n_embd" if config.bias else "n_embd", f"{1 + b} × {d}", ln),
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
        row("Two LayerNorms", "2 × 2·n_embd" if b else "2 × n_embd", f"2 × {ln}", 2 * ln),
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
