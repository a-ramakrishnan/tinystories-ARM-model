"""The model's internals for one prompt, for the Explorer view in "Inside the transformer".

explain_forward runs the model block by block, exactly as GPT.forward does, and records what each stage
produced: the embeddings, every head's attention scores (before and after the mask and softmax), the queries,
keys and values of one chosen layer, the feed-forward layer's activations (or, with experts, the router's
choices), the size of each token's vector after every block, and what the model would predict if it stopped
after each block (the "logit lens": the final norm and output layer applied early).

Vectors are long (384 or 768 numbers), so for drawing they're sampled: `strip` keeps every k-th value.
"""

import math

import torch
import torch.nn.functional as F

from model import MoE, SwiGLU, apply_rope, rope_cos_sin


def strip(v, n=48):
    """Every k-th value of the last dimension, n values in all (a picture of a long vector)."""
    d = v.shape[-1]
    idx = torch.linspace(0, d - 1, min(n, d)).round().long().to(v.device)
    return v.index_select(-1, idx)


def r(t, nd=3):
    """A tensor as nested lists of rounded floats (JSON)."""
    return torch.round(t.float().cpu().double() * 10 ** nd).div(10 ** nd).tolist()  # float64: 0.182 stays 0.182


@torch.no_grad()
def explain_forward(model, ids, device, layer=0, top_n=60, lens_k=3):
    cfg = model.config
    T = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    pos = torch.arange(T, device=device)
    hd = cfg.n_embd // cfg.n_head
    tok = model.transformer.wte(idx)
    rope = None
    if cfg.pos_encoding == "rope":
        rope = rope_cos_sin(pos, hd, cfg.rope_theta)
        pos_emb, x = None, tok
    else:
        pos_emb = model.transformer.wpe(pos)[None]
        x = tok + pos_emb
    out = {
        "embedding": {"token": r(strip(tok[0])), "position": r(strip(pos_emb[0])) if pos_emb is not None else None,
                      "sum": r(strip(x[0])), "token_norm": r(tok[0].norm(dim=-1), 2)},
        "layers": [],
    }
    lens_head = lambda h: F.softmax(model.lm_head(model.transformer.ln_f(h)).float(), dim=-1)  # noqa: E731
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    n_pass = 0
    for b, block in enumerate(model.transformer.h):
        for _ in range(max(1, cfg.block_repeat)):
            a = block.attn
            h_in = block.ln1(x)
            q, k, v = a.c_attn(h_in).split([a.n_embd, a.kv_dim, a.kv_dim], dim=2)
            q = q.view(1, T, a.n_head, hd).transpose(1, 2)
            k = k.view(1, T, a.n_kv_head, hd).transpose(1, 2)
            v = v.view(1, T, a.n_kv_head, hd).transpose(1, 2)
            if a.qk_norm:
                q, k = a.q_norm(q), a.k_norm(k)
            if rope is not None:
                q, k = apply_rope(q, *rope), apply_rope(k, *rope)
            ke, ve = a._expand(k), a._expand(v)
            scores = (q @ ke.transpose(-2, -1)) / math.sqrt(hd)                    # (1, H, T, T)
            att = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
            heads = att @ ve                                                       # (1, H, T, hd)
            y = heads.transpose(1, 2).contiguous().view(1, T, cfg.n_embd)
            attn_out = a.c_proj(y)
            x = x + attn_out
            m_in = block.ln2(x)
            rec = {"block": b, "attention": r(att[0]), "resid_norm_mid": r(x[0].norm(dim=-1), 2)}
            mlp = block.mlp
            if isinstance(mlp, MoE):
                logits_r = mlp.router(m_in)[0].float()
                p = F.softmax(logits_r, dim=-1)
                tp, ti = torch.topk(p, mlp.top_k, dim=-1)
                rec["router"] = {"probs": r(p, 3), "chosen": ti.tolist()}
                mlp_out = mlp(m_in)
            else:
                if isinstance(mlp, SwiGLU):
                    gate, up = mlp.c_fc(m_in).chunk(2, dim=-1)
                    hidden = F.silu(gate) * up
                else:
                    hidden = mlp.gelu(mlp.c_fc(m_in))
                rec["mlp"] = {"hidden": r(strip(hidden[0], 64)), "width": hidden.shape[-1],
                              "active": r((hidden[0].abs() > 0.1 * hidden[0].abs().amax(-1, keepdim=True)).float().mean(-1), 3)}
                mlp_out = mlp.c_proj(hidden)
            x = x + mlp_out
            rec["resid_norm"] = r(x[0].norm(dim=-1), 2)
            lp = lens_head(x)[0]                                                  # (T, V)
            lv, li = torch.topk(lp, lens_k, dim=-1)
            rec["lens"] = [[{"id": int(i), "p": round(float(pp), 3)} for pp, i in zip(lv[t], li[t])] for t in range(T)]
            if n_pass == layer:                                                   # the chosen layer in detail
                rec["detail"] = {
                    "ln_in": r(strip(h_in[0])), "q": r(q[0], 2), "k": r(ke[0], 2), "v": r(ve[0], 2),
                    "scores": r(scores[0], 2), "heads": r(strip(heads[0], 16)), "attn_out": r(strip(attn_out[0])),
                    "mlp_out": r(strip(mlp_out[0])), "after": r(strip(x[0])),
                }
            out["layers"].append(rec)
            n_pass += 1
    final = model.lm_head(model.transformer.ln_f(x))[0, -1].float()
    tv, ti = torch.topk(final, top_n)
    out["logits"] = [{"id": int(i), "logit": round(float(v), 3)} for v, i in zip(tv, ti)]
    out["config"] = {"n_layer": cfg.n_layer, "passes": n_pass, "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head or cfg.n_head,
                     "n_embd": cfg.n_embd, "head_dim": hd, "pos_encoding": cfg.pos_encoding, "norm": cfg.norm,
                     "mlp": cfg.mlp, "qk_norm": cfg.qk_norm, "n_experts": cfg.n_experts, "moe_top_k": cfg.moe_top_k,
                     "block_repeat": cfg.block_repeat, "vocab_size": cfg.vocab_size, "block_size": cfg.block_size,
                     "bias": cfg.bias}
    out["layer"] = layer
    return out
