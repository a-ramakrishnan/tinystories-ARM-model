# The denoiser: a bidirectional transformer (every token sees every other token) that reads a partly masked
# sequence and the noise level t, and predicts the original token at every position. As in v1_original.py.
#
# Changes from v1: weights start small (normal, std 0.02, as in GPT-2) instead of PyTorch's defaults (the tied
# embedding started at std 1, so the first logits were huge); nested tensors are off (unused without padding).

import json
import os
from dataclasses import asdict

import torch
import torch.nn as nn

from .config import DiffusionLMConfig


class DiffusionTransformerLM(nn.Module):
    def __init__(self, cfg: DiffusionLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)   # one vector per noise level t
        enc_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
                                               dropout=cfg.dropout, batch_first=True, activation="gelu",
                                               norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # tied, as in v1
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, timesteps, attention_mask=None):
        """input_ids [B, L]; timesteps [B], integers in 1..T; attention_mask [B, L], True for real tokens.
        Returns logits [B, L, vocab]."""
        B, L = input_ids.shape
        if L > self.cfg.seq_len:
            raise ValueError(f"Sequence length {L} > cfg.seq_len {self.cfg.seq_len}")
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos) + self.time_emb(timesteps).unsqueeze(1)
        x = self.drop(x)
        pad = None if attention_mask is None else ~attention_mask
        x = self.encoder(x, src_key_padding_mask=pad)
        return self.lm_head(self.ln_f(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class GPTDenoiser(nn.Module):
    """The denoiser built from the autoregressive base model's blocks (model.py: RoPE, SwiGLU, no biases, QK-norm)
    with bidirectional attention, plus v1's embedding of the noise level t added to every position."""

    def __init__(self, cfg: DiffusionLMConfig):
        super().__init__()
        from model import GPT, GPTConfig
        self.cfg = cfg
        self.gpt = GPT(GPTConfig(block_size=cfg.seq_len, vocab_size=cfg.vocab_size, n_layer=cfg.n_layers,
                                 n_head=cfg.n_heads, n_embd=cfg.d_model, dropout=cfg.dropout, bias=False,
                                 pos_encoding="rope", mlp="swiglu", qk_norm=True, causal=False))
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)
        nn.init.normal_(self.time_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, timesteps, attention_mask=None):
        """As DiffusionTransformerLM.forward: logits [B, L, vocab]."""
        B, L = input_ids.shape
        if L > self.cfg.seq_len:
            raise ValueError(f"Sequence length {L} > cfg.seq_len {self.cfg.seq_len}")
        g = self.gpt
        x = g.transformer.wte(input_ids) + self.time_emb(timesteps).unsqueeze(1)
        x = g.transformer.drop(x)
        rope = g.rope_tables(0, L, input_ids.device)
        mask = None if attention_mask is None else attention_mask[:, None, None, :].bool()
        for block in g.transformer.h:
            x = block(x, None, rope, mask)
        return g.lm_head(g.transformer.ln_f(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build(cfg: DiffusionLMConfig):
    """The denoiser for a config: v1's encoder or the base model's design."""
    return GPTDenoiser(cfg) if getattr(cfg, "arch", "v1") == "gpt" else DiffusionTransformerLM(cfg)


def save(model, run_dir, name="model.pt", extra=None):
    os.makedirs(run_dir, exist_ok=True)
    m = getattr(model, "_orig_mod", model)   # unwrap torch.compile
    torch.save(m.state_dict(), os.path.join(run_dir, name))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"model_type": "masked-diffusion", "model": asdict(m.cfg), **(extra or {})}, f, indent=2)


def load(run_dir, device="cpu", name="model.pt"):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = DiffusionLMConfig(**json.load(f)["model"])
    model = build(cfg)
    model.load_state_dict(torch.load(os.path.join(run_dir, name), map_location=device))
    return model.to(device).eval()
