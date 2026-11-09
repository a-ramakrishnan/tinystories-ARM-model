# Inference: load trained weights and generate text from a prompt

import json
from dataclasses import asdict

import torch

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
