"""An original model and its quantized copy, side by side (the app's Quantize page).

- Size: the weight file on disk and the weights in memory.
- Perplexity: held-out loss (evaluate.heldout_loss: the same fixed windows of each validation split as the evaluation
  suite) and e^loss, for both models.
- Text: the same prompt continued by both, greedy by default, so any difference comes from the weights.

The result is kept with the copy (models/<copy>/quantize.json).
"""

import json
import os
import time

import torch

import registry
from infra.results import _title
from evaluate import _repetition, _sync, eval_datasets, heldout_loss
from inference import generate_stream, load_model, load_model_config
from quantization.checkpoint import quant_info
from quantization.rtn import weight_bytes
from tokenization import EOT, get_tokenizer

RESULT = "quantize.json"
PROMPT = "The water cycle is the process by which"
TOKENS = 80


def load(model_id, device):
    weights, config_path = registry.paths_of(model_id)
    config, _ = load_model_config(config_path)
    return load_model(config, weights, device)


def memory_bytes(model):
    return weight_bytes(model) if quant_info(model) else sum(p.numel() * p.element_size() for p in model.parameters())


@torch.no_grad()
def generate(model, prompt, device, tokens=TOKENS, temperature=0.0, seed=1234):
    """Continue `prompt`: text, speed, confidence and repetition."""
    enc = get_tokenizer()
    torch.manual_seed(seed)
    out, probs, times = [], [], []
    ids = enc.encode_ordinary(prompt)[-model.config.block_size:]
    for ev in generate_stream(model, ids, device, tokens, temperature, None, stop_token=EOT):
        out.append(ev["id"])
        probs.append(ev["prob"])
        times.append(ev["ms"])
    _sync(device)
    decode = (times[-1] - times[0]) / 1000 if len(times) > 1 else None
    distinct, repeated = _repetition(out)
    return {"text": enc.decode(out), "tokens": len(out),
            "tokens_per_sec": round((len(out) - 1) / decode, 1) if decode else None,
            "avg_prob": round(sum(probs) / len(probs), 3) if probs else None,
            "repeated_4grams": repeated}


@torch.no_grad()
def perplexities(model, device, datasets=None):
    out = {}
    for d in datasets if datasets is not None else eval_datasets():
        r = heldout_loss(model, d["path"], device)
        out[d["key"]] = {"title": _title(d["key"], d["title"]), "loss": r["loss"], "perplexity": r["perplexity"], "tokens": r["tokens"]}
    return out


def compare(original_id, copy_id, device, say=print, prompt=PROMPT, tokens=TOKENS):
    """Measure both models and save the result with the copy. Returns it."""
    result = {"original": original_id, "copy": copy_id, "measured_at": time.strftime("%Y-%m-%d %H:%M"),
              "device": device, "prompt": prompt, "decoding": "greedy"}
    datasets = eval_datasets()
    for role, mid in (("original_model", original_id), ("quantized_model", copy_id)):
        meta = registry.get_model(mid)
        say(f"{'original' if role == 'original_model' else 'quantized'} model: perplexity on "
            f"{len(datasets)} held-out set{'s' if len(datasets) != 1 else ''}")
        model = load(mid, device)
        r = {"id": mid, "name": meta["name"], "file_bytes": meta["bytes"], "memory_bytes": memory_bytes(model),
             "quantized": quant_info(model), "perplexity": perplexities(model, device, datasets)}
        say(f"{'original' if role == 'original_model' else 'quantized'} model: writing {tokens} tokens")
        r["sample"] = generate(model, prompt, device, tokens)
        result[role] = r
        del model
    with open(os.path.join(registry.model_dir(copy_id), RESULT), "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_result(copy_id):
    try:
        with open(os.path.join(registry.model_dir(copy_id), RESULT)) as f:
            return json.load(f)
    except (OSError, ValueError, KeyError):
        return None
