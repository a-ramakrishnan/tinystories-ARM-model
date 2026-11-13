# How a model writes text, step by step: the data behind the web app's "Writing text" page.
#
# The page compares ways of producing text on the same footing: the same prompts (the start of real held-out
# documents, so there's a true continuation to score against), the same tokenizer and the same length. This file
# records the autoregressive side: one model pass per token, left to right, with what was chosen at each step,
# how confident the model was and how long it took, plus the model's loss on the real continuation.
#
# Results for a saved model go to models/<id>/textgen.json.

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import registry
from inference import generate_stream, load_model, load_model_config
from tokenization import get_tokenizer

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
RESULTS = "textgen.json"
BASE_MODEL = "mini-fineweb-edu-202611130718-14ed26"  # the project's base model (docs/plan.md): the default here

# held-out sets the prompts come from: (data folder, title). Documents are separated by <|endoftext|>.
SOURCES = [("baominhtran__mini-fineweb-edu-small", "Mini FineWeb-Edu"), ("roneneldan__tinystories", "TinyStories")]
PROMPT_TOKENS, REF_TOKENS, PER_SOURCE = 32, 64, 4


def heldout_prompts(prompt_tokens=PROMPT_TOKENS, ref_tokens=REF_TOKENS, per_source=PER_SOURCE):
    """The fixed prompt set: the first `prompt_tokens` of documents from each held-out set, with the next
    `ref_tokens` as the real continuation. Documents are picked evenly through the file, so the set is the same
    every time."""
    enc = get_tokenizer()
    eot, need = enc.eot_token, prompt_tokens + ref_tokens
    out = []
    for key, title in SOURCES:
        path = os.path.join(DATA_ROOT, key, "validation.bin")
        if not os.path.exists(path):
            continue
        data = np.memmap(path, dtype=np.uint16, mode="r")
        head = np.asarray(data[:min(len(data), 4_000_000)])
        starts = np.flatnonzero(head == eot) + 1
        # documents long enough to hold the prompt and the whole continuation
        ok = [s for s, e in zip(starts, list(starts[1:]) + [len(head)]) if e - s > need]
        for i in range(per_source):
            if not ok:
                break
            s = int(ok[(i * len(ok)) // per_source])
            ids = head[s:s + need].astype(int).tolist()
            out.append({"key": f"{key.split('__')[-1]}-{i + 1}", "source": title, "prompt_ids": ids[:prompt_tokens],
                        "ref_ids": ids[prompt_tokens:], "prompt": enc.decode(ids[:prompt_tokens]),
                        "reference": enc.decode(ids[prompt_tokens:])})
    return out


@torch.no_grad()
def continuation_loss(model, prompt_ids, ref_ids, device):
    """The model's loss on the real continuation, reading it token by token (teacher forcing): the average
    surprise, in nats, of each real next token. Returns the mean, the per-token values, and the share of real
    tokens the model ranked first."""
    ids = (prompt_ids + ref_ids)[-model.config.block_size:]
    n = len(ref_ids)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x, all_logits=True)
    target = torch.tensor(ids[-n:], device=device)
    nll = F.cross_entropy(logits[0, -n:].float(), target, reduction="none")
    top1 = (logits[0, -n:].argmax(-1) == target).float().mean().item()
    per = nll.tolist()
    return sum(per) / n, per, top1


def ar_trace(model, prompt_ids, device, n_tokens, temperature=0.0, top_k=None, seed=1234, stop_token=None):
    """Generate left to right and record every step: the token chosen, the model's probability for it, the top
    alternatives and the time. Step k writes position k: one model pass per token (with the KV cache)."""
    enc = get_tokenizer()
    torch.manual_seed(seed)
    steps, stats = [], {}
    for k, ev in enumerate(generate_stream(model, prompt_ids, device, n_tokens, temperature, top_k, stats=stats,
                                           stop_token=stop_token)):
        steps.append({"step": k + 1, "pos": k, "id": ev["id"], "text": enc.decode([ev["id"]]),
                      "prob": round(ev["prob"], 4), "ms": round(ev["ms"], 2),
                      "alternatives": [{"text": enc.decode([i]), "prob": round(p, 4)} for i, p in ev["alternatives"]]})
    out = [s["id"] for s in steps]
    times = [s["ms"] for s in steps]
    total = times[-1] if times else 0.0
    decode = total - times[0] if len(times) > 1 else 0.0
    return {"steps": steps, "text": enc.decode(out), "tokens": len(out), "model_passes": len(out),
            "total_ms": round(total, 1), "first_ms": round(times[0], 1) if times else None,
            "ms_per_token": round(decode / (len(out) - 1), 2) if len(out) > 1 else None,
            "tokens_per_sec": round((len(out) - 1) / (decode / 1000), 1) if len(out) > 1 and decode > 0 else None,
            "avg_prob": round(sum(s["prob"] for s in steps) / len(steps), 4) if steps else None,
            "stopped": stats.get("stopped", False), **repetition(out)}


def repetition(ids):
    """Share of distinct token pairs, and share of 4-token runs seen earlier in the same text."""
    pairs = list(zip(ids, ids[1:]))
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    seen, repeats = set(), 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return {"distinct_2": round(len(set(pairs)) / len(pairs), 3) if pairs else None,
            "repeated_4grams": round(repeats / len(grams), 3) if grams else None}


def run_prompt(model, p, device, n_tokens=REF_TOKENS, temperature=0.0, top_k=None, seed=1234):
    """One prompt: the trace, plus how well the model predicts the real continuation (loss, and the share of
    real tokens it ranks first)."""
    tr = ar_trace(model, p["prompt_ids"], device, n_tokens, temperature, top_k, seed)
    loss, per, top1 = (continuation_loss(model, p["prompt_ids"], p["ref_ids"], device) if p.get("ref_ids")
                       else (None, [], None))
    return {**tr, "ref_loss": round(loss, 4) if loss is not None else None,
            "ref_ppl": round(math.exp(loss), 2) if loss is not None else None,
            "ref_token_loss": [round(v, 3) for v in per], "ref_top1": round(top1, 3) if top1 is not None else None}


def load_library_model(model_id, device):
    weights, cfg_path = registry.paths_of(model_id)
    config, _ = load_model_config(cfg_path)
    return load_model(config, weights, device)


def warm_up(model, device):
    """A short throwaway generation, so the first measured prompt doesn't pay for GPU start-up."""
    for _ in ar_trace(model, [get_tokenizer().eot_token], device, 8)["steps"]:
        pass


def run_set(model, model_id, device, n_tokens=REF_TOKENS, temperature=0.0, top_k=None, seed=1234, progress=None):
    """Every prompt in the fixed set; saves models/<id>/textgen.json and returns it."""
    prompts = heldout_prompts()
    warm_up(model, device)
    rows = []
    for i, p in enumerate(prompts):
        if progress:
            progress(i, len(prompts))
        r = run_prompt(model, p, device, n_tokens, temperature, top_k, seed)
        rows.append({k: p[k] for k in ("key", "source", "prompt", "reference")} | r)
    mean = lambda k: (round(sum(r[k] for r in rows if r[k] is not None) / max(1, sum(r[k] is not None for r in rows)), 4)  # noqa: E731
                      if rows else None)
    res = {"method": "autoregressive", "measured_at": time.strftime("%Y-%m-%d %H:%M"), "device": device,
           "settings": {"tokens": n_tokens, "temperature": temperature, "top_k": top_k, "seed": seed,
                        "prompt_tokens": PROMPT_TOKENS, "ref_tokens": REF_TOKENS},
           "summary": {k: mean(k) for k in ("ref_loss", "ref_top1", "avg_prob", "tokens_per_sec", "ms_per_token",
                                            "distinct_2", "repeated_4grams", "tokens", "model_passes")},
           "prompts": rows}
    if res["summary"]["ref_loss"] is not None:
        res["summary"]["ref_ppl"] = round(math.exp(res["summary"]["ref_loss"]), 2)
    if model_id:
        with open(os.path.join(registry.model_dir(model_id), RESULTS), "w") as f:
            json.dump(res, f, indent=1)
    return res


def load_results(model_id):
    try:
        with open(os.path.join(registry.model_dir(model_id), RESULTS)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
