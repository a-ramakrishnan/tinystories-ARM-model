# Evaluation suite: the same tests for every model, so models can be compared fairly.
#
#   1. Held-out loss: next-token loss (and perplexity) on fixed windows of each dataset's validation.bin,
#      the same token positions for every model.
#   2. Generation: a fixed set of prompts, continued twice with a fixed seed, greedy and with the Balanced
#      sampling preset. Measures repetition, variety, confidence and speed, and keeps the texts.
#   3. Size: parameters, weights in memory, KV cache per token.
#
# Results are saved with each saved model (models/<id>/eval.json, see registry.py).
#
# Command line:
#   python3 evaluate.py                     evaluate every saved model not yet evaluated
#   python3 evaluate.py --all               evaluate every saved model again
#   python3 evaluate.py <model id> ...      evaluate these models
#   python3 evaluate.py --table             print the comparison table

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import registry
from inference import generate_stream, kv_bytes_per_token, load_model, load_model_config
from tokenization import EOT, get_tokenizer

SUITE_VERSION = 1
EVAL_WINDOWS = 64          # windows per dataset
EVAL_CONTEXT = 256         # tokens per window (or the model's context window, if smaller)
GEN_TOKENS = 100
GEN_SEED = 1234
BALANCED = {"temperature": 0.8, "top_k": None, "top_p": 0.9, "repetition_penalty": 1.15}
PROMPTS = [
    "Once upon a time there was a little girl who",
    "Tom and his dog went to the park.",
    "The water cycle is the process by which",
    "Photosynthesis is the process",
    "In the history of science,",
    "The best way to learn a new language is",
]

DATASET_TITLES = {
    "roneneldan__tinystories": "TinyStories",
    "baominhtran__mini-fineweb-edu-small": "Mini FineWeb-Edu",
    "salesforce__wikitext-wikitext-2-raw-v1": "WikiText-2",
    "salesforce__wikitext-wikitext-103-raw-v1": "WikiText-103",
}


def eval_datasets(data_root=registry.DATA_ROOT):
    """Datasets on this machine with a tokenized validation split: [{key, title, tokens}]."""
    out = []
    if not os.path.isdir(data_root):
        return out
    for d in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, d, "validation.bin")
        if d.startswith("mix-") or not os.path.exists(path):
            continue
        n = os.path.getsize(path) // 2
        if n < EVAL_CONTEXT + 1:
            continue
        title = DATASET_TITLES.get(d) or ("Tiny Shakespeare" if "input.txt" in d else d)
        out.append({"key": d, "title": title, "tokens": n, "path": path})
    return out


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def heldout_loss(model, path, device, n_windows=EVAL_WINDOWS, context=EVAL_CONTEXT, batch=16):
    """Mean next-token loss on n_windows evenly spaced windows of a validation.bin file."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    T = min(context, model.config.block_size)
    starts = np.linspace(0, len(data) - T - 1, n_windows).astype(np.int64)
    total, count = 0.0, 0
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        x = torch.from_numpy(np.stack([data[j:j + T].astype(np.int64) for j in s])).to(device)
        y = torch.from_numpy(np.stack([data[j + 1:j + 1 + T].astype(np.int64) for j in s])).to(device)
        logits, _ = model(x, y)  # with targets the model returns logits for every position
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum")
        total += float(loss)
        count += y.numel()
    mean = total / count
    return {"loss": round(mean, 4), "perplexity": round(math.exp(mean), 2), "context": T, "tokens": count}


def _repetition(ids):
    pairs = list(zip(ids, ids[1:]))
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    seen, repeats = set(), 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return (len(set(pairs)) / len(pairs) if pairs else None), (repeats / len(grams) if grams else None)


def _generate(model, enc, prompt, device, settings):
    torch.manual_seed(GEN_SEED)
    ids = enc.encode_ordinary(prompt)
    out, probs, times = [], [], []
    extras = {k: v for k, v in settings.items() if k not in ("temperature", "top_k")}
    for ev in generate_stream(model, ids, device, GEN_TOKENS, settings["temperature"], settings.get("top_k"),
                              stop_token=EOT, sampling=extras):
        out.append(ev["id"])
        probs.append(ev["prob"])
        times.append(ev["ms"])
    _sync(device)
    distinct, repeated = _repetition(out)
    decode = (times[-1] - times[0]) / 1000 if len(times) > 1 else None
    return {"text": enc.decode(out), "tokens": len(out), "distinct_2": distinct, "repeated_4grams": repeated,
            "avg_prob": sum(probs) / len(probs) if probs else None, "ttft_ms": times[0] if times else None,
            "tokens_per_sec": (len(out) - 1) / decode if decode else None}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def evaluate_model(model_id, device, datasets=None, progress=None, root=registry.LIB_ROOT):
    """Run the suite on one saved model and store the results with it."""
    weights, config_path = registry.paths_of(model_id, root)
    config, info = load_model_config(config_path)
    if info.get("model_type", "autoregressive") != "autoregressive":
        raise ValueError("The evaluation suite only knows autoregressive models so far.")
    model = load_model(config, weights, device)
    enc = get_tokenizer()
    datasets = eval_datasets() if datasets is None else datasets
    say = progress or (lambda msg: None)
    t0 = time.time()

    losses = {}
    for d in datasets:
        say(f"held-out loss on {d['title']}")
        losses[d["key"]] = {**heldout_loss(model, d["path"], device), "title": d["title"]}

    samples, greedy, balanced = [], [], []
    for i, prompt in enumerate(PROMPTS):
        say(f"generating ({i + 1} of {len(PROMPTS)})")
        g = _generate(model, enc, prompt, device, {"temperature": 0, "top_k": None})
        b = _generate(model, enc, prompt, device, BALANCED)
        greedy.append(g)
        balanced.append(b)
        samples.append({"prompt": prompt, "greedy": g["text"], "balanced": b["text"]})

    n_params = sum(p.numel() for p in model.parameters())
    results = {
        "suite_version": SUITE_VERSION, "evaluated_at": time.strftime("%Y-%m-%d %H:%M"), "device": device,
        "seconds": round(time.time() - t0, 1),
        "heldout": losses,
        "generation": {
            "greedy_repeated_4grams": _mean([g["repeated_4grams"] for g in greedy]),
            "balanced_repeated_4grams": _mean([b["repeated_4grams"] for b in balanced]),
            "balanced_distinct_2": _mean([b["distinct_2"] for b in balanced]),
            "balanced_avg_prob": _mean([b["avg_prob"] for b in balanced]),
            "tokens_per_sec": _mean([b["tokens_per_sec"] for b in balanced]),
            "ttft_ms": _mean([b["ttft_ms"] for b in balanced]),
            "settings": {"tokens": GEN_TOKENS, "seed": GEN_SEED, "balanced": BALANCED},
        },
        "size": {"params": n_params, "weights_bytes": n_params * 4, "kv_bytes_per_token": kv_bytes_per_token(config),
                 "context_window": config.block_size},
        "samples": samples,
    }
    registry.save_eval(model_id, results, root)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return results


def comparison_rows(models=None):
    """One row per saved model with its headline numbers (None where not evaluated)."""
    rows = []
    for m in models if models is not None else registry.list_models():
        ev = m.get("eval") or {}
        gen = ev.get("generation") or {}
        rows.append({"id": m["id"], "name": m["name"], "params": m.get("params"), "best_val": m.get("best_val"),
                     "heldout": {k: v["loss"] for k, v in (ev.get("heldout") or {}).items()},
                     "greedy_rep": gen.get("greedy_repeated_4grams"), "balanced_rep": gen.get("balanced_repeated_4grams"),
                     "tok_s": gen.get("tokens_per_sec")})
    return rows


def print_table():
    rows = comparison_rows()
    keys = sorted({k for r in rows for k in r["heldout"]})
    names = {k: v.get("title", k) for m in registry.list_models() for k, v in ((m.get("eval") or {}).get("heldout") or {}).items()}
    titles = [names.get(k, k)[:14] for k in keys]
    print(f"{'model':52s} " + " ".join(f"{t:>14s}" for t in titles) + "   greedy rep  tok/s")
    for r in rows:
        cells = " ".join(f"{r['heldout'][k]:14.3f}" if k in r["heldout"] else f"{'–':>14s}" for k in keys)
        rep = f"{r['greedy_rep'] * 100:9.0f}%" if r["greedy_rep"] is not None else f"{'–':>10s}"
        tps = f"{r['tok_s']:6.0f}" if r["tok_s"] else "     –"
        print(f"{r['name'][:52]:52s} {cells}   {rep}  {tps}")


if __name__ == "__main__":
    from train_config import setup_device
    ap = argparse.ArgumentParser(description="Evaluate saved models on the same tests.")
    ap.add_argument("ids", nargs="*", help="saved model ids (default: every model not yet evaluated)")
    ap.add_argument("--all", action="store_true", help="re-evaluate every saved model")
    ap.add_argument("--table", action="store_true", help="only print the comparison table")
    args = ap.parse_args()
    if not args.table:
        dev = setup_device().device
        todo = args.ids or [m["id"] for m in registry.list_models()
                            if args.all or (m.get("eval") or {}).get("suite_version") != SUITE_VERSION]
        for mid in todo:
            print(f"Evaluating {mid} on {dev}…")
            r = evaluate_model(mid, dev, progress=lambda s: print("  " + s))
            print(f"  done in {r['seconds']} s")
    print_table()
