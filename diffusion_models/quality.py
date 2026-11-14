# How good is a diffusion model's text? Generates continuations of the fixed held-out prompts (textgen.py: the
# first 32 tokens of held-out documents) with given sampler settings and scores them:
#   - judge loss: the base autoregressive model's loss on the generated continuation (nats per token; lower reads
#     more like real text to it). Repeating one word also scores low, so read it with the repetition numbers.
#   - repeated 4-grams: share of 4-token runs already seen earlier in the same continuation (0 is best);
#   - distinct 2-grams: share of distinct token pairs (1 is best);
#   - length: tokens before the model ended the document.
# For scale it also scores the real continuations and the base autoregressive model's own sampled text.
#
#   python3 -m diffusion_models.quality diffusion_models/runs/fineweb-edu-small-colab --source "Mini FineWeb-Edu"

import argparse
import json
import os
import time

import torch

import textgen
from train_config import setup_device

from .model import load
from .sample import diffusion_generate

# name -> diffusion_generate settings
CONFIGS = {
    "keep (confident first), top-k 50": dict(top_k=50, repetition_penalty=1.0),
    "random order, temp 0.8": dict(sampler="random", temperature=0.8, repetition_penalty=1.0),
    "random order, temp 1.0": dict(sampler="random", repetition_penalty=1.0),
    "keep, repetition 1.3": dict(top_k=50, repetition_penalty=1.3),
    "keep, blocks of 16": dict(top_k=50, block_size=16, repetition_penalty=1.0),
    "keep, blocks of 8, rep 1.3": dict(top_k=50, block_size=8, repetition_penalty=1.3),
    "keep, remask 0.1": dict(top_k=50, remask=0.1, repetition_penalty=1.0),
    "keep, noise level from masks": dict(top_k=50, time_from="mask", repetition_penalty=1.0),
    # around the repetition penalty
    "keep, rep 1.15": dict(top_k=50, repetition_penalty=1.15),
    "keep, rep 1.2": dict(top_k=50, repetition_penalty=1.2),
    "keep, rep 1.5": dict(top_k=50, repetition_penalty=1.5),
    "keep, rep 1.2, blocks of 16": dict(top_k=50, repetition_penalty=1.2, block_size=16),
    "keep, rep 1.2, blocks of 32": dict(top_k=50, repetition_penalty=1.2, block_size=32),
    "keep, rep 1.2, 128 steps": dict(top_k=50, repetition_penalty=1.2, diffusion_steps=128),
    "keep, rep 1.2, temp 0.8": dict(top_k=50, repetition_penalty=1.2, temperature=0.8),
    "keep, rep 1.2, remask 0.1": dict(top_k=50, repetition_penalty=1.2, remask=0.1),
    "keep, rep 1.2, noise from masks": dict(top_k=50, repetition_penalty=1.2, time_from="mask"),
    "keep, rep 1.2, top-p 0.9": dict(repetition_penalty=1.2, top_p=0.9),
    # one reveal per token id per step
    "keep, distinct": dict(top_k=50, distinct=True, repetition_penalty=1.0),
    "keep, distinct, rep 1.1": dict(top_k=50, distinct=True, repetition_penalty=1.1),
    "keep, distinct, rep 1.2": dict(top_k=50, distinct=True, repetition_penalty=1.2),
    "keep, distinct, rep 1.2, blocks of 16": dict(top_k=50, distinct=True, repetition_penalty=1.2, block_size=16),
    "keep, distinct, blocks of 16": dict(top_k=50, distinct=True, block_size=16, repetition_penalty=1.0),
    "keep, distinct, 16 steps": dict(top_k=50, distinct=True, diffusion_steps=16, repetition_penalty=1.0),
    "keep, 16 steps": dict(top_k=50, diffusion_steps=16, repetition_penalty=1.0),
    # left to right: blocks of 1 is one token per step in reading order (an AR order with a bidirectional model)
    "left to right (blocks of 1)": dict(top_k=50, block_size=1, repetition_penalty=1.0),
    "left to right, rep 1.1": dict(top_k=50, block_size=1, repetition_penalty=1.1),
    "left to right, rep 1.2": dict(top_k=50, block_size=1, repetition_penalty=1.2),
    "keep, blocks of 4": dict(top_k=50, block_size=4, repetition_penalty=1.0),
    "keep, blocks of 4, rep 1.2": dict(top_k=50, block_size=4, repetition_penalty=1.2),
    "keep, blocks of 8, rep 1.2": dict(top_k=50, block_size=8, repetition_penalty=1.2),
}


def score(judge, device, prompt_ids, gen_ids):
    rep = textgen.repetition(gen_ids)
    loss = textgen.continuation_loss(judge, prompt_ids, gen_ids, device)[0] if len(gen_ids) > 1 else None
    return {"judge_loss": loss, "length": len(gen_ids), **rep}


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(v) / len(v), 3) if v else None


def summarize(rows):
    return {k: mean(rows, k) for k in ("judge_loss", "repeated_4grams", "distinct_2", "length", "seconds")}


def evaluate(run_dir, configs, source=None, n_tokens=64, steps=None, seeds=(1, 2), per_source=8, say=print, reference=True):
    dev = setup_device()
    device = dev.device
    model = load(run_dir, device)
    judge = textgen.load_library_model(textgen.BASE_MODEL, device)
    judge.eval()
    prompts = [p for p in textgen.heldout_prompts(per_source=per_source) if source is None or p["source"] == source]
    say(f"{len(prompts)} prompts x {len(seeds)} seeds, {n_tokens} new tokens; judge: {textgen.BASE_MODEL}")

    out = {"run": os.path.basename(run_dir.rstrip("/")), "source": source, "n_tokens": n_tokens, "seeds": list(seeds),
           "judge": textgen.BASE_MODEL, "results": {}}
    if reference:
        ref = [score(judge, device, p["prompt_ids"], p["ref_ids"][:n_tokens]) for p in prompts]
        ar = []
        for p in prompts:
            for sd in seeds:
                tr = textgen.ar_trace(judge, p["prompt_ids"], device, n_tokens, temperature=1.0, top_k=50, seed=sd)
                ar.append(score(judge, device, p["prompt_ids"], [s["id"] for s in tr["steps"]]))
        out["results"]["real continuation"] = {"summary": summarize(ref)}
        out["results"]["base AR model, top-k 50 (judge's own text)"] = {"summary": summarize(ar)}

    for name, kw in configs.items():
        rows, examples = [], []
        for p in prompts:
            for sd in seeds:
                kw2 = {"diffusion_steps": steps, **kw}
                r = diffusion_generate(model, max_new_tokens=n_tokens, record_steps=False, seed=sd,
                                       prompt_ids=p["prompt_ids"], **kw2)
                rows.append({**score(judge, device, p["prompt_ids"], r["ids"]), "seconds": r["seconds"]})
                if sd == seeds[0] and len(examples) < 3:
                    examples.append({"prompt": p["prompt"], "completion": r["completion"]})
        out["results"][name] = {"settings": kw, "summary": summarize(rows), "examples": examples}
    return out


def table(out):
    lines = [f"{'settings':48s} {'judge':>6s} {'rep4':>6s} {'dist2':>6s} {'len':>5s}"]
    for name, r in out["results"].items():
        s = r["summary"]
        f = lambda v, w=6: f"{v:{w}.3f}" if isinstance(v, float) else f"{str(v):>{w}s}"
        lines.append(f"{name[:48]:48s} {f(s['judge_loss'])} {f(s['repeated_4grams'])} {f(s['distinct_2'])} {f(s['length'], 5)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Score a diffusion model's generated text.")
    ap.add_argument("run_dir")
    ap.add_argument("--source", default=None, help='"Mini FineWeb-Edu" or "TinyStories" (default: both)')
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated substrings of config names to run")
    ap.add_argument("--no-reference", action="store_true", help="skip the real text and AR rows")
    ap.add_argument("--out", default=None, help="JSON file (default <run_dir>/quality.json)")
    a = ap.parse_args()
    cfgs = CONFIGS if not a.only else {k: v for k, v in CONFIGS.items() if any(s in k for s in a.only.split(","))}
    t0 = time.time()
    out = evaluate(a.run_dir, cfgs, a.source, a.tokens, a.steps, reference=not a.no_reference)
    path = a.out or os.path.join(a.run_dir, "quality.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(table(out))
    print(f"({time.time() - t0:.0f} s; saved {path})")


if __name__ == "__main__":
    main()
