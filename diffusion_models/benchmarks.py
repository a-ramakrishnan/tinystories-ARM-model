# The five zero-shot benchmarks (benchmarks.py: HellaSwag, LAMBADA, ARC-Easy, PIQA, Winogrande) for a masked
# diffusion model: the same questions, prompts and metrics, so the numbers sit next to the autoregressive models'.
#
# An autoregressive model scores an answer by the sum of its tokens' log probabilities. A diffusion model has no
# exact log probability, so each answer gets the likelihood bound (as in LLaDA): with the context visible, mask l of
# the answer's n tokens at random, and take n/l times the log probability of the masked tokens; averaged over
# `samples` draws (l spread evenly over 1..n), that's an estimate of a lower bound on log p(answer | context).
# LAMBADA's accuracy masks the whole last word and checks the model's most likely token at every position.
#
#   python3 -m diffusion_models.benchmarks diffusion_models/runs/<run> [--tasks ...] [--limit N] [--samples 8]
#
# Results go to <run>/benchmarks.json, in the same layout as models/<id>/benchmarks.json; the app's Benchmarks page
# shows them next to the autoregressive models.

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import benchmarks as B
from train_config import setup_device

from .model import load
from .tokens import EOT_ID, MASK_ID


@torch.no_grad()
def score_requests(model, enc, requests, device, samples=8, batch=64, seed=0):
    """For each (context, continuation): (estimated log likelihood of the continuation, greedy, n_tokens)."""
    T, L_max = model.cfg.diffusion_steps, model.cfg.seq_len
    g = torch.Generator().manual_seed(seed)
    rows = []          # (request index, ids, answer start, masked positions, weight) ; weight 0 = the greedy row
    n_tok = []
    for r, (ctx, cont) in enumerate(requests):
        c_ids, k_ids = enc.encode_ordinary(ctx), enc.encode_ordinary(cont)
        ids = ([EOT_ID] + c_ids + k_ids)[-L_max:]       # cut from the left if longer than the window
        n = min(len(k_ids), len(ids) - 1)
        n_tok.append(n)
        start = len(ids) - n
        for k in range(samples):
            l = max(1, min(n, math.ceil(n * (k + 1) / samples)))   # masked count spread evenly over 1..n
            pos = (torch.randperm(n, generator=g)[:l] + start).tolist()
            rows.append((r, ids, start, pos, n / l))
        rows.append((r, ids, start, list(range(start, len(ids))), 0.0))
    ll = [0.0] * len(requests)
    greedy = [True] * len(requests)
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i][1]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        Lb = max(len(rows[i][1]) for i in idx)
        x = torch.full((len(idx), Lb), EOT_ID, dtype=torch.long)
        y = torch.full((len(idx), Lb), -1, dtype=torch.long)
        att = torch.zeros(len(idx), Lb, dtype=torch.bool)
        t = torch.zeros(len(idx), dtype=torch.long)
        for b, i in enumerate(idx):
            r, ids, start, pos, w = rows[i]
            x[b, :len(ids)] = torch.tensor(ids)
            att[b, :len(ids)] = True
            p = torch.tensor(pos)
            y[b, p] = x[b, p]
            x[b, p] = MASK_ID
            # the noise level the model saw in training: the share of the whole input that is masked
            t[b] = max(1, min(T, math.ceil(T * len(pos) / len(ids))))
        x, y, att, t = x.to(device), y.to(device), att.to(device), t.to(device)
        logits = model(x, t, attention_mask=att).float()[..., :MASK_ID]
        mask = y >= 0
        lp = (logits.gather(2, y.clamp(min=0)[..., None])[..., 0] - torch.logsumexp(logits, -1)) * mask
        sums = lp.sum(1).tolist()
        hit = ((logits.argmax(-1) == y) | ~mask).all(1).tolist()
        for b, i in enumerate(idx):
            r, _, _, _, w = rows[i]
            if w:
                ll[r] += w * sums[b] / samples
            else:
                greedy[r] = bool(hit[b])
        if (s // batch) % 200 == 0 and len(order) > 20000:
            print(f"    {s:,} of {len(order):,} rows scored", flush=True)
    return [(ll[r], greedy[r], n_tok[r]) for r in range(len(requests))]


def run(run_dir, tasks=B.TASKS, limit=None, samples=8, save=True, device=None):
    import tiktoken
    device = device or setup_device().device
    model = load(run_dir, device).eval()
    enc = tiktoken.get_encoding("gpt2")
    old_score = B.score_requests
    B.score_requests = lambda m, e, reqs, d, batch=16: score_requests(m, e, reqs, d, samples=samples)
    results = {}
    try:
        for t in tasks:
            print(f"{os.path.basename(run_dir)}: {t} …", flush=True)
            results[t] = B.run_task(model, enc, device, t, limit)
            print(f"  {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in results[t].items()})}", flush=True)
    finally:
        B.score_requests = old_score
    if save and not limit:
        path = os.path.join(run_dir, "benchmarks.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update({"results": {**old.get("results", {}), **results}, "context": model.cfg.seq_len,
                    "measured_at": time.strftime("%Y-%m-%d %H:%M"), "samples": samples,
                    "harness": "lm-evaluation-harness style, zero-shot; answers scored by the likelihood bound"})
        json.dump(old, open(path, "w"), indent=1)
    return results


def load_results(run_dir):
    path = os.path.join(run_dir, "benchmarks.json")
    return json.load(open(path)) if os.path.exists(path) else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Zero-shot benchmarks for a diffusion model.")
    ap.add_argument("runs", nargs="+", help="run folders (diffusion_models/runs/<run>)")
    ap.add_argument("--tasks", default=",".join(B.TASKS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions per task (not saved)")
    ap.add_argument("--samples", type=int, default=8, help="masking draws per answer")
    a = ap.parse_args()
    for r in a.runs:
        run(r, [t.strip() for t in a.tasks.split(",") if t.strip()], a.limit, a.samples)
