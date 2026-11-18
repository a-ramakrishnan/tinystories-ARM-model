# F4: the autoregressive and the masked diffusion model, compared like for like.
#
# Two pairs, each the same size, data, number of training tokens and tokenizer, so only the training objective
# differs:
#   124m         the base model (stage 2, 2.5B tokens) and the F6 diffusion model (2.5B tokens of the same mix)
#   51m-stories  Medium 51M · TinyStories · 20k steps and the TinyStories diffusion run (164M tokens each)
#
# What is measured:
#   1. Held-out text: the AR model's loss and the diffusion model's likelihood bound on exactly the same windows
#      (evaluate.py's 64 evenly spaced 256-token windows of each validation.bin). The bound is an upper limit on the
#      diffusion model's real loss, so the true gap is at most the measured one.
#   2. The five benchmarks, from each model's benchmarks.json (the diffusion side scored by the likelihood bound).
#   3. Writing: the 8 held-out prompts (textgen.py) × 2 seeds, 64 new tokens. AR sampled (top-k 50, temperature 1)
#      and greedy; diffusion with the recommended sampler (confident first, repetition penalty 1.2, blocks of 16)
#      at 16, 32 and 64 passes. Each text is scored by a judge that is neither model, OpenAI's GPT-2 small (its loss
#      on the text), and for repetition, variety, time and tokens per second.
#
#   python3 -m diffusion_models.compare [124m 51m-stories] [--skip heldout,writing]
#
# Results: outputs/f4/<pair>.json, and Markdown tables printed at the end.
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "outputs", "f4")
RUNS = os.path.join(ROOT, "diffusion_models", "runs")
JUDGE = "webtext-openai-201902140000-f3da9f"            # OpenAI GPT-2 small: trained on neither model's data
PAIRS = {
    "124m": {"title": "124M, FineWeb-Edu full + Cosmopedia, 2.5B tokens",
             "ar": "mini-fineweb-edu-ful-202611151103-bb160c", "diffusion": "f6-124m-stage2-2p5b"},
    "51m-stories": {"title": "51M, TinyStories, 164M tokens",
                    "ar": "tinystories-202611100814-5da893", "diffusion": "tinystories-colab"},
}
DIFFUSION_PASSES = (16, 32, 64)
SAMPLER = dict(sampler="keep", top_k=50, repetition_penalty=1.2, block_size=16)
SEEDS = (1, 2)
TOKENS = 64
LEVELS = 16                 # noise levels per held-out window (rotated across windows, so all T are covered)


# ---------- 1. held-out text ----------

@torch.no_grad()
def heldout_bound(model, path, device, n_windows=64, context=256, levels=LEVELS, batch=32, seed=0):
    """The likelihood bound (nats per token) on the same windows as evaluate.heldout_loss. Each window is masked at
    `levels` noise levels spread evenly over 1..T (offset per window); each row contributes
    (cross-entropy summed over the masked tokens) / (t/T) / L, the masked diffusion bound; the mean over rows and
    windows estimates the expected bound."""
    from .noising import corrupt_with_mask
    T, mask_id = model.cfg.diffusion_steps, model.cfg.mask_id
    data = np.memmap(path, dtype=np.uint16, mode="r")
    L = min(context, model.cfg.seq_len)
    starts = np.linspace(0, len(data) - L - 1, n_windows).astype(np.int64)
    rows = []
    for w, s in enumerate(starts):
        for k in range(levels):
            t = 1 + (k * T // levels + w) % T          # evenly spread, shifted by one level per window
            rows.append((int(s), t))
    g = torch.Generator(device="cpu").manual_seed(seed)
    total, n = 0.0, 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        x = torch.from_numpy(np.stack([data[s:s + L].astype(np.int64) for s, _ in chunk]))
        t = torch.tensor([t for _, t in chunk])
        noisy, labels, masked = corrupt_with_mask(x, t, T, mask_id, generator=g)
        logits = model(noisy.to(device), timesteps=t.to(device)).float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.to(device).view(-1), ignore_index=-100,
                             reduction="none").view(len(chunk), L)
        total += float((ce.sum(1) / (t.to(device).float() / T)).sum() / L)
        n += len(chunk)
    return {"bound": round(total / n, 4), "context": L, "windows": n_windows, "levels": levels}


def ar_heldout(model_id):
    """The AR model's held-out losses, as the evaluation suite saved them."""
    import registry
    p = os.path.join(registry.model_dir(model_id), "eval.json")
    ev = json.load(open(p)) if os.path.exists(p) else {}
    return {k: {"loss": v["loss"], "title": v.get("title", k)} for k, v in (ev.get("heldout") or {}).items()}


def heldout(pair, device, say):
    import evaluate
    from .model import load
    ar = ar_heldout(pair["ar"])
    dm = load(os.path.join(RUNS, pair["diffusion"]), device)
    out = {}
    for d in evaluate.eval_datasets():
        if d["key"] not in ar:
            continue
        say(f"held-out bound on {d['title']}")
        b = heldout_bound(dm, d["path"], device)
        out[d["key"]] = {"title": ar[d["key"]]["title"], "ar_loss": ar[d["key"]]["loss"], "diffusion_bound": b["bound"],
                         "gap": round(b["bound"] - ar[d["key"]]["loss"], 4)}
    del dm
    return out


# ---------- 2. benchmarks ----------

def benchmarks(pair):
    import benchmarks as B
    import registry
    a = B.load_results(pair["ar"]) or {}
    p = os.path.join(RUNS, pair["diffusion"], "benchmarks.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    out = {}
    for t in B.TASKS:
        m = B.TASK_INFO[t]["metric"]
        av, dv = (a.get("results") or {}).get(t, {}).get(m), (d.get("results") or {}).get(t, {}).get(m)
        out[t] = {"title": B.TASK_INFO[t]["title"], "metric": m, "ar": av, "diffusion": dv,
                  "chance": B.TASK_INFO[t]["chance"], "gpt2": B.TASK_INFO[t]["gpt2"].get(m)}
    return out


# ---------- 3. writing ----------

def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def writing(pair, device, say):
    import textgen
    from .model import load
    from .sample import diffusion_generate
    judge = textgen.load_library_model(JUDGE, device).eval()
    ar = textgen.load_library_model(pair["ar"], device).eval()
    dm = load(os.path.join(RUNS, pair["diffusion"]), device)
    prompts = textgen.heldout_prompts()
    textgen.warm_up(ar, device)
    diffusion_generate(dm, max_new_tokens=8, diffusion_steps=2, prompt_ids=prompts[0]["prompt_ids"][:8], record_steps=False)

    def score(p, ids, seconds):
        rep = textgen.repetition(ids)
        loss = textgen.continuation_loss(judge, p["prompt_ids"], ids, device)[0] if len(ids) > 1 else None
        return {"judge_loss": loss, "length": len(ids), "seconds": seconds, **rep}

    settings = {"AR, sampled (top-k 50)": ("ar", 1.0), "AR, greedy": ("ar", 0.0)}
    settings.update({f"diffusion, {k} passes": ("diffusion", k) for k in DIFFUSION_PASSES})
    out = {"judge": JUDGE, "tokens": TOKENS, "seeds": list(SEEDS), "prompts": len(prompts), "settings": {}}
    ref = [score(p, p["ref_ids"][:TOKENS], None) for p in prompts]
    out["real continuation"] = _summary(ref)
    for name, (kind, arg) in settings.items():
        say(f"writing: {name}")
        rows, examples = [], []
        for p in prompts:
            for sd in (SEEDS if not (kind == "ar" and arg == 0) else SEEDS[:1]):
                if kind == "ar":
                    _sync(device)
                    t0 = time.perf_counter()
                    tr = textgen.ar_trace(ar, p["prompt_ids"], device, TOKENS, temperature=arg, top_k=50 if arg else None, seed=sd)
                    _sync(device)
                    secs = time.perf_counter() - t0
                    ids, text = [s["id"] for s in tr["steps"]], tr["text"]
                else:
                    r = diffusion_generate(dm, max_new_tokens=TOKENS, diffusion_steps=arg, seed=sd, record_steps=False,
                                           prompt_ids=p["prompt_ids"], **SAMPLER)
                    ids, text, secs = r["ids"], r["completion"], r["seconds"]
                rows.append({**score(p, ids, secs), "source": p["source"]})
                if sd == SEEDS[0] and len(examples) < len(prompts):
                    examples.append({"source": p["source"], "prompt": p["prompt"], "text": text})
        out["settings"][name] = {"passes": TOKENS if kind == "ar" else arg, **_summary(rows),
                                 "by_source": {s: _summary([r for r in rows if r["source"] == s])
                                               for s in sorted({r["source"] for r in rows})},
                                 "examples": examples}
    out["real continuation by source"] = {s: _summary([r for r, p in zip(ref, prompts) if p["source"] == s])
                                          for s in sorted({p["source"] for p in prompts})}
    del judge, ar, dm
    return out


def _summary(rows):
    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return round(sum(v) / len(v), 4) if v else None
    s = {k: mean(k) for k in ("judge_loss", "repeated_4grams", "distinct_2", "length", "seconds")}
    s["tokens_per_sec"] = round(s["length"] / s["seconds"], 1) if s.get("seconds") else None
    return s


# ---------- tables ----------

def markdown(name, r):
    pair = r["pair"]
    lines = [f"### {pair['title']}", "", f"Autoregressive `{pair['ar']}` · diffusion `{pair['diffusion']}`", ""]
    if r.get("heldout"):
        lines += ["| Held-out text (nats/token) | AR loss | Diffusion bound | Gap |", "|---|---|---|---|"]
        from infra.results import _title
        for k, v in r["heldout"].items():
            lines.append(f"| {_title(k, v['title'])} | {v['ar_loss']:.3f} | {v['diffusion_bound']:.3f} | +{v['gap']:.3f} |")
        lines.append("")
    if r.get("benchmarks"):
        lines += ["| Benchmark | AR | Diffusion | Chance | GPT-2 small |", "|---|---|---|---|---|"]
        pct = lambda v: "–" if v is None else f"{100 * v:.1f}"
        a = [v["ar"] for v in r["benchmarks"].values()]
        d = [v["diffusion"] for v in r["benchmarks"].values()]
        for v in r["benchmarks"].values():
            lines.append(f"| {v['title']} | {pct(v['ar'])} | {pct(v['diffusion'])} | {pct(v['chance'])} | {pct(v['gpt2'])} |")
        if all(x is not None for x in a + d):
            lines.append(f"| **Average** | **{pct(sum(a) / 5)}** | **{pct(sum(d) / 5)}** | | |")
        lines.append("")
    if r.get("writing"):
        w = r["writing"]
        lines += [f"| Writing {w['tokens']} tokens ({w['prompts']} prompts × {len(w['seeds'])} seeds) | Passes | GPT-2 judge loss | Repeated 4-grams | Distinct pairs | Tokens/s |",
                  "|---|---|---|---|---|---|"]
        rc = w["real continuation"]
        lines.append(f"| Real continuation | – | {rc['judge_loss']:.3f} | {100 * rc['repeated_4grams']:.1f}% | {100 * rc['distinct_2']:.1f}% | – |")
        for k, v in w["settings"].items():
            lines.append(f"| {k} | {v['passes']} | {v['judge_loss']:.3f} | {100 * v['repeated_4grams']:.1f}% | "
                         f"{100 * v['distinct_2']:.1f}% | {v['tokens_per_sec']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="F4: autoregressive vs masked diffusion, like for like.")
    ap.add_argument("pairs", nargs="*", default=list(PAIRS), help=f"which pairs ({', '.join(PAIRS)})")
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: heldout, writing")
    ap.add_argument("--tables", action="store_true", help="only print the tables of the saved results")
    a = ap.parse_args()
    if a.tables:
        for name in a.pairs:
            print(markdown(name, json.load(open(os.path.join(OUT, f"{name}.json")))))
        return
    from train_config import setup_device
    device = setup_device().device
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    os.makedirs(OUT, exist_ok=True)
    say = lambda m: print(m, flush=True)
    for name in a.pairs:
        pair = PAIRS[name]
        path = os.path.join(OUT, f"{name}.json")
        r = json.load(open(path)) if os.path.exists(path) else {}
        r.update({"pair": pair, "device": device, "measured_at": time.strftime("%Y-%m-%d %H:%M")})
        say(f"== {name}: {pair['title']}")
        r["benchmarks"] = benchmarks(pair)
        if "heldout" not in skip:
            r["heldout"] = heldout(pair, device, say)
        if "writing" not in skip:
            r["writing"] = writing(pair, device, say)
        with open(path, "w") as f:
            json.dump(r, f, indent=1)
        say(f"saved {os.path.relpath(path, ROOT)}")
        print(markdown(name, r))


if __name__ == "__main__":
    main()
