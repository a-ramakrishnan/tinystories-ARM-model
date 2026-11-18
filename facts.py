"""Closed-book facts: does the model still know what it read? (evals/facts.jsonl, 200 questions)

Instruction tuning can overwrite knowledge the base model had: on 18 Nov the chat model said the capital of India
was "Vrindindjana", while the base model it was tuned from ranks " Delhi" among its top three next words. This
measures that, the same way for every model:

  choice     each question has its answer and three wrong ones of the same kind ("New Delhi" vs Mumbai, Kolkata,
             Bangalore), each scored by its log probability after the stem, as the benchmarks do:
               acc         the right answer has the highest total log probability
               acc_norm    the same, divided by the answer's length in characters (the harness's length-normalized
                           accuracy; "New Delhi" isn't penalized for being two tokens)
               right_lp    the right answer's mean log probability: a smooth number that moves before accuracy does
  generated  the model answers on its own, greedily: right when the answer (or an accepted alias) appears in the
             first 40 tokens (chat) or the first 12 (base, continuing the stem)

A chat model reads the question in its chat format and the stem as the start of its answer:
  <|endoftext|>User: What is the capital of India?\\n\\nAssistant: The capital of India is | New Delhi
A base model reads only the stem: "The capital of India is | New Delhi".

    python3 facts.py <model id> [<model id> ...]       # saves models/<id>/facts.json and prints a table
    python3 facts.py <ids> --as base                    # score chat models as plain text too
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch

import registry

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(PROJECT_DIR, "evals", "facts.jsonl")
VERSION = 1


def load_facts(path=FACTS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def context(item, enc, chat):
    if not chat:
        return item["stem"]
    from sft import format_prompt
    return format_prompt([{"role": "user", "content": item["question"]}], enc) + enc.encode_ordinary(" " + item["stem"])


def is_right(text, item):
    t = " " + text.lower().replace("\n", " ")
    return any(a.lower() in t for a in [item["answer"]] + item["aliases"])


def generate(model, enc, item, device, chat, n):
    """Greedy, no repetition penalty: the chat model answers the question; the base model continues the stem."""
    from inference import generate_stream
    from sft import format_prompt
    ids = (format_prompt([{"role": "user", "content": item["question"]}], enc) if chat
           else enc.encode_ordinary(item["stem"]))
    out = [ev["id"] for ev in generate_stream(model, ids, device, n, 0.0, None, n_alternatives=1,
                                               stop_token=enc.eot_token)]
    return enc.decode(out).strip()


@torch.no_grad()
def evaluate(model, enc, device, chat, items=None, batch=16, say=print, generated=True):
    """generated=False: only the choice scores (quick: what sft.py checks during training)."""
    from benchmarks import score_requests
    items = items or load_facts()
    reqs, owner = [], []
    for q, it in enumerate(items):
        ctx = context(it, enc, chat)
        for c, ans in enumerate([it["answer"]] + it["wrong"]):
            reqs.append((ctx, " " + ans))
            owner.append((q, c))
    t0 = time.time()
    res = score_requests(model, enc, reqs, device, batch=batch)
    lps = [[None] * 4 for _ in items]
    for (q, c), r in zip(owner, res):
        lps[q][c] = r[0]
    per_item, topics = [], defaultdict(lambda: [0, 0])
    for it, lp in zip(items, lps):
        choices = [it["answer"]] + it["wrong"]
        norm = [l / len(ch) for l, ch in zip(lp, choices)]
        gen = generate(model, enc, it, device, chat, 40 if chat else 12) if generated else ""
        per_item.append({"id": it["id"], "topic": it["topic"], "right_lp": round(lp[0], 3),
                         "best_wrong_lp": round(max(lp[1:]), 3), "acc": lp[0] > max(lp[1:]),
                         "acc_norm": norm[0] > max(norm[1:]), "generated": is_right(gen, it), "text": gen})
        topics[it["topic"]][0] += per_item[-1]["acc_norm"]
        topics[it["topic"]][1] += 1
    n = len(items)
    summary = {"acc": sum(p["acc"] for p in per_item) / n, "acc_norm": sum(p["acc_norm"] for p in per_item) / n,
               "right_lp": sum(p["right_lp"] for p in per_item) / n,
               "generated": sum(p["generated"] for p in per_item) / n if generated else None, "n": n,
               "by_topic": {t: round(r / k, 3) for t, (r, k) in sorted(topics.items())},
               "seconds": round(time.time() - t0, 1)}
    return summary, per_item


def run(model_id, as_base=False, device=None, save=True, say=print):
    import tiktoken
    from inference import load_model, load_model_config
    from train_config import setup_device
    device = device or setup_device().device
    w, c = registry.paths_of(model_id)
    cfg, info = load_model_config(c)
    chat = bool(info.get("chat")) and not as_base
    model = load_model(cfg, w, device).eval()
    enc = tiktoken.get_encoding("gpt2")
    summary, items = evaluate(model, enc, device, chat, say=say)
    out = {"summary": summary, "format": "chat" if chat else "plain", "facts_version": VERSION,
           "measured_at": time.strftime("%Y-%m-%d %H:%M"), "items": items}
    if save:
        path = os.path.join(registry.LIB_ROOT, model_id, "facts.json" if not (as_base and info.get("chat"))
                            else "facts_plain.json")
        json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
        from infra.results import record_file
        record_file(path, model_id, say=say)
    return out


def main():
    ap = argparse.ArgumentParser(description="Closed-book facts (evals/facts.jsonl): choice and generated answers.")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--as", dest="as_", choices=["auto", "base"], default="auto",
                    help="base: score chat models on the plain stem too (saved as facts_plain.json)")
    a = ap.parse_args()
    rows = []
    for mid in a.ids:
        print(f"{mid} …", flush=True)
        r = run(mid, as_base=a.as_ == "base")
        rows.append((mid, r))
    print(f"\n{'model':44} {'format':6} {'acc':>6} {'norm':>6} {'right lp':>9} {'generated':>9}")
    for mid, r in rows:
        s = r["summary"]
        print(f"{mid:44} {r['format']:6} {s['acc']:6.1%} {s['acc_norm']:6.1%} {s['right_lp']:9.2f} {s['generated']:9.1%}")
        print(f"{'':44} by topic (acc_norm): " + ", ".join(f"{t} {v:.0%}" for t, v in s["by_topic"].items()))


if __name__ == "__main__":
    main()
