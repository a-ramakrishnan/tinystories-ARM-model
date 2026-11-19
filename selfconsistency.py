"""Self-consistency on closed-book facts: ask the chat model several times and choose among its answers.

Greedy decoding answers 31% of the 200 fact questions (evals/facts.jsonl) right, but in the best-of-N pilot some
answer among seven was right for 43%: the knowledge is often there, and greedy doesn't always find it. Wrong
answers tend to scatter while the right one repeats, so a vote can find it without any training.

For each question: the greedy answer and N - 1 samples at a temperature (bestofn.generate). Each answer's **key**
is the thing it names: the words after the question's stem ("The capital of India is | New Delhi, ...") up to the
first comma or full stop, without "the"/"a"/"an", first four words; when the answer doesn't repeat the stem, its
first sentence without the question's own words. Choosers, over the first k answers (greedy first) for k = 1, 3, 5, …:

  greedy       the greedy answer (k = 1)
  vote         the most common key; a tie goes to the key that appeared first (greedy's, if it's in the tie)
  confidence   the answer the model itself finds most likely: highest mean log probability per token
  vote+conf    the most common key, ties broken by the summed confidence of the answers behind it
  judge        the reward model's favourite (Skywork-Reward-V2-Qwen3-0.6B, as in bestofn.py)
  oracle       right if any of the k answers is right: the ceiling no chooser can pass

Right means facts.py's rule: the answer (or an alias) appears in the answer's first 40 tokens. The choosers never
see the right answer or the wrong options.

    python3 selfconsistency.py --model <chat model id> [--n 7] [--temperature 0.5] [--no-judge] [--limit 20]

Answers go to outputs/selfconsistency/<model>-n<N>-t<T>-candidates.jsonl (reused on a rerun), the report to the
same name with .json.
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from contextlib import nullcontext

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "selfconsistency")
FACT_TOKENS = 40
ARTICLES = {"the", "a", "an"}


def answer_key(text, item):
    """What the answer names, normalized: see the module docstring."""
    t = " ".join(text.split())
    low = t.lower()
    stem = item["stem"].lower()
    i = low.find(stem)
    if i >= 0:
        rest = t[i + len(stem):]
        drop = set()
    else:
        rest = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
        drop = set(re.findall(r"[a-z0-9']+", (item["question"] + " " + item["stem"]).lower()))
    rest = re.split(r"[,.;:!?\n(]", rest, maxsplit=1)[0]
    words = [w[:-2] if w.endswith("'s") else w for w in re.findall(r"[a-z0-9'\-]+", rest.lower())]
    words = [w for w in words if w not in ARTICLES and w not in drop]
    return " ".join(words[:4])


@torch.no_grad()
def mean_logprobs(model, enc, rows, device, batch=8):
    """Adds "lp": the model's mean log probability per token of each answer (its own confidence)."""
    from dpo import batch_tensors, seq_logps
    from sft import format_prompt
    todo = [(r, a) for r in rows for a in r["answers"] if "lp" not in a and a["ids"]]
    for r in rows:
        for a in r["answers"]:
            if not a["ids"]:
                a["lp"] = float("-inf")
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        seqs = [(format_prompt([{"role": "user", "content": r["question"]}], enc), a["ids"]) for r, a in chunk]
        x, y = batch_tensors(seqs, device, model.config.block_size)
        lp, n = seq_logps(model, x, y, nullcontext())
        for (_, a), v, k in zip(chunk, lp.tolist(), n.tolist()):
            a["lp"] = round(v / max(1, k), 4)


def choose(answers, keys, how):
    """Index of the chosen answer among `answers` (greedy first)."""
    if how == "greedy":
        return 0
    if how == "confidence":
        return max(range(len(answers)), key=lambda i: answers[i]["lp"])
    if how == "judge":
        return max(range(len(answers)), key=lambda i: answers[i]["reward"])
    counts = Counter(keys)
    top = max(counts.values())
    tied = [k for k in dict.fromkeys(keys) if counts[k] == top]      # in order of first appearance
    if how == "vote":
        best = tied[0]
    else:  # vote+conf
        conf = defaultdict(float)
        for k, a in zip(keys, answers):
            conf[k] += a["lp"]
        best = max(tied, key=lambda k: conf[k])
    return keys.index(best)


def report(rows, facts_by_id, enc, ks, judged):
    from facts import is_right
    hows = ["greedy", "vote", "confidence", "vote+conf"] + (["judge"] if judged else [])
    out = {}
    for k in ks:
        res = {h: 0 for h in hows + ["oracle"]}
        for r in rows:
            item = facts_by_id[r["id"]]
            ans = r["answers"][:k]
            texts = [enc.decode(a["ids"][:FACT_TOKENS]) for a in ans]
            keys = [answer_key(enc.decode(a["ids"]), item) for a in ans]
            right = [is_right(t, item) for t in texts]
            for h in hows:
                res[h] += right[choose(ans, keys, h)]
            res["oracle"] += any(right)
        out[k] = {h: round(v / len(rows), 4) for h, v in res.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description="Self-consistency on the closed-book facts: vote among several answers.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=7, help="answers per question, greedy included")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--no-judge", action="store_true", help="skip the reward model")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    import bestofn
    from facts import load_facts
    from inference import load_model, load_model_config
    from sft import resolve_base
    from tokenization import get_tokenizer
    from train_config import setup_device
    os.makedirs(OUT_DIR, exist_ok=True)
    facts = load_facts()[:a.limit] if a.limit else load_facts()
    items = [{"id": f["id"], "set": "facts", "question": f["question"]} for f in facts]
    facts_by_id = {f["id"]: f for f in facts}
    tag = f"{a.model}-n{a.n}-t{a.temperature:g}" + (f"-limit{a.limit}" if a.limit else "")
    cand = os.path.join(OUT_DIR, f"{tag}-candidates.jsonl")
    t0 = time.time()
    bestofn.generate(a.model, items, a.n - 1, a.temperature, cand)
    gen_s = time.time() - t0
    rows = [json.loads(l) for l in open(cand)]
    device = setup_device().device
    enc = get_tokenizer()
    if any("lp" not in x for r in rows for x in r["answers"]):
        w, cp = resolve_base(a.model)
        cfg, _ = load_model_config(cp)
        model = load_model(cfg, w, device).eval()
        mean_logprobs(model, enc, rows, device)
        del model
    if not a.no_judge and any("reward" not in x for r in rows for x in r["answers"]):
        bestofn.score(rows, device)
    with open(cand, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ks = [k for k in (1, 3, 5, 7, 9, 11, 15) if k <= a.n]
    res = report(rows, facts_by_id, enc, ks, judged=not a.no_judge)
    path = os.path.join(OUT_DIR, f"{tag}.json")
    json.dump({"model": a.model, "n": a.n, "temperature": a.temperature, "questions": len(rows),
               "generation_seconds": round(gen_s, 1), "measured_at": time.strftime("%Y-%m-%d %H:%M"),
               "right_by_k": res}, open(path, "w"), indent=1)
    hows = list(next(iter(res.values())))
    print(f"\nfacts answered right, {len(rows)} questions (answers: greedy + samples at {a.temperature})")
    print(f"  {'k':>3} " + " ".join(f"{h:>11}" for h in hows))
    for k, r in res.items():
        print(f"  {k:>3} " + " ".join(f"{r[h]:11.1%}" for h in hows))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
