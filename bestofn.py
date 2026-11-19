"""Best-of-N with a reward model: is a judge's pick of N sampled answers better than the greedy answer? (D4 pilot)

The first step before training on reward-model preferences (on-policy pairs, then SimPO or DPO): if the reward
model's favourite of N samples isn't clearly better than greedy decoding, pairs built from its choices won't teach
the model much either. Everything runs locally (PyTorch + transformers); nothing is sent anywhere.

  questions  the 100 held-out D1b questions (data/dpo-pairs-d1b.jsonl, the same 100 that `dpo.py eval` uses and DPO
             training never saw) and the 200 closed-book fact questions (evals/facts.jsonl), which have a right answer
  answers    per question: the greedy answer, and N samples at temperature 0.8 (inference.generate_batch), each up
             to 200 tokens
  judge      Skywork/Skywork-Reward-V2-Qwen3-0.6B (Apache-2.0, 1.2 GB, downloaded once into the Hugging Face cache):
             a sequence classifier that gives each (question, answer) one score, higher = better
  compared   greedy, one random sample (the first), the judge's best and worst of the N samples, and its best of N + 1 (greedy in the pool), on the
             judge's score, repeated 4-grams, answers that end by themselves, length, and for the fact questions
             whether the answer names the right answer within its first 40 tokens (facts.py's rule)

    python3 bestofn.py --model <chat model id> [--n 6] [--temperature 0.5] [--limit 10]
    python3 bestofn.py --model <chat model id> --pairs 2000 --temperature 0.5   # → data/dpo-pairs-rm.jsonl

The pilot (19 Nov, 4c0ef4): at temperature 0.8 the judge's best of 6 loses to greedy (facts 26.5% vs 30%: samples
that hot are much worse); at 0.5 its best of greedy + 6 wins (facts 37% vs 30%, judge score -2.87 vs -5.43, half the
repetition). So --pairs builds on-policy preference pairs at 0.5 for DPO or SimPO (`dpo.py train --path ...`).

The answers are saved first (outputs/bestofn/<model>-candidates.jsonl) and reused, so the scoring can be rerun
alone; the report goes to outputs/bestofn/<model>-n<N>[-t<temperature>].json (no tag at the default 0.8).
"""

import argparse
import json
import os
import random
import time

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "bestofn")
PAIRS = os.path.join(PROJECT_DIR, "data", "dpo-pairs-d1b.jsonl")
JUDGE = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
PAIRS_OUT = os.path.join(PROJECT_DIR, "data", "dpo-pairs-rm.jsonl")
MAX_ANSWER = 200
FACT_TOKENS = 40     # facts.py: a chat answer is right when the answer appears in its first 40 tokens


def questions(pairs_path=PAIRS, held_out=100):
    """The held-out chat questions (same shuffle as dpo.held_out_eval) and the fact questions."""
    from dpo import load_pairs
    from facts import load_facts
    from tokenization import get_tokenizer
    pairs = load_pairs(pairs_path, get_tokenizer())
    random.Random(0).shuffle(pairs)
    out = [{"id": f"chat-{p['row']}", "set": "chat", "question": p["question"]} for *_, p in pairs[:held_out]]
    out += [{"id": f["id"], "set": "facts", "question": f["question"], "fact": f} for f in load_facts()]
    return out


def generate(model_id, items, n, temperature, path, say=print):
    """Greedy + n samples per question, appended to `path` (resumable)."""
    from inference import generate_batch, generate_stream, load_model, load_model_config
    from sft import format_prompt, resolve_base
    from tokenization import get_tokenizer
    from train_config import setup_device
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path)}
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        return
    dev = setup_device().device
    enc = get_tokenizer()
    w, cp = resolve_base(model_id)
    cfg, _ = load_model_config(cp)
    model = load_model(cfg, w, dev).eval()
    say(f"generating: {len(done)} questions done, {len(todo)} to go (greedy + {n} samples each)")
    t0 = time.time()
    with open(path, "a") as f, torch.no_grad():
        for k, it in enumerate(todo):
            ids = format_prompt([{"role": "user", "content": it["question"]}], enc)
            greedy = [ev["id"] for ev in generate_stream(model, ids, dev, MAX_ANSWER, 0.0, None, n_alternatives=1,
                                                         stop_token=enc.eot_token)]
            samples = generate_batch(model, ids, dev, n=n, max_new_tokens=MAX_ANSWER, temperature=temperature,
                                     seed=k, stop_token=enc.eot_token)
            row = {k2: v for k2, v in it.items() if k2 != "fact"}
            row["answers"] = [{"kind": "greedy" if j == 0 else "sample", "ids": a} for j, a in enumerate([greedy] + samples)]
            f.write(json.dumps(row) + "\n")
            if (k + 1) % 20 == 0:
                f.flush()
                rate = (k + 1) / (time.time() - t0)
                say(f"  {len(done) + k + 1} questions ({rate:.2f}/s, about {(len(todo) - k - 1) / rate / 60:.0f} min left)")
    del model


def score(rows, device, batch=8, say=print):
    """Adds "reward" to every answer: the judge's score for (question, answer)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tokenization import get_tokenizer
    enc = get_tokenizer()
    tok = AutoTokenizer.from_pretrained(JUDGE)
    rm = AutoModelForSequenceClassification.from_pretrained(JUDGE, dtype=torch.float32, num_labels=1).to(device).eval()
    todo = [(r, a) for r in rows for a in r["answers"] if "reward" not in a]
    say(f"scoring {len(todo)} answers with {JUDGE}")
    t0 = time.time()
    tok.padding_side = "right"
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["question"]},
                                          {"role": "assistant", "content": enc.decode(a["ids"]).strip()}],
                                         tokenize=False) for r, a in chunk]
        x = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = rm(**x).logits[:, 0].float().tolist()
        for (_, a), v in zip(chunk, logits):
            a["reward"] = round(v, 4)
        if (s // batch + 1) % 50 == 0:
            say(f"  {s + len(chunk)}/{len(todo)} ({(s + len(chunk)) / (time.time() - t0):.1f} answers/s)")
    del rm


def report(rows, facts_by_id, n):
    """Per method (greedy, first sample, judge's best, judge's worst) and per question set."""
    from facts import is_right
    from sft import repeated_4grams
    from tokenization import get_tokenizer
    enc = get_tokenizer()
    pick = {
        "greedy": lambda ans: ans[0],
        "sample": lambda ans: ans[1],
        f"best_of_{n}": lambda ans: max(ans[1:], key=lambda a: a["reward"]),
        f"worst_of_{n}": lambda ans: min(ans[1:], key=lambda a: a["reward"]),
        f"best_of_{n + 1}": lambda ans: max(ans, key=lambda a: a["reward"]),   # greedy in the pool too
    }
    out = {}
    for qset in ("chat", "facts", "all"):
        sel = [r for r in rows if qset == "all" or r["set"] == qset]
        if not sel:
            continue
        out[qset] = {}
        for name, fn in pick.items():
            chosen = [(r, fn(r["answers"])) for r in sel]
            m = {"reward": sum(a["reward"] for _, a in chosen) / len(chosen),
                 "repeated_4grams": sum(repeated_4grams(a["ids"]) for _, a in chosen) / len(chosen),
                 "ended": sum(len(a["ids"]) < MAX_ANSWER for _, a in chosen) / len(chosen),
                 "mean_length": sum(len(a["ids"]) for _, a in chosen) / len(chosen)}
            fr = [(r, a) for r, a in chosen if r["set"] == "facts"]
            if fr:
                m["facts_right"] = sum(is_right(enc.decode(a["ids"][:FACT_TOKENS]), facts_by_id[r["id"]])
                                       for r, a in fr) / len(fr)
            out[qset][name] = {k: round(v, 4) for k, v in m.items()}
        # how often the judge's best beats greedy on its own score, and an upper bound: any sample right?
        out[qset]["best_beats_greedy"] = round(sum(max(a["reward"] for a in r["answers"][1:]) > r["answers"][0]["reward"]
                                                   for r in sel) / len(sel), 4)
        fs = [r for r in sel if r["set"] == "facts"]
        if fs:
            out[qset]["any_sample_right"] = round(sum(any(is_right(enc.decode(a["ids"][:FACT_TOKENS]), facts_by_id[r["id"]])
                                                          for a in r["answers"][1:]) for r in fs) / len(fs), 4)
    return out


def pair_questions(n, exclude=(PAIRS,)):
    """n single-turn smol-smoltalk questions (dpo.candidate_questions' filter) that no earlier pair file used, so
    the held-out D1b questions stay unseen; the pool is drawn with a different seed than the D1b pairs."""
    from dpo import candidate_questions
    used = set()
    for path in exclude:
        if os.path.exists(path):
            used |= {json.loads(l)["row"] for l in open(path)}
    pool = candidate_questions(n + len(used), seed=1)
    out = [{"id": f"row-{c['row']}", "set": "train", "question": c["question"], "row": c["row"]}
           for c in pool if c["row"] not in used]
    return out[:n]


def make_pairs(model_id, n, n_samples, temperature, out_path, say=print):
    """Generate greedy + n_samples answers for n new questions, score them all, and write one pair per question:
    chosen = the highest-scoring answer, rejected = the lowest (both the model's own tokens), with every score."""
    from tokenization import get_tokenizer
    from train_config import setup_device
    items = pair_questions(n)
    cand = os.path.join(OUT_DIR, f"pairs-{model_id}-n{n_samples}-t{temperature:g}-candidates.jsonl")
    generate(model_id, items, n_samples, temperature, cand, say=say)
    rows = [json.loads(l) for l in open(cand)]
    if any("reward" not in a for r in rows for a in r["answers"]):
        score(rows, setup_device().device, say=say)
        with open(cand, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    enc = get_tokenizer()
    kept = 0
    with open(out_path, "w") as f:
        for r in rows:
            best = max(r["answers"], key=lambda a: a["reward"])
            worst = min(r["answers"], key=lambda a: a["reward"])
            if best["ids"] == worst["ids"]:
                continue
            f.write(json.dumps({"row": r["row"], "question": r["question"], "chosen": enc.decode(best["ids"]).strip(),
                                "chosen_ids": best["ids"], "chosen_ended": len(best["ids"]) < MAX_ANSWER,
                                "chosen_kind": best["kind"], "chosen_reward": best["reward"],
                                "rejected_ids": worst["ids"], "rejected_ended": len(worst["ids"]) < MAX_ANSWER,
                                "rejected_reward": worst["reward"],
                                "rewards": [a["reward"] for a in r["answers"]], "judge": JUDGE,
                                "temperature": temperature}) + "\n")
            kept += 1
    margins = sorted(json.loads(l)["chosen_reward"] - json.loads(l)["rejected_reward"] for l in open(out_path))
    say(f"{kept} pairs in {out_path}; reward margin median {margins[len(margins) // 2]:.2f}, "
        f"greedy chosen in {sum(json.loads(l)['chosen_kind'] == 'greedy' for l in open(out_path)) / kept:.0%}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Best-of-N with a reward model vs greedy (D4 pilot).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int, help="only the first N chat and N fact questions (a quick test)")
    ap.add_argument("--no-score", action="store_true", help="generate the answers only")
    ap.add_argument("--pairs", type=int, metavar="N",
                    help="instead of the pilot: preference pairs from N new smol-smoltalk questions (chosen = the "
                         "judge's best of greedy + n samples, rejected = its worst) into --out")
    ap.add_argument("--out", default=PAIRS_OUT)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.pairs:
        return make_pairs(a.model, a.pairs, a.n, a.temperature, a.out)
    items = questions()
    if a.limit:
        items = [it for it in items if it["set"] == "chat"][:a.limit] + [it for it in items if it["set"] == "facts"][:a.limit]
    tag = (f"{a.model}-n{a.n}" + (f"-t{a.temperature:g}" if a.temperature != 0.8 else "")
           + (f"-limit{a.limit}" if a.limit else ""))
    cand = os.path.join(OUT_DIR, f"{tag}-candidates.jsonl")
    generate(a.model, items, a.n, a.temperature, cand)
    if a.no_score:
        print(f"answers in {cand}")
        return
    from train_config import setup_device
    rows = [json.loads(l) for l in open(cand)]
    score(rows, setup_device().device)
    with open(cand, "w") as f:                      # keep the scores with the answers
        for r in rows:
            f.write(json.dumps(r) + "\n")
    facts_by_id = {it["id"]: it["fact"] for it in items if it["set"] == "facts"}
    res = report(rows, facts_by_id, a.n)
    path = os.path.join(OUT_DIR, f"{tag}.json")
    json.dump({"model": a.model, "judge": JUDGE, "n": a.n, "temperature": a.temperature, "questions": len(rows),
               "measured_at": time.strftime("%Y-%m-%d %H:%M"), "results": res}, open(path, "w"), indent=1)
    for qset, r in res.items():
        print(f"\n{qset} ({sum(1 for x in rows if qset == 'all' or x['set'] == qset)} questions)")
        print(f"  {'':12} {'reward':>7} {'rep4':>6} {'ended':>6} {'length':>7} {'facts right':>11}")
        for name, m in r.items():
            if isinstance(m, dict):
                fr = f"{m['facts_right']:11.1%}" if "facts_right" in m else ""
                print(f"  {name:12} {m['reward']:7.2f} {m['repeated_4grams']:6.1%} {m['ended']:6.0%} {m['mean_length']:7.1f} {fr}")
        print(f"  judge's best beats greedy on its score: {r['best_beats_greedy']:.0%}"
              + (f"; some sample right: {r['any_sample_right']:.0%}" if "any_sample_right" in r else ""))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
