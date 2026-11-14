# Preference tuning with DPO (D2 in docs/plan.md): teach the chat model to stop looping and to finish its answers.
#
# The instruction-tuned model (sft.py) answers in the right shape, but its greedy answers repeat themselves (27.5% of
# 4-token runs seen before) and rarely end. DPO (Direct Preference Optimization, Rafailov et al. 2023) learns from
# pairs of answers to the same question, one preferred: it raises the probability of the chosen answer and lowers the
# rejected one's, both measured against a frozen copy of the starting model, so the model moves only as far as the
# preferences ask:
#
#   loss = -log sigmoid( beta * [ (log p(chosen) - log p_ref(chosen)) - (log p(rejected) - log p_ref(rejected)) ] )
#
# The pairs come for free:
#   chosen   = the reference answer from smol-smoltalk, for questions whose answer is short (it ends);
#   rejected = the chat model's own greedy answer to the same question (it loops and runs on).
# A small likelihood term on the chosen answer (as in RPO) keeps DPO from lowering both answers' probability.
#
#   python3 dpo.py pairs --model <chat model id> [--n 2000]        # generate the rejected answers
#   python3 dpo.py train --model <chat model id> [--epochs 2] [--out folder]
#
# The result is a normal chat checkpoint (model_config.json with "chat": true and "dpo": {...}).

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from inference import generate_stream, load_model, load_model_config, save_model_config
from sft import CHAT_FORMAT, DATASET, format_prompt, repeated_4grams, resolve_base
from tokenization import get_tokenizer
from train_config import setup_device

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(PROJECT_DIR, "data", "dpo-pairs.jsonl")
MAX_ANSWER = 200          # tokens: short reference answers only, and the rejected answer's length limit


def candidate_questions(n, seed=0, enc=None, max_prompt=150):
    """Single-turn smol-smoltalk conversations (no system message) whose reference answer is short."""
    from datasets import load_dataset
    enc = enc or get_tokenizer()
    ds = load_dataset(DATASET, split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        msgs = ds[i]["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        q, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        nq, na = len(enc.encode_ordinary(q)), len(enc.encode_ordinary(" " + a))
        if nq > max_prompt or not 20 <= na <= MAX_ANSWER - 1:
            continue
        out.append({"row": i, "question": q, "chosen": a})
        if len(out) >= n:
            break
    return out


def make_pairs(model_id, n=2000, path=PAIRS, say=print):
    """Greedy answers from the chat model as the rejected side. Appends to `path`, so it can be resumed."""
    dev = setup_device()
    enc = get_tokenizer()
    w, cp = resolve_base(model_id)
    config, _ = load_model_config(cp)
    model = load_model(config, w, dev.device)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["row"] for l in open(path)}
    todo = [c for c in candidate_questions(n, enc=enc) if c["row"] not in done]
    say(f"{len(done)} pairs already made, {len(todo)} to go")
    t0 = time.time()
    with open(path, "a") as f:
        for k, c in enumerate(todo):
            ids = format_prompt([{"role": "user", "content": c["question"]}], enc)
            out = [ev["id"] for ev in generate_stream(model, ids, dev.device, MAX_ANSWER, 0.0, None, stop_token=enc.eot_token)]
            ended = len(out) < MAX_ANSWER
            f.write(json.dumps({**c, "rejected_ids": out, "rejected_ended": ended,
                                "rejected_rep4": round(repeated_4grams(out), 3)}) + "\n")
            if (k + 1) % 50 == 0:
                f.flush()
                rate = (k + 1) / (time.time() - t0)
                say(f"  {len(done) + k + 1:,} pairs ({rate:.2f}/s, about {(len(todo) - k - 1) / rate / 60:.0f} min left)")
    return path


def load_pairs(path, enc):
    """(prompt ids, chosen ids, rejected ids) for each pair; chosen ends with <|endoftext|>."""
    out = []
    for line in open(path):
        p = json.loads(line)
        prompt = format_prompt([{"role": "user", "content": p["question"]}], enc)
        chosen = enc.encode_ordinary(" " + p["chosen"]) + [enc.eot_token]
        rejected = list(p["rejected_ids"]) + ([enc.eot_token] if p["rejected_ended"] else [])
        if chosen == rejected:
            continue
        out.append((prompt, chosen, rejected, p))
    return out


def batch_tensors(seqs, device, block):
    """seqs: [(prompt, answer)] -> x, y (targets -1 outside the answer), padded to the longest."""
    rows = [(pr + an)[:block + 1] for pr, an in seqs]
    starts = [len(pr) for pr, _ in seqs]
    L = max(len(r) for r in rows) - 1
    L = min(block, -(-L // 64) * 64)            # a few fixed lengths: the MPS memory cache stops growing with every new shape
    x = torch.zeros(len(rows), L, dtype=torch.long)
    y = torch.full((len(rows), L), -1, dtype=torch.long)
    for i, (r, s) in enumerate(zip(rows, starts)):
        t = torch.tensor(r)
        x[i, :len(r) - 1] = t[:-1]
        y[i, s - 1:len(r) - 1] = t[s:]          # predict the answer tokens only
    return x.to(device), y.to(device)


def seq_logps(model, x, y, ctx):
    """Sum of log probabilities of the target tokens per row (and the count). No cross-entropy and no full
    log-softmax copy: the chosen token's logit minus the logsumexp, which keeps memory to one logits tensor."""
    with ctx:
        logits, _ = model(x, all_logits=True)
    mask = y >= 0
    tok = logits.gather(2, y.clamp(min=0).unsqueeze(-1)).squeeze(-1).float() - torch.logsumexp(logits.float(), dim=-1)
    return (tok * mask).sum(1), mask.sum(1)


def train(model_id, out_dir, path=PAIRS, epochs=2, batch_pairs=4, lr=2e-6, beta=0.1, sft_weight=0.2, warmup=20,
          held_out=100, say=print):
    dev = setup_device()
    device = dev.device
    ctx = dev.ctx if device == "cuda" else nullcontext()
    enc = get_tokenizer()
    w, cp = resolve_base(model_id)
    config, info = load_model_config(cp)
    policy = load_model(config, w, device)
    policy.eval()              # no dropout: the policy and reference log probabilities must be comparable
    ref = load_model(config, w, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    pairs = load_pairs(path, enc)
    random.Random(0).shuffle(pairs)
    test, pairs = pairs[:held_out], pairs[held_out:]
    block = config.block_size

    @torch.no_grad()
    def ref_logps(items):
        out = []
        for i in range(0, len(items), 4):
            chunk = items[i:i + 4]
            xc, yc = batch_tensors([(a[0], a[1]) for a in chunk], device, block)
            xr, yr = batch_tensors([(a[0], a[2]) for a in chunk], device, block)
            out += list(zip(seq_logps(ref, xc, yc, ctx)[0].tolist(), seq_logps(ref, xr, yr, ctx)[0].tolist()))
        return out

    say(f"{len(pairs):,} training pairs, {len(test)} held out; the reference model's log probabilities first …")
    ref_train, ref_test = ref_logps(pairs), ref_logps(test)
    del ref
    if device == "mps":
        torch.mps.empty_cache()

    def dpo_terms(items, refs, train_mode):
        xc, yc = batch_tensors([(a[0], a[1]) for a in items], device, block)
        xr, yr = batch_tensors([(a[0], a[2]) for a in items], device, block)
        lc, nc = seq_logps(policy, xc, yc, ctx)
        lr_, _ = seq_logps(policy, xr, yr, ctx)
        rc = torch.tensor([r[0] for r in refs], device=device)
        rr = torch.tensor([r[1] for r in refs], device=device)
        margin = beta * ((lc - rc) - (lr_ - rr))
        loss = -F.logsigmoid(margin).mean() + sft_weight * (-(lc / nc).mean())
        return loss, margin.detach(), (lc - rc).detach(), (lr_ - rr).detach()

    @torch.no_grad()
    def evaluate():
        acc, m, n = 0, 0.0, 0
        for i in range(0, len(test), 4):
            _, margin, _, _ = dpo_terms(test[i:i + 4], ref_test[i:i + 4], False)
            acc += (margin > 0).sum().item()
            m += margin.sum().item()
            n += len(margin)
        return acc / max(1, n), m / max(1, n)

    steps = math.ceil(len(pairs) / batch_pairs) * epochs
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    # Before training the margin is exactly 0 (policy = reference); what the starting model prefers is shown by
    # its own log probabilities: how often the reference answer is more likely than its own greedy one.
    acc0 = sum(rc > rr for rc, rr in ref_test) / max(1, len(ref_test))
    say(f"{steps:,} updates of {batch_pairs} pairs, lr {lr:g}, beta {beta}; "
        f"the starting model finds the reference answer more likely than its own on {acc0:.1%} of held-out pairs")
    history, t0, order = [], time.time(), list(range(len(pairs)))
    step = 0
    for ep in range(epochs):
        random.Random(ep + 1).shuffle(order)
        for i in range(0, len(order), batch_pairs):
            idx = order[i:i + batch_pairs]
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))
            # one pair at a time with gradients accumulated: the same update, a quarter of the memory (16 GB Macs)
            parts = []
            for j in idx:
                l1, m1, c1, r1 = dpo_terms([pairs[j]], [ref_train[j]], True)
                (l1 / len(idx)).backward()
                parts.append((l1.detach(), m1, c1, r1))
            loss = torch.stack([q[0] for q in parts]).mean()
            margin, dc, dr = (torch.cat([q[k] for q in parts]) for k in (1, 2, 3))
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if device == "mps" and step % 10 == 0:
                torch.mps.empty_cache()
            if step % 25 == 0 or step == steps:
                history.append([step, round(loss.item(), 4), round((margin > 0).float().mean().item(), 3),
                                round(dc.mean().item(), 3), round(dr.mean().item(), 3)])
                say(f"update {step:,}/{steps:,}  loss {loss.item():.4f}  chosen {dc.mean().item():+.2f}  "
                    f"rejected {dr.mean().item():+.2f}  ({(time.time() - t0) / 60:.1f} min)")
        acc, m = evaluate()
        history.append([step, "eval", round(acc, 3), round(m, 3)])
        say(f"  epoch {ep + 1}: held-out preference accuracy {acc:.1%}, mean margin {m:.3f}")
    secs = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(out_dir, "best_model_params.pt"))
    datasets = list(info.get("datasets") or [])
    extra = {**{k: v for k, v in info.items() if k != "model"}, "datasets": datasets + ["DPO: smol-smoltalk answers vs own greedy"],
             "chat": True, "chat_format": CHAT_FORMAT, "base_model": info.get("base_model"), "sft_model": model_id,
             "dpo": {"pairs": len(pairs), "held_out": len(test), "epochs": epochs, "batch_pairs": batch_pairs, "lr": lr,
                     "beta": beta, "sft_weight": sft_weight, "steps": steps, "seconds": round(secs),
                     "start_prefers_chosen": round(acc0, 3), "accuracy_after": round(acc, 3), "history": history},
             "trained_at": time.strftime("%Y-%m-%d %H:%M"), "device": device,
             "platform": "Colab" if os.path.exists("/content") else "local"}
    save_model_config(os.path.join(out_dir, "model_config.json"), config, extra)
    say(f"Done in {secs / 60:.1f} min: held-out preference accuracy {acc:.1%} (the start preferred the reference on {acc0:.1%}). Saved in {out_dir}")
    return extra


def held_out_eval(models, path=PAIRS, held_out=100, out_path=None, say=print):
    """Greedy answers to the held-out pairs' questions (the same 100 that training never saw): repetition, how many
    end by themselves within MAX_ANSWER tokens, length, and the per-token loss on the reference answers."""
    dev = setup_device()
    enc = get_tokenizer()
    pairs = load_pairs(path, enc)
    random.Random(0).shuffle(pairs)
    test = pairs[:held_out]
    results = {}
    for mid in models:
        w, cp = resolve_base(mid)
        config, _ = load_model_config(cp)
        model = load_model(config, w, dev.device)
        reps, ended, lens, nll, ntok, samples = [], 0, [], 0.0, 0, []
        with torch.no_grad():
            for k, (prompt, chosen, _, p) in enumerate(test):
                out = [ev["id"] for ev in generate_stream(model, prompt, dev.device, MAX_ANSWER, 0.0, None, stop_token=enc.eot_token)]
                ended += len(out) < MAX_ANSWER
                lens.append(len(out))
                reps.append(repeated_4grams(out))
                x, y = batch_tensors([(prompt, chosen)], dev.device, config.block_size)
                lp, n = seq_logps(model, x, y, nullcontext())
                nll -= lp.item()
                ntok += n.item()
                if k < 5:
                    samples.append({"question": p["question"], "answer": enc.decode(out).strip()})
        results[mid] = {"repeated_4grams": round(sum(reps) / len(reps), 3), "ended": round(ended / len(test), 3),
                        "mean_length": round(sum(lens) / len(lens), 1), "reference_loss": round(nll / ntok, 4),
                        "samples": samples}
        r = results[mid]
        say(f"{mid}: repetition {r['repeated_4grams']:.1%}, ended {r['ended']:.0%}, length {r['mean_length']}, "
            f"loss on the reference answers {r['reference_loss']:.3f}")
        del model
    if out_path:
        json.dump({"measured_at": time.strftime("%Y-%m-%d %H:%M"), "n": len(test), "results": results},
                  open(out_path, "w"), indent=1)
    return results


def main():
    ap = argparse.ArgumentParser(description="DPO against repetition, on an instruction-tuned model.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--path", default=PAIRS)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--path", default=PAIRS)
    t.add_argument("--out", default=None)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-6)
    t.add_argument("--beta", type=float, default=0.1)
    t.add_argument("--sft-weight", type=float, default=0.2)
    t.add_argument("--held-out", type=int, default=100)
    e = sub.add_parser("eval", help="greedy answers to the held-out questions: repetition, ended, reference loss")
    e.add_argument("models", nargs="+")
    e.add_argument("--path", default=PAIRS)
    e.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "eval":
        held_out_eval(a.models, a.path, out_path=a.out)
    elif a.cmd == "pairs":
        make_pairs(a.model, a.n, a.path)
    else:
        out = a.out or os.path.join(PROJECT_DIR, "outputs", "dpo", time.strftime("%Y%m%d-%H%M"))
        train(a.model, out, a.path, a.epochs, lr=a.lr, beta=a.beta, sft_weight=a.sft_weight, held_out=a.held_out)


if __name__ == "__main__":
    main()
