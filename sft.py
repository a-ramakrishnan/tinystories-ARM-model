# Instruction tuning (supervised fine-tuning, SFT): teach a pre-trained model to answer instead of continuing text.
#
# The base model has only read documents, so "What is the water cycle?" makes it write more questions, or an
# article. Here it reads conversations (HuggingFaceTB/smol-smoltalk: 460K, written for small models) in a plain
# text format GPT-2's tokenizer handles without new tokens:
#
#   <|endoftext|>System: Be brief.                 (only if the conversation has one)
#
#   User: What is the water cycle?
#
#   Assistant: The water cycle is ...<|endoftext|>User: ...
#
# Every assistant turn ends with <|endoftext|>, so generation stops there. The loss counts only the assistant's
# tokens (and that <|endoftext|>): the model learns to answer, not to write questions. Conversations are packed
# whole into 512-token windows (never split), the rest of a window masked out.
#
# Forgetting (18 Nov): the first stage 2 run (lr 2e-4, conversations only) lost facts the base model knew; on
# facts.py's 200 closed-book questions the right answer became about 100 times less likely (mean log prob -5.66 ->
# -10.6) and choice accuracy fell from 57.5% to 43%. Two remedies, both optional:
#   --lr 3e-5 or so: a gentler step, so the format is learned with smaller changes to the weights
#   --replay 0.2: a fifth of each batch is plain pre-training text (the base model's own mix, from prepare-replay),
#                 loss on every token, so what it read keeps being rehearsed while it learns to answer
# Every evaluation also scores the facts (choice only), so forgetting shows up as training goes, not afterwards.
#
#   python3 sft.py prepare [--limit N]                    # tokenize to data/sft-smoltalk/
#   python3 sft.py prepare-replay [--tokens 32000000]     # pre-training text for --replay, to data/sft-replay/
#   python3 sft.py train --base <model id or weights.pt> [--steps N] [--out folder] [--lr 3e-5] [--replay 0.2]
#   python3 sft.py chat <weights folder> "What is the water cycle?"
#
# The result is a normal checkpoint (best_model_params.pt + model_config.json, with "chat": true), so the app's
# model library, Generate and benchmarks all work with it; Generate shows a Chat mode for it.

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from inference import generate_stream, load_model, load_model_config, save_model_config
from tokenization import get_tokenizer
from train_config import setup_device

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = "HuggingFaceTB/smol-smoltalk"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "sft-smoltalk")
REPLAY_DIR = os.path.join(PROJECT_DIR, "data", "sft-replay")
# the stage 2 base model's mix (stage2.py), by share of the replay tokens
REPLAY_SOURCES = [("BaoMinhTran/mini-fineweb-edu", "full", 0.6), ("HuggingFaceTB/cosmopedia", "stanford", 0.1),
                  ("HuggingFaceTB/cosmopedia", "openstax", 0.1), ("HuggingFaceTB/cosmopedia", "wikihow", 0.1),
                  ("HuggingFaceTB/cosmopedia", "stories", 0.1)]
CHAT_FORMAT = "smoltalk-plain-v1"
ROLE_PREFIX = {"system": "System: ", "user": "User: ", "assistant": "Assistant: "}
EVAL_QUESTIONS = ["What is the water cycle?", "Why is the sky blue?", "How do vaccines work?",
                  "Give me three tips for staying focused while studying.", "Explain photosynthesis to a ten-year-old.",
                  "What causes the seasons on Earth?", "Write a short poem about the sea.",
                  "What is the difference between weather and climate?", "How can I save money on groceries?",
                  "Summarize the causes of World War I in a few sentences.", "What is a prime number? Give examples.",
                  "Suggest a name for a bakery and explain why."]
QUESTIONS = ["What is the water cycle?", "Give me three tips for staying focused while studying.",
             "Explain photosynthesis to a ten-year-old.", "Write a short poem about the sea.",
             "What is the capital of France, and what is it known for?"]


# ---------------------------------------------------------------- the chat format
def format_prompt(messages, enc):
    """Token ids for a conversation so far, ending with "Assistant:" so the model writes the next answer."""
    ids, _ = encode_conversation(messages + [{"role": "assistant", "content": None}], enc)
    return ids


def encode_conversation(messages, enc):
    """(ids, mask): mask is 1 on the tokens the model should learn to write (assistant answers + their end)."""
    eot = enc.eot_token
    ids, mask = [eot], [0]
    for k, m in enumerate(messages):
        role = m["role"] if m["role"] in ROLE_PREFIX else "user"
        sep = "" if k == 0 or messages[k - 1]["role"] == "assistant" else "\n\n"   # after <|endoftext|>: nothing
        head = enc.encode_ordinary(sep + ROLE_PREFIX[role].rstrip())       # "Assistant:"; the space starts the body
        ids += head
        mask += [0] * len(head)
        if m["content"] is None:            # an open assistant turn: the prompt ends here
            break
        body = enc.encode_ordinary(" " + m["content"].strip())
        if role == "assistant":
            ids += body + [eot]
            mask += [1] * (len(body) + 1)
        else:
            ids += body
            mask += [0] * len(body)
    return ids, mask


def chat_prompt_text(question):
    """The same format as text, for showing it."""
    return f"User: {question}\n\nAssistant:"


# ---------------------------------------------------------------- data
def prepare(limit=None, block=512, out_dir=DATA_DIR, say=print):
    """Tokenize the conversations and pack them whole into windows of block + 1 tokens."""
    from datasets import load_dataset
    enc = get_tokenizer()
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for split in ("train", "test"):
        ds = load_dataset(DATASET, split=split)
        if limit:
            ds = ds.select(range(min(limit if split == "train" else max(500, limit // 20), len(ds))))
        windows, masks = [], []
        cur_i, cur_m = [], []
        kept = dropped = truncated = 0
        n_tokens = n_answer = 0
        for row in ds:
            msgs = row["messages"]
            ids, mask = encode_conversation(msgs, enc)
            if len(ids) > block + 1:
                # keep the turns that fit: cut after the last complete assistant answer within the window
                ends = [i + 1 for i in range(min(len(ids), block + 1)) if mask[i] and ids[i] == enc.eot_token]
                if not ends:
                    dropped += 1
                    continue
                ids, mask = ids[:ends[-1]], mask[:ends[-1]]
                truncated += 1
            kept += 1
            n_tokens += len(ids)
            n_answer += sum(mask)
            if len(cur_i) + len(ids) > block + 1:
                windows.append(np.array(cur_i + [enc.eot_token] * (block + 1 - len(cur_i)), dtype=np.uint16))
                masks.append(np.array(cur_m + [0] * (block + 1 - len(cur_m)), dtype=np.uint8))
                cur_i, cur_m = [], []
            cur_i += ids
            cur_m += mask
        if cur_i:
            windows.append(np.array(cur_i + [enc.eot_token] * (block + 1 - len(cur_i)), dtype=np.uint16))
            masks.append(np.array(cur_m + [0] * (block + 1 - len(cur_m)), dtype=np.uint8))
        np.save(os.path.join(out_dir, f"{split}_ids.npy"), np.stack(windows))
        np.save(os.path.join(out_dir, f"{split}_mask.npy"), np.stack(masks))
        stats[split] = {"conversations": kept, "dropped_too_long": dropped, "truncated": truncated,
                        "windows": len(windows), "tokens": n_tokens, "answer_tokens": n_answer,
                        "fill": round(n_tokens / max(1, len(windows) * (block + 1)), 3)}
        say(f"{split}: {kept:,} conversations ({truncated:,} cut to the turns that fit, {dropped:,} too long) -> "
            f"{len(windows):,} windows of {block + 1}, {n_tokens / 1e6:.1f}M tokens, {n_answer / 1e6:.1f}M to learn")
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"dataset": DATASET, "format": CHAT_FORMAT, "block": block, "limit": limit, **stats}, f, indent=1)
    return stats


def prepare_replay(tokens=32_000_000, out_dir=REPLAY_DIR, sources=REPLAY_SOURCES, say=print):
    """Stream documents from the base model's own datasets into one token file (documents end with <|endoftext|>),
    for --replay. 32M tokens covers 20% of one epoch's batches without repeating a window."""
    from datasets import load_dataset
    enc = get_tokenizer()
    os.makedirs(out_dir, exist_ok=True)
    out, counts = [], {}
    for name, config, share in sources:
        want, got = int(tokens * share), 0
        for row in load_dataset(name, config, split="train", streaming=True):
            ids = enc.encode_ordinary(row["text"]) + [enc.eot_token]
            out.append(np.array(ids, dtype=np.uint16))
            got += len(ids)
            if got >= want:
                break
        counts[f"{name}@{config}"] = got
        say(f"{name} {config}: {got / 1e6:.1f}M tokens")
    arr = np.concatenate(out)
    arr.tofile(os.path.join(out_dir, "replay.bin"))
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"tokens": int(len(arr)), "sources": counts}, f, indent=1)
    say(f"replay: {len(arr) / 1e6:.1f}M tokens in {out_dir}")
    return len(arr)


class SFTLoader:
    def __init__(self, data_dir, device, batch_size, seed=0, replay=0.0, replay_dir=REPLAY_DIR):
        self.device, self.batch_size = device, batch_size
        self.n_replay = int(round(batch_size * replay))         # rows of each batch that are pre-training text
        self.replay = None
        if self.n_replay:
            path = os.path.join(replay_dir, "replay.bin")
            if not os.path.exists(path):
                raise SystemExit(f"--replay needs {path}: run `python3 sft.py prepare-replay` first")
            self.replay = np.memmap(path, dtype=np.uint16, mode="r")
        self.ids = {s: np.load(os.path.join(data_dir, f"{s}_ids.npy"), mmap_mode="r") for s in ("train", "test")}
        self.mask = {s: np.load(os.path.join(data_dir, f"{s}_mask.npy"), mmap_mode="r") for s in ("train", "test")}
        self.rng = np.random.default_rng(seed)
        self.order, self.pos, self.epoch = self.rng.permutation(len(self.ids["train"])), 0, 0

    def _to(self, ids, mask):
        x = torch.from_numpy(ids[:, :-1].astype(np.int64))
        y = torch.from_numpy(ids[:, 1:].astype(np.int64))
        y[torch.from_numpy(mask[:, 1:] == 0)] = -1           # learn only the answers
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def batch(self):
        if self.pos + self.batch_size > len(self.order):
            self.order, self.pos, self.epoch = self.rng.permutation(len(self.order)), 0, self.epoch + 1
        k = self.batch_size - self.n_replay
        idx = np.sort(self.order[self.pos:self.pos + k])
        self.pos += k
        ids, mask = self.ids["train"][idx], self.mask["train"][idx]
        if self.n_replay:       # plain text windows, every token learned (as in pre-training)
            L = ids.shape[1]
            starts = self.rng.integers(0, len(self.replay) - L, self.n_replay)
            ids = np.concatenate([ids, np.stack([self.replay[s:s + L] for s in starts])])
            mask = np.concatenate([mask, np.ones((self.n_replay, L), dtype=mask.dtype)])
        return self._to(ids, mask)

    def test_batches(self, n):
        k = min(len(self.ids["test"]), n * self.batch_size)
        return [self._to(self.ids["test"][i:i + self.batch_size], self.mask["test"][i:i + self.batch_size])
                for i in range(0, k, self.batch_size)]


# ---------------------------------------------------------------- training
def resolve_base(base):
    """A library model id, a folder with best_model_params.pt + model_config.json, or the weights file itself."""
    if os.path.isdir(base):
        return os.path.join(base, "best_model_params.pt"), os.path.join(base, "model_config.json")
    if base.endswith(".pt"):
        return base, os.path.join(os.path.dirname(base), "model_config.json")
    import registry
    return registry.paths_of(base)


@torch.no_grad()
def test_loss(model, batches, ctx):
    model.eval()
    tot = n = 0.0
    for x, y in batches:
        with ctx:
            _, loss = model(x, y)
        k = (y >= 0).sum().item()
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(1, n)


def answer(model, enc, question, device, max_new_tokens=160, temperature=0.0, top_k=None):
    ids = format_prompt([{"role": "user", "content": question}], enc)
    out = []
    sampling = {"dry_multiplier": 0.8} if temperature <= 0 else None
    for ev in generate_stream(model, ids, device, max_new_tokens, temperature, top_k, stop_token=enc.eot_token,
                              sampling=sampling):
        out.append(ev["id"])
    return enc.decode([i for i in out if i != enc.eot_token]).strip()


def facts_check(model, enc, device, ctx):
    """facts.py's choice scores in the chat format (no generation): about 800 short sequences."""
    from facts import evaluate as facts_eval
    model.eval()
    with ctx:
        s, _ = facts_eval(model, enc, device, chat=True, generated=False)
    model.train()
    return round(s["acc_norm"], 4), round(s["right_lp"], 3)


def train(base, out_dir, steps=None, epochs=1.0, batch_size=32, lr=2e-4, min_lr=2e-5, warmup=200, eval_every=500,
          data_dir=DATA_DIR, compile_model=None, say=print, base_id=None, replay=0.0, replay_dir=REPLAY_DIR,
          check_facts=True):
    dev = setup_device()
    device = dev.device
    ctx = dev.ctx if device == "cuda" else nullcontext()
    weights, cfg_path = resolve_base(base)
    config, info = load_model_config(cfg_path)
    model = load_model(config, weights, device)
    model.train()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    loader = SFTLoader(data_dir, device, batch_size, replay=replay, replay_dir=replay_dir)
    n_windows = len(loader.ids["train"])
    steps = steps or max(1, int(epochs * n_windows / (batch_size - loader.n_replay)))
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), **({"fused": True} if device == "cuda" else {}))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dev.dtype == "float16")
    run = torch.compile(model) if (compile_model if compile_model is not None else device == "cuda") else model
    tests = loader.test_batches(40)
    enc = get_tokenizer()
    os.makedirs(out_dir, exist_ok=True)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = min(1.0, (s - warmup) / max(1, steps - warmup))
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))

    base_loss = test_loss(model, tests, ctx)
    say(f"Base model on held-out conversations (answers only): {base_loss:.4f}")
    conv = batch_size - loader.n_replay
    say(f"{steps:,} updates x {batch_size} x {config.block_size} tokens ({steps * conv / n_windows:.2f} epochs of {n_windows:,} windows), lr {lr:g}"
        + (f", {loader.n_replay} of each {batch_size} rows pre-training text (replay)" if loader.n_replay else ""))
    history = {"train": [], "test": [[0, round(base_loss, 4)]], "facts": []}
    if check_facts:
        fa = facts_check(model, enc, device, ctx)
        history["facts"].append([0, *fa])
        say(f"Facts (choice, chat format): {fa[0]:.1%} right, right answer's log prob {fa[1]:.2f}")
    best, t0, t_log, tok = float("inf"), time.time(), time.time(), 0
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = loader.batch()
        with ctx:
            _, loss = run(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        tok += x.numel()
        u = s + 1
        if u % 50 == 0:
            tps = tok / (time.time() - t_log)
            history["train"].append([u, round(loss.item(), 4), round(tps)])
            t_log, tok = time.time(), 0
            if u % 250 == 0:
                say(f"update {u:,}/{steps:,}  loss {loss.item():.4f}  lr {lr_at(s):.2e}  {tps:,.0f} tokens/s")
        if u % eval_every == 0 or u == steps:
            tl = test_loss(model, tests, ctx)
            history["test"].append([u, round(tl, 4)])
            if check_facts:
                fa = facts_check(model, enc, device, ctx)
                history["facts"].append([u, *fa])
                say(f"  facts @ {u:,}: {fa[0]:.1%} right, log prob {fa[1]:.2f}")
            mark = ""
            if tl < best:
                best, mark = tl, "  (best, saved)"
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model_params.pt"))
            say(f"  test @ {u:,}: {tl:.4f}{mark}")
    secs = time.time() - t0
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model_params.pt"), map_location=device))
    model.eval()
    samples = [{"question": q, "answer": answer(model, enc, q, device)} for q in QUESTIONS]
    tps = [h[2] for h in history["train"][1:]]
    datasets = list(info.get("datasets") or []) + [DATASET]
    extra = {**{k: v for k, v in info.items() if k != "model"},
             "datasets": datasets, "chat": True, "chat_format": CHAT_FORMAT, "base_model": base_id or base,
             "base_tokens": info.get("tokens"), "base_best_val": info.get("best_val"),
             "sft": {"dataset": DATASET, "steps": steps, "batch_size": batch_size, "lr": lr, "min_lr": min_lr,
                     "warmup": warmup, "epochs": round(steps * conv / n_windows, 3), "windows": n_windows,
                     "replay": replay, "replay_rows": loader.n_replay,
                     "base_test_loss": round(base_loss, 4), "best_test_loss": round(best, 4), "history": history,
                     "samples": samples, "seconds": round(secs), "tokens_per_sec": round(sum(tps) / len(tps)) if tps else None},
             "best_val": round(best, 4), "steps": steps, "updates": steps, "seconds": round(secs),
             "trained_at": time.strftime("%Y-%m-%d %H:%M"), "device": device, "dtype": dev.dtype,
             "chip": torch.cuda.get_device_name(0) if device == "cuda" else info.get("chip"),
             "platform": "Colab" if os.path.exists("/content") else "local"}
    save_model_config(os.path.join(out_dir, "model_config.json"), config, extra)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump({"kind": "sft", "base": base_id or base, "best_test_loss": best, "base_test_loss": base_loss,
                   "seconds": round(secs), "samples": samples, "history": history}, f, indent=1)
    say(f"Done in {secs / 60:.1f} min: held-out answer loss {base_loss:.4f} -> {best:.4f}. Saved in {out_dir}")
    for x in samples:
        say(f"\nUser: {x['question']}\nAssistant: {x['answer']}")
    return extra


def repeated_4grams(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def compare(models, out_path=None, data_dir=DATA_DIR, say=print):
    """Base vs chat model(s): held-out answer loss, the same questions answered greedily (repetition without DRY,
    and the answers with DRY), and whether each answer ends by itself."""
    dev = setup_device()
    enc = get_tokenizer()
    ctx = dev.ctx if dev.device == "cuda" else nullcontext()
    results = {}
    loader = SFTLoader(data_dir, dev.device, 8)
    tests = loader.test_batches(40)
    for mid in models:
        w, cp = resolve_base(mid)
        config, info = load_model_config(cp)
        model = load_model(config, w, dev.device)
        loss = test_loss(model, tests, ctx)
        rows, reps, ended = [], [], 0
        for q in EVAL_QUESTIONS:
            ids = format_prompt([{"role": "user", "content": q}], enc)
            plain = [ev["id"] for ev in generate_stream(model, ids, dev.device, 160, 0.0, None, stop_token=enc.eot_token)]
            ended += len(plain) < 160
            reps.append(repeated_4grams(plain))
            rows.append({"question": q, "greedy": enc.decode(plain).strip(),
                         "greedy_dry": answer(model, enc, q, dev.device)})
        results[mid] = {"chat": bool(info.get("chat")), "answer_loss": round(loss, 4),
                        "greedy_repeated_4grams": round(sum(reps) / len(reps), 3),
                        "ended_by_itself": f"{ended}/{len(EVAL_QUESTIONS)}", "answers": rows}
        say(f"{mid}: held-out answer loss {loss:.4f}, greedy repetition {results[mid]['greedy_repeated_4grams']:.1%}, "
            f"ended by itself {ended}/{len(EVAL_QUESTIONS)}")
        del model
    if out_path:
        json.dump({"measured_at": time.strftime("%Y-%m-%d %H:%M"), "format": CHAT_FORMAT, "results": results},
                  open(out_path, "w"), indent=1)
        from infra.results import record_file     # J5: the results store
        record_file(out_path, say=say)
    return results


def main():
    ap = argparse.ArgumentParser(description="Instruction tuning (SFT) on smol-smoltalk.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--limit", type=int, default=None, help="only the first N training conversations")
    r = sub.add_parser("prepare-replay", help="pre-training text for --replay (the base model's own datasets)")
    r.add_argument("--tokens", type=int, default=32_000_000)
    t = sub.add_parser("train")
    t.add_argument("--base", required=True, help="library model id, checkpoint folder or weights file")
    t.add_argument("--out", default=None)
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--min-lr", type=float, default=None, help="default: lr / 10")
    t.add_argument("--replay", type=float, default=0.0, help="share of each batch that is pre-training text, e.g. 0.2")
    t.add_argument("--no-facts", action="store_true", help="skip the facts check at each evaluation")
    t.add_argument("--eval-every", type=int, default=500)
    k = sub.add_parser("compare", help="base vs chat model: answer loss, the same questions, repetition")
    k.add_argument("models", nargs="+", help="library ids or checkpoint folders")
    k.add_argument("--out", default=None)
    c = sub.add_parser("chat")
    c.add_argument("folder")
    c.add_argument("question")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.limit)
    elif a.cmd == "prepare-replay":
        prepare_replay(a.tokens)
    elif a.cmd == "train":
        out = a.out or os.path.join(PROJECT_DIR, "outputs", "sft", time.strftime("%Y%m%d-%H%M"))
        train(a.base, out, a.steps, a.epochs, a.batch_size, a.lr, a.min_lr or a.lr / 10, eval_every=a.eval_every,
              base_id=a.base, replay=a.replay, check_facts=not a.no_facts)
    elif a.cmd == "compare":
        compare(a.models, a.out)
    else:
        w, cp = resolve_base(a.folder)
        config, _ = load_model_config(cp)
        dev = setup_device()
        print(answer(load_model(config, w, dev.device), get_tokenizer(), a.question, dev.device))


if __name__ == "__main__":
    main()
