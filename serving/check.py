# S7: check that the served model is the model. Two checks, on a GGUF file made by serving/gguf.sh:
#
#   engine  llama.cpp computes the numbers, our code scores them: an adapter that looks like our GPT to the existing
#           evaluation code (config.block_size, model(x) → logits) but runs the GGUF through llama-cpp-python, so
#           evaluate.heldout_loss and benchmarks.run_task run unchanged. Compared with the scores stored for the model.
#           (llama-server can't do this: it returns log probabilities only for tokens it generates, not for given
#           text, and the benchmarks need the probability of given answers.)
#   api     llama-server over HTTP, as a client would use it: LAMBADA by greedy generation on /v1/completions (our
#           harness counts a prompt right when greedy decoding writes the last word), and for chat models the 100
#           held-out questions of `dpo.py eval` on /v1/chat/completions (the chat template, the stop at
#           <|endoftext|>), next to our own PyTorch model on the same questions.
#
#   .venv/bin/python -m serving.check engine exports/gguf/fineweb-edu-124m-base-bf16.gguf --model <id> [--limit N]
#   .venv/bin/python -m serving.check api exports/gguf/fineweb-edu-124m-chat-dpo-bf16.gguf --model <id> [--limit N]
#
# Results go to outputs/s7/<gguf name>.json (and the results store as kind "served") unless --limit is given.
import argparse
import contextlib
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "outputs", "s7")


# ---------------------------------------------------------------- engine: llama.cpp behind our model's interface
class LlamaCppModel:
    """Looks like our GPT to evaluate.py and benchmarks.py: model(x) returns (logits for every position, None)."""

    def __init__(self, path, n_ctx=512):
        from llama_cpp import Llama
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_batch=n_ctx, logits_all=True, n_gpu_layers=-1, verbose=False)
        self.config = SimpleNamespace(block_size=n_ctx)

    def __call__(self, x, targets=None, all_logits=True):
        rows = []
        for r in x.tolist():                    # one sequence at a time; right padding only affects masked positions
            self.llm.reset()
            self.llm.eval(r)
            rows.append(torch.from_numpy(np.array(self.llm.scores[:len(r)], dtype=np.float32)))
        return torch.stack(rows), None

    def eval(self):
        return self

    def close(self):
        self.llm.close()                         # explicitly: freeing it while Python shuts down crashes Metal


def stored(model_id):
    d = os.path.join(ROOT, "models", model_id)
    load = lambda f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else {}  # noqa: E731
    return load("eval.json"), load("benchmarks.json")


def engine(gguf, model_id, limit=None, say=print):
    import benchmarks
    import evaluate
    from tokenization import get_tokenizer
    ev, bench = stored(model_id)
    m = LlamaCppModel(gguf)
    out = {"heldout": {}, "benchmarks": {}}
    try:
        datasets = [d for d in evaluate.eval_datasets() if d["key"] in (ev.get("heldout") or {})]
        for d in datasets:
            n = min(limit, evaluate.EVAL_WINDOWS) if limit else evaluate.EVAL_WINDOWS
            r = evaluate.heldout_loss(m, d["path"], "cpu", n_windows=n, batch=4)
            ref = ev["heldout"][d["key"]]["loss"]
            out["heldout"][d["key"]] = {"loss": r["loss"], "stored": ref, "diff": round(r["loss"] - ref, 4), "title": d["title"]}
            say(f"  held-out {d['title']}: {r['loss']:.4f} (stored {ref:.4f}, {r['loss'] - ref:+.4f})")
        enc = get_tokenizer()
        for t in benchmarks.TASKS:
            r = benchmarks.run_task(m, enc, "cpu", t, limit, batch=8)
            metric = benchmarks.TASK_INFO[t]["metric"]
            ref = (bench.get("results") or {}).get(t, {}).get(metric)
            out["benchmarks"][t] = {"metric": metric, "score": r[metric], "stored": ref, "stderr": r["stderr"], "n": r["n"],
                                    "diff": round(r[metric] - ref, 4) if ref is not None else None, "seconds": r["seconds"]}
            say(f"  {t}: {r[metric]:.4f} (stored {ref:.4f}, {100 * (r[metric] - ref):+.1f} points, ±{100 * r['stderr']:.1f})"
                if ref is not None else f"  {t}: {r[metric]:.4f}")
    finally:
        m.close()
    scores = [b["score"] for b in out["benchmarks"].values()]
    refs = [b["stored"] for b in out["benchmarks"].values() if b["stored"] is not None]
    if scores and len(refs) == len(scores):
        out["average"] = {"score": round(100 * sum(scores) / len(scores), 2), "stored": round(100 * sum(refs) / len(refs), 2)}
        say(f"  benchmark average {out['average']['score']:.2f} (stored {out['average']['stored']:.2f})")
    return out


# ---------------------------------------------------------------- api: llama-server over HTTP
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def llama_server(gguf, n_ctx=512):
    port = free_port()
    proc = subprocess.Popen(["llama-server", "-m", gguf, "--port", str(port), "--jinja", "-c", str(n_ctx)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if b"ok" in urllib.request.urlopen(url + "/health", timeout=1).read():
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("llama-server didn't start")
        yield url
    finally:
        proc.terminate()
        proc.wait()


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def lambada_api(url, limit=None, say=print):
    """Greedy continuation of each LAMBADA passage, as many tokens as the real last word has: right when it's the word."""
    import benchmarks
    from tokenization import get_tokenizer
    enc = get_tokenizer()
    items = benchmarks.load_task("lambada", limit)
    right = 0
    t0 = time.time()
    for k, it in enumerate(items):
        ctx, word = it["context"], it["choices"][0]
        ctx_ids, word_ids = enc.encode_ordinary(ctx), enc.encode_ordinary(word)
        r = post(url, "/v1/completions", {"prompt": ctx_ids[-(512 - len(word_ids)):], "max_tokens": len(word_ids),
                                          "temperature": 0, "logprobs": 1})
        got = [c["id"] for c in (r["choices"][0].get("logprobs") or {}).get("content") or []]   # none: it stopped at once
        right += got == word_ids
        if k and k % 1000 == 0:
            say(f"    {k:,} of {len(items):,}")
    return {"acc": right / len(items), "n": len(items), "seconds": round(time.time() - t0, 1)}


def chat_questions(n=100):
    """The held-out questions of `dpo.py eval` (the same shuffle, the same first n)."""
    import dpo
    from tokenization import get_tokenizer
    pairs = dpo.load_pairs(dpo.PAIRS, get_tokenizer())    # dpo.py eval's default file (it reproduces the stored numbers)
    random.Random(0).shuffle(pairs)
    return [(prompt, p["question"]) for prompt, _, _, p in pairs[:n]]


def answer_stats(answers, cap):
    from sft import repeated_4grams
    return {"repeated_4grams": round(sum(repeated_4grams(a) for a in answers) / len(answers), 4),
            "ended": round(sum(len(a) < cap for a in answers) / len(answers), 3),
            "mean_length": round(sum(map(len, answers)) / len(answers), 1), "n": len(answers)}


def chat_api(url, model_id, limit=None, say=print):
    """The same questions through /v1/chat/completions (greedy, up to 200 tokens) and through our PyTorch model."""
    import dpo
    from inference import generate_stream, load_model, load_model_config
    from train_config import setup_device
    qs = chat_questions(limit or 100)
    served, same_prompt = [], 0
    for prompt, question in qs:
        r = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": question}], "temperature": 0,
                                               "max_tokens": dpo.MAX_ANSWER, "logprobs": True})
        content = (r["choices"][0].get("logprobs") or {}).get("content") or []
        served.append([c["id"] for c in content if c["id"] != 50256])
        same_prompt += r["usage"]["prompt_tokens"] == len(prompt)
    dev = setup_device().device
    d = os.path.join(ROOT, "models", model_id)
    cfg, _ = load_model_config(os.path.join(d, "model_config.json"))
    model = load_model(cfg, os.path.join(d, "best_model_params.pt"), dev)
    ours = [[ev["id"] for ev in generate_stream(model, prompt, dev, dpo.MAX_ANSWER, 0.0, None, stop_token=50256)]
            for prompt, _ in qs]
    ref = json.load(open(os.path.join(d, "dpo_eval.json"))).get("results", {}).get(model_id) \
        if os.path.exists(os.path.join(d, "dpo_eval.json")) else None
    out = {"served": answer_stats(served, dpo.MAX_ANSWER), "ours": answer_stats(ours, dpo.MAX_ANSWER),
           "stored": {k: ref[k] for k in ("repeated_4grams", "ended", "mean_length")} if ref else None,
           "identical_answers": sum(a == b for a, b in zip(served, ours)), "prompt_tokens_match": same_prompt,
           "sample": {"question": qs[0][1], "served": None}}
    from tokenization import get_tokenizer
    out["sample"]["served"] = get_tokenizer().decode(served[0])
    s, o = out["served"], out["ours"]
    say(f"  chat, {len(qs)} held-out questions: repetition {s['repeated_4grams']:.1%} (ours {o['repeated_4grams']:.1%}), "
        f"ended {s['ended']:.0%} ({o['ended']:.0%}), length {s['mean_length']} ({o['mean_length']}); "
        f"identical answers {out['identical_answers']}/{len(qs)}, prompt tokens match {same_prompt}/{len(qs)}")
    return out


def api(gguf, model_id, limit=None, say=print):
    _, bench = stored(model_id)
    chat = json.load(open(os.path.join(ROOT, "models", model_id, "model_config.json"))).get("chat")
    out = {}
    with llama_server(gguf) as url:
        r = lambada_api(url, limit, say)
        ref = (bench.get("results") or {}).get("lambada", {}).get("acc")
        out["lambada"] = {**r, "stored": ref, "diff": round(r["acc"] - ref, 4) if ref is not None else None}
        say(f"  LAMBADA through /v1/completions: {r['acc']:.4f}" + (f" (stored {ref:.4f})" if ref is not None else ""))
        if chat:
            out["chat"] = chat_api(url, model_id, limit, say)
    return out


# ---------------------------------------------------------------- saving
def save(gguf, model_id, part, result):
    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(gguf))[0]
    path = os.path.join(OUT_DIR, f"{name}.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({"gguf": os.path.relpath(gguf, ROOT), "model": model_id, part: result,
                f"{part}_measured_at": time.strftime("%Y-%m-%d %H:%M")})
    json.dump(old, open(path, "w"), indent=1)
    try:                                          # J5: the results store, as kind "served"
        from infra import results
        recs = [{"kind": "served", "model": model_id, "key": f"{name}:{part}", "numbers": result,
                 "settings": {"gguf": old["gguf"], "engine": "llama.cpp"}, "measured_at": old[f"{part}_measured_at"],
                 "from": os.path.relpath(path, ROOT)}]
        results.add(recs)
    except Exception as e:  # noqa: BLE001
        print(f"(not added to the results store: {e})")
    print(f"saved {os.path.relpath(path, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="S7: check that the served model is the model.")
    ap.add_argument("part", choices=["engine", "api"])
    ap.add_argument("gguf")
    ap.add_argument("--model", required=True, help="the library model the GGUF was exported from")
    ap.add_argument("--limit", type=int, help="a quick check on the first N questions (saves nothing)")
    a = ap.parse_args()
    print(f"{a.part}: {os.path.basename(a.gguf)} against {a.model}" + (f" (first {a.limit})" if a.limit else ""))
    result = (engine if a.part == "engine" else api)(a.gguf, a.model, a.limit)
    if not a.limit:
        save(a.gguf, a.model, a.part, result)


if __name__ == "__main__":
    main()
