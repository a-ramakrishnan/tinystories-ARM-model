# J6: evaluate a model end to end on the Mac with one command, picking up where it stopped.
#
#   python3 -m infra.eval_queue <target> [<target> ...]       run every task not yet done
#   python3 -m infra.eval_queue <target> --plan               list the tasks and what's already done
#   python3 -m infra.eval_queue <target> --quick              a few questions per task, nothing saved (a check that
#                                                             everything runs before starting the long version)
#   python3 -m infra.eval_queue <target> --only quality,bench:lambada --redo
#
# A target is a library model id (an autoregressive model in models/) or a diffusion run folder
# (diffusion_models/runs/<run>, or just <run>). The tasks:
#   autoregressive  suite (held-out losses, generation, size: evaluate.py), then the five benchmarks
#   diffusion       bound (the training's best likelihood bound, into the results store), quality (every sampler
#                   setting in diffusion_models/quality.py, as for earlier runs), then the five benchmarks
# Benchmarks run one task at a time, the quick ones first and HellaSwag (2 h of the ~3 h for a 124M diffusion model)
# last; each saves its own result as it finishes.
#
# Resumable: <target folder>/eval_queue.json keeps each task's state (done, failed, how long, batch size). A task
# that finished is skipped next time (--redo runs it again), so after a crash or a closed lid the same command
# carries on with the next task.
#
# Memory: the batch size of the scoring passes is chosen from the GPU memory left once the model is loaded (on the
# Mac, Metal's recommended working-set size). If a task still runs out of memory, it is retried with half the batch,
# down to 2. On macOS the queue runs under `caffeinate`, so the Mac doesn't sleep half-way through.
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STATE = "eval_queue.json"
BENCH_ORDER = ["lambada", "arc_easy", "piqa", "winogrande", "hellaswag"]
QUICK_LIMIT = 20                 # questions per benchmark with --quick
MIN_BATCH, MAX_BATCH = 2, 64
MEMORY_SHARE = 0.7               # share of the free GPU memory the scoring passes may plan to use
LOGIT_COPIES = 2.5               # logits (B × L × vocab, float32) plus the log-sum-exp temporaries


# ---------- targets ----------

def resolve(target):
    """{kind: "ar" | "diffusion", name, dir, ...} for a model id or a diffusion run folder."""
    import registry
    for d in (target, os.path.join(ROOT, "diffusion_models", "runs", target)):
        if os.path.isfile(os.path.join(d, "model.pt")) and os.path.isfile(os.path.join(d, "config.json")):
            d = os.path.abspath(d)
            return {"kind": "diffusion", "name": os.path.basename(d), "dir": d}
    try:
        meta = registry.get_model(target)
    except Exception:
        meta = None
    if not meta:
        raise SystemExit(f"{target}: neither a library model id nor a diffusion run folder")
    return {"kind": "ar", "name": target, "dir": registry.model_dir(target), "title": meta.get("name", target)}


def task_names(kind):
    first = ["suite"] if kind == "ar" else ["bound", "quality"]
    return first + ["bench:" + t for t in BENCH_ORDER]


# ---------- state ----------

def load_state(folder):
    try:
        with open(os.path.join(folder, STATE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"tasks": {}}


def save_state(folder, state):
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = os.path.join(folder, STATE + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, os.path.join(folder, STATE))


# ---------- memory ----------

def free_gpu_bytes(device):
    """Bytes the GPU can still take, or None when unknown (CPU)."""
    import torch
    if device == "mps":
        return torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        return free
    return None


def pick_batch(free_bytes, seq_len, vocab, default):
    """Largest power of two whose logits (rows × seq_len × vocab floats, a few copies) fit in MEMORY_SHARE of the
    free memory, between MIN_BATCH and MAX_BATCH; `default` when the free memory is unknown."""
    if free_bytes is None:
        return default
    per_row = seq_len * vocab * 4 * LOGIT_COPIES
    fit = int(free_bytes * MEMORY_SHARE // per_row)
    b = MIN_BATCH
    while b * 2 <= min(fit, MAX_BATCH):
        b *= 2
    return b


def is_oom(e):
    m = str(e).lower()
    return "out of memory" in m or "mps backend out of memory" in m or "failed to allocate" in m


def empty_cache(device):
    import gc
    import torch
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


# ---------- the tasks ----------
# Each task is fn(target, ctx) -> a one-line summary; ctx: device, batch (None = choose), quick, say.

def task_suite(t, ctx):
    import evaluate
    if ctx["quick"]:
        return "skipped with --quick (it takes a few minutes and always saves)"
    r = evaluate.evaluate_model(t["name"], ctx["device"], progress=ctx["say"])
    fw = next((v["loss"] for k, v in r["heldout"].items() if "fineweb" in k), None)
    return f"held-out Mini FineWeb-Edu loss {fw}, greedy repetition {r['generation']['greedy_repeated_4grams']}"


def task_bound(t, ctx):
    path = os.path.join(t["dir"], "metrics.json")
    with open(path) as f:
        m = json.load(f)
    if not ctx["quick"]:
        from infra.results import record_file
        record_file(path, "diffusion:" + t["name"], say=ctx["say"])
    val = m.get("history", {}).get("val") or []
    last = f", last {val[-1][2]:.4f} at update {val[-1][0]:,}" if val else ""
    return f"best likelihood bound {m['best_bound']:.4f}{last} ({m.get('tokens', 0) / 1e9:.2f}B tokens)"


def task_quality(t, ctx):
    from diffusion_models import quality
    if ctx["quick"]:
        cfgs = dict(list(quality.CONFIGS.items())[:1])
        out = quality.evaluate(t["dir"], cfgs, per_source=1, seeds=(1,), say=ctx["say"], reference=False)
    else:
        out = quality.evaluate(t["dir"], quality.CONFIGS, say=ctx["say"])
        quality.save(t["dir"], out)
    best = out["results"].get("keep, rep 1.2, blocks of 16") or next(iter(out["results"].values()))
    s = best["summary"]
    return f"{len(out['results'])} settings; e.g. judge loss {s['judge_loss']}, repeated 4-grams {s['repeated_4grams']}"


def task_bench(name):
    def run(t, ctx):
        limit = QUICK_LIMIT if ctx["quick"] else None
        if t["kind"] == "ar":
            import benchmarks as B
            import registry
            from inference import load_model_config
            cfg, _ = load_model_config(registry.paths_of(t["name"])[1])
            free = free_gpu_bytes(ctx["device"])      # the model isn't loaded yet: leave room for it
            if free is not None:
                free -= 4 * sum_params(cfg)
            b = ctx["batch"] or pick_batch(free, cfg.block_size, cfg.vocab_size, 16)
            ctx["say"](f"batch size {b}" + (f" ({free / 1e9:.1f} GB of GPU memory to plan with)" if free else ""))
            ctx["last_batch"] = b
            r = B.run(t["name"], [name], limit, ctx["device"], save=not limit, batch=b)[name]
        else:
            from diffusion_models import benchmarks as DB
            with open(os.path.join(t["dir"], "config.json")) as f:
                mc = json.load(f)["model"]
            free = free_gpu_bytes(ctx["device"])
            if free is not None:                      # the model isn't loaded yet: leave room for it
                try:
                    with open(os.path.join(t["dir"], "metrics.json")) as f:
                        free -= 4 * json.load(f)["params"]
                except (OSError, ValueError, KeyError):
                    free -= 4 * 130_000_000
            b = ctx["batch"] or pick_batch(free, mc["seq_len"], mc["vocab_size"], 64)
            ctx["say"](f"batch size {b}" + (f" ({free / 1e9:.1f} GB of GPU memory to plan with)" if free else ""))
            ctx["last_batch"] = b
            r = DB.run(t["dir"], [name], limit, save=not limit, device=ctx["device"], batch=b)[name]
        main = "acc" if name in ("lambada", "arc_easy", "piqa", "winogrande") else "acc_norm"
        return f"{main} {100 * r[main]:.1f}% on {r['n']:,} questions in {r['seconds'] / 60:.1f} min"
    return run


def sum_params(cfg):
    d, L, V = cfg.n_embd, cfg.n_layer, cfg.vocab_size
    return V * d + L * 12 * d * d


def task_fn(name):
    if name.startswith("bench:"):
        return task_bench(name.split(":", 1)[1])
    return {"suite": task_suite, "bound": task_bound, "quality": task_quality}[name]


# ---------- the queue ----------

def run_target(t, tasks, device, batch=None, quick=False, redo=False, say=print, fns=None):
    """Run `tasks` for one target in order, skipping those already done. Returns {task: status}."""
    state = load_state(t["dir"])
    state.update({"target": t["name"], "kind": t["kind"]})
    out = {}
    for name in tasks:
        rec = state["tasks"].get(name) or {}
        if rec.get("status") == "done" and not redo and not quick:
            say(f"[{name}] done {rec.get('finished_at')}: {rec.get('summary')} (skipped; --redo runs it again)")
            out[name] = "skipped"
            continue
        fn = (fns or {}).get(name) or task_fn(name)
        b = batch
        while True:
            ctx = {"device": device, "batch": b, "quick": quick, "say": lambda m, n=name: say(f"[{n}] {m}")}
            say(f"[{name}] starting{' (quick check)' if quick else ''}")
            if not quick:
                state["tasks"][name] = {**rec, "status": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_state(t["dir"], state)
            t0 = time.time()
            try:
                summary = fn(t, ctx)
                status = "done"
            except KeyboardInterrupt:
                if not quick:
                    state["tasks"][name] = {**rec, "status": "stopped"}
                    save_state(t["dir"], state)
                raise
            except Exception as e:
                empty_cache(device)
                used = ctx.get("last_batch") or b
                if is_oom(e) and (used or MAX_BATCH) > MIN_BATCH:
                    b = max(MIN_BATCH, (used or MAX_BATCH * 2) // 2)
                    say(f"[{name}] out of GPU memory; retrying with batch size {b}")
                    continue
                summary, status = f"{type(e).__name__}: {e}", "failed"
                say(traceback.format_exc().rstrip())
            break
        secs = round(time.time() - t0, 1)
        say(f"[{name}] {status} in {secs / 60:.1f} min: {summary}")
        out[name] = status
        if not quick:
            state["tasks"][name] = {"status": status, "summary": summary, "seconds": secs,
                                    "finished_at": time.strftime("%Y-%m-%d %H:%M"), "device": device,
                                    **({"batch": ctx.get("last_batch") or b} if name.startswith("bench:") else {})}
            save_state(t["dir"], state)
        empty_cache(device)
    return out


def plan(t, tasks):
    state = load_state(t["dir"])
    lines = [f"{t['name']} ({'autoregressive' if t['kind'] == 'ar' else 'diffusion'}; state in "
             f"{os.path.relpath(os.path.join(t['dir'], STATE), ROOT)})"]
    for name in tasks:
        r = state["tasks"].get(name) or {}
        st = r.get("status", "to do")
        extra = f"  {r.get('finished_at', '')}  {r.get('summary', '')}" if r else ""
        lines.append(f"  {name:20s} {st:8s}{extra}")
    return "\n".join(lines)


def keep_awake():
    """On macOS, stop the Mac sleeping while this process runs."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        return subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate models end to end (J6): losses or bound, quality, benchmarks.")
    ap.add_argument("targets", nargs="+", help="library model ids or diffusion run folders")
    ap.add_argument("--only", default=None, help="comma-separated tasks (suite, bound, quality, bench, bench:<task>)")
    ap.add_argument("--redo", action="store_true", help="run tasks again even if done")
    ap.add_argument("--quick", action="store_true", help=f"{QUICK_LIMIT} questions per benchmark, a little generation; nothing saved")
    ap.add_argument("--batch", type=int, default=None, help="fix the scoring batch size instead of choosing it")
    ap.add_argument("--plan", action="store_true", help="only list the tasks and their state")
    a = ap.parse_args(argv)

    targets = [resolve(x) for x in a.targets]
    chosen = {}
    for t in targets:
        names = task_names(t["kind"])
        if a.only:
            want = [w.strip() for w in a.only.split(",") if w.strip()]
            names = [n for n in names if n in want or (n.startswith("bench:") and "bench" in want)]
        chosen[t["name"]] = names
    if a.plan:
        for t in targets:
            print(plan(t, chosen[t["name"]]))
        return 0

    from train_config import setup_device
    device = setup_device().device
    awake = keep_awake()
    print(f"Evaluation queue on {device}{' (caffeinate: the Mac stays awake)' if awake else ''}", flush=True)
    failed = 0
    try:
        for t in targets:
            print(plan(t, chosen[t["name"]]), flush=True)
            res = run_target(t, chosen[t["name"]], device, a.batch, a.quick, a.redo,
                             say=lambda m: print(m, flush=True))
            failed += sum(s == "failed" for s in res.values())
    finally:
        if awake:
            awake.terminate()
    if not a.quick:
        for t in targets:
            print(plan(t, chosen[t["name"]]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
