# Local web front end for training the model step by step.
# Run from the project folder:  python3 app/server.py   (then open http://127.0.0.1:8000)
#
# The server wraps the same modules full-model.py uses. Long-running work (downloading,
# tokenizing, training) runs in a background thread; the page polls /api/status for progress.

import argparse
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict
from typing import Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from catalog import DATASET_CATALOG  # noqa: E402
from data import (DEFAULT_DATASET, BatchLoader, SubsetNeeded, dataset_paths, get_dataset,  # noqa: E402
                  inspect_source, resolve_source)
from inference import generate_text, load_model, load_model_config, save_model_config  # noqa: E402
from model import GPT, GPTConfig, param_breakdown  # noqa: E402
from pretrain import pretrain  # noqa: E402
from tokenization import bin_files_exist, get_tokenizer, tokenize_and_save  # noqa: E402
from train_config import TrainConfig, setup_device  # noqa: E402

DATA_ROOT = os.path.join(PROJECT_DIR, "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def chip_name():
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.dev = setup_device()
        self.enc = get_tokenizer()
        self.chip = chip_name()
        self.dataset = None            # dict: name, config_name, text_column, paths, ds, was_local
        self.batch = {"block_size": 128, "batch_size": 32, "seed": 42}
        self.model_config = None
        self.model = None
        self.job = {"kind": None, "status": "idle", "message": "", "progress": None, "error": None}
        self.train = None              # live training progress
        self.stop_requested = False
        self.inference_cache = None    # (checkpoint path, mtime, model)


S = State()
app = FastAPI(title="TinyStories LM Studio")


# ---------- helpers ----------

def start_job(kind, fn):
    with S.lock:
        if S.job["status"] == "running":
            raise HTTPException(409, f"Please wait: {S.job['kind']} is still running.")
        S.job = {"kind": kind, "status": "running", "message": "Starting…", "progress": None,
                 "error": None, "started": time.time()}

    def run():
        try:
            fn()
            if S.job["status"] == "running":
                S.job.update(status="done", progress=1.0)
        except Exception as e:  # surface any failure to the page
            traceback.print_exc()
            S.job.update(status="error", error=str(e), message=str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


def require_idle(action):
    if S.job["status"] == "running":
        raise HTTPException(409, f"Can't {action} while {S.job['kind']} is running.")


def require_dataset():
    if S.dataset is None:
        raise HTTPException(400, "Load a dataset first (step 1).")
    return S.dataset


def require_tokenized():
    d = require_dataset()
    if not bin_files_exist(d["paths"].root):
        raise HTTPException(400, "Tokenize the dataset first (step 2).")
    return d


def tokens_of(ids):
    return [{"id": int(i), "text": S.enc.decode([int(i)])} for i in ids]


def checkpoint_info(paths):
    if not (os.path.exists(paths.checkpoint) and os.path.exists(paths.model_config)):
        return None
    try:
        cfg, info = load_model_config(paths.model_config)
    except Exception:
        return None
    return {"model": asdict(cfg), "params": param_breakdown(cfg), "steps": info.get("steps"),
            "best_val": info.get("best_val"), "trained_at": info.get("trained_at"),
            "stopped_early": info.get("stopped_early", False)}


def dataset_info():
    d = S.dataset
    if not d:
        return None
    p = d["paths"]
    tokenized = bin_files_exist(p.root)
    return {
        "name": d["name"], "config_name": d["config_name"], "text_column": d["text_column"],
        "path": os.path.relpath(p.root, PROJECT_DIR), "was_local": d["was_local"],
        "splits": {k: len(v) for k, v in d["ds"].items()},
        "tokenized": tokenized,
        "tokens": {"train": os.path.getsize(p.train_bin) // 2, "validation": os.path.getsize(p.val_bin) // 2}
        if tokenized else None,
        "checkpoint": checkpoint_info(p),
    }


# ---------- request bodies ----------

class DatasetReq(BaseModel):
    name: str = DEFAULT_DATASET
    config_name: Optional[str] = None
    text_column: str = "text"


class TextReq(BaseModel):
    text: str


class BatchReq(BaseModel):
    block_size: int = 128
    batch_size: int = 32
    seed: int = 42


class ModelReq(BaseModel):
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


class TrainReq(BaseModel):
    max_iters: int = 2000
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 200
    eval_interval: int = 250
    eval_iters: int = 50
    gradient_accumulation_steps: int = 8


class GenerateReq(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50


# ---------- API ----------

@app.get("/api/status")
def status():
    return {
        "device": S.dev.device, "dtype": S.dev.dtype, "chip": S.chip,
        "torch": torch.__version__, "default_dataset": DEFAULT_DATASET,
        "dataset": dataset_info(),
        "batch": S.batch,
        "model": {"config": asdict(S.model_config), "params": sum(p.numel() for p in S.model.parameters()),
                  "breakdown": param_breakdown(S.model_config)} if S.model is not None else None,
        "job": S.job,
        "train": S.train,
    }


@app.get("/api/catalog")
def catalog():
    """Public text datasets known to work, with whether each is already downloaded, tokenized or trained."""
    out = []
    for item in DATASET_CATALOG:
        p = dataset_paths(item["dataset"], item["config"], DATA_ROOT)
        out.append({**item, "downloaded": os.path.isdir(p.hf_dir), "tokenized": bin_files_exist(p.root),
                    "trained": checkpoint_info(p) is not None})
    return out


def _clean_request(req):
    name = req.name.strip() or DEFAULT_DATASET
    config_name = (req.config_name or "").strip() or None
    text_column = req.text_column.strip() or "text"
    try:
        src = resolve_source(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if src["kind"] == "url" and src["builder"] == "text":
        text_column = "text"  # plain text files have a single 'text' column
    return name, config_name, text_column


@app.post("/api/dataset/check")
def check_dataset(req: DatasetReq):
    """Validate a dataset (name or link) without downloading it."""
    name, config_name, text_column = _clean_request(req)
    try:
        info = inspect_source(name, config_name, text_column)
    except SubsetNeeded as e:
        return {"ok": False, "error": str(e), "configs": e.configs}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    p = dataset_paths(name, config_name, DATA_ROOT)
    return {"ok": True, **info, "downloaded": os.path.isdir(p.hf_dir), "path": os.path.relpath(p.root, PROJECT_DIR)}


@app.post("/api/dataset")
def load_dataset_endpoint(req: DatasetReq):
    name, config_name, text_column = _clean_request(req)
    paths = dataset_paths(name, config_name, DATA_ROOT)

    def job():
        was_local = os.path.isdir(paths.hf_dir)
        S.job["message"] = "Loading the local copy…" if was_local else f"Downloading {name} (first time only)…"
        ds = get_dataset(paths.hf_dir, name, config_name=config_name, text_column=text_column)
        changed = S.dataset is None or S.dataset["paths"].root != paths.root
        S.dataset = {"name": name, "config_name": config_name, "text_column": text_column,
                     "paths": paths, "ds": ds, "was_local": was_local}
        if changed:  # a new dataset invalidates anything built for the old one
            S.model, S.model_config, S.train, S.inference_cache = None, None, None, None
        S.job["message"] = "Dataset ready."

    return start_job("dataset", job)


@app.get("/api/dataset/sample")
def dataset_sample(split: str = "train", index: int = -1):
    d = require_dataset()
    if split not in d["ds"]:
        raise HTTPException(400, f"Unknown split '{split}'.")
    total = len(d["ds"][split])
    if index < 0 or index >= total:
        index = random.randrange(total)
    return {"split": split, "index": index, "total": total, "text": d["ds"][split][index]["text"]}


@app.post("/api/tokenize/preview")
def tokenize_preview(req: TextReq):
    ids = S.enc.encode_ordinary(req.text[:5000])
    return {"chars": len(req.text), "tokens": tokens_of(ids)}


@app.post("/api/tokenize")
def tokenize_endpoint():
    d = require_dataset()
    if bin_files_exist(d["paths"].root):
        raise HTTPException(400, "This dataset is already tokenized.")

    def job():
        S.job["message"] = "Tokenizing every document. This is the slow part…"

        def progress(split, done, total):
            S.job.update(message=f"Writing {split}.bin", progress=done / total)

        tokenize_and_save(d["ds"], d["paths"].root, num_proc=os.cpu_count() or 4, progress=progress)
        S.job["message"] = "Tokenized."

    return start_job("tokenize", job)


@app.post("/api/batch")
def batch_endpoint(req: BatchReq):
    require_idle("sample a batch")
    d = require_tokenized()
    if not (8 <= req.block_size <= 1024 and 1 <= req.batch_size <= 512):
        raise HTTPException(400, "block_size must be 8–1024 and batch_size 1–512.")
    for f in (d["paths"].train_bin, d["paths"].val_bin):
        n_tokens = os.path.getsize(f) // 2
        if n_tokens <= req.block_size + 1:
            raise HTTPException(400, f"{os.path.basename(f)} has only {n_tokens:,} tokens, fewer than one "
                                     f"{req.block_size}-token window. Use a smaller context window or a bigger dataset.")
    if S.model_config is not None and S.model_config.block_size != req.block_size:
        S.model, S.model_config = None, None  # model was built for another context size
    S.batch = req.model_dump()
    torch.manual_seed(req.seed)
    loader = BatchLoader(d["paths"].root, req.block_size, req.batch_size, "cpu")
    X, Y = loader.get_batch("train")
    n = min(24, req.block_size)
    return {"shape": list(X.shape), "input": tokens_of(X[0, :n].tolist()), "target": tokens_of(Y[0, :n].tolist())}


@app.post("/api/model")
def model_endpoint(req: ModelReq):
    require_idle("build a model")
    if req.n_embd % req.n_head != 0:
        raise HTTPException(400, f"n_embd ({req.n_embd}) must be divisible by n_head ({req.n_head}).")
    if not (1 <= req.n_layer <= 48 and 1 <= req.n_head <= 32 and 16 <= req.n_embd <= 2048 and 0 <= req.dropout < 1):
        raise HTTPException(400, "Model settings are out of range.")
    config = GPTConfig(vocab_size=S.enc.n_vocab, block_size=S.batch["block_size"], n_layer=req.n_layer,
                       n_head=req.n_head, n_embd=req.n_embd, dropout=req.dropout, bias=True)
    torch.manual_seed(S.batch["seed"])
    model = GPT(config).to(S.dev.device)
    S.model, S.model_config, S.train = model, config, None

    result = {"params": sum(p.numel() for p in model.parameters()), "config": asdict(config),
              "breakdown": param_breakdown(config), "random_loss": math.log(S.enc.n_vocab), "initial_loss": None}
    model.eval()
    if S.dataset and bin_files_exist(S.dataset["paths"].root):
        loader = BatchLoader(S.dataset["paths"].root, config.block_size, S.batch["batch_size"], S.dev.device)
        X, Y = loader.get_batch("train")
        with torch.no_grad():
            _, loss = model(X, Y)
        result["initial_loss"] = loss.item()
    result["sample"] = generate_text(model, S.enc, "Once upon a time", S.dev.device, max_new_tokens=30)
    model.train()
    return result


@app.post("/api/train")
def train_endpoint(req: TrainReq):
    d = require_tokenized()
    if S.model is None:
        raise HTTPException(400, "Build the model first (step 4).")
    if not (1 <= req.max_iters <= 1_000_000) or not (0 <= req.warmup_steps < req.max_iters):
        raise HTTPException(400, "Need max_iters ≥ 1 and warmup_steps below max_iters.")
    if req.eval_interval < 1 or req.eval_iters < 1 or req.gradient_accumulation_steps < 1:
        raise HTTPException(400, "eval_interval, eval_iters and gradient_accumulation_steps must be ≥ 1.")
    if not (0 < req.min_lr <= req.learning_rate):
        raise HTTPException(400, "min_lr must be positive and no higher than learning_rate.")

    paths = d["paths"]
    cfg = TrainConfig(batch_size=S.batch["batch_size"], seed=S.batch["seed"],
                      best_model_path=paths.checkpoint, **req.model_dump())
    model, config = S.model, S.model_config

    def job():
        S.stop_requested = False
        for f in (paths.checkpoint, paths.model_config):  # replace any earlier model for this dataset
            if os.path.exists(f):
                os.remove(f)
        S.inference_cache = None
        tr = S.train = {"max_iters": cfg.max_iters, "step": 0, "lr": None, "points": [], "evals": [],
                        "best_val": None, "started": time.time(), "it_per_sec": None, "config": asdict(cfg)}
        stride = max(1, cfg.max_iters // 1000)
        t0 = time.time()

        def callback(ev):
            if ev["type"] == "step":
                step = ev["step"]
                tr["step"], tr["lr"] = step, ev["lr"]
                if step == 1 or step % stride == 0:
                    tr["points"].append([step, round(ev["loss"], 4)])
                tr["it_per_sec"] = step / max(time.time() - t0, 1e-6)
                S.job.update(progress=step / cfg.max_iters, message=f"Step {step:,} of {cfg.max_iters:,}")
            else:
                tr["evals"].append({"step": ev["step"], "train": round(ev["train"], 4), "val": round(ev["val"], 4)})
                if tr["best_val"] is None or ev["val"] < tr["best_val"]:
                    tr["best_val"] = round(ev["val"], 4)
            return not S.stop_requested

        S.job["message"] = "Training…"
        pretrain(model, BatchLoader(paths.root, config.block_size, cfg.batch_size, S.dev.device), cfg, S.dev, callback)

        if not os.path.exists(paths.checkpoint):  # stopped (or finished) before the first evaluation
            torch.save(model.state_dict(), paths.checkpoint)
        save_model_config(paths.model_config, config, {
            "dataset": d["name"], "dataset_config": d["config_name"], "steps": tr["step"],
            "best_val": tr["best_val"], "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "stopped_early": S.stop_requested, "train_config": asdict(cfg)})
        if S.stop_requested:
            S.job.update(status="stopped", message=f"Stopped at step {tr['step']:,}. Model saved.")
        else:
            S.job["message"] = "Training complete. Model saved."

    return start_job("train", job)


@app.post("/api/train/stop")
def train_stop():
    if S.job["kind"] == "train" and S.job["status"] == "running":
        S.stop_requested = True
        S.job["message"] = "Stopping after this step…"
    return {"ok": True}


@app.post("/api/generate")
def generate_endpoint(req: GenerateReq):
    d = require_dataset()
    if S.job["kind"] == "train" and S.job["status"] == "running":
        raise HTTPException(409, "Training is running. Wait for it to finish, or stop it.")
    paths = d["paths"]
    if checkpoint_info(paths) is None:
        raise HTTPException(400, "No trained model for this dataset yet. Train one in step 5.")

    mtime = os.path.getmtime(paths.checkpoint)
    if not (S.inference_cache and S.inference_cache[0] == paths.checkpoint and S.inference_cache[1] == mtime):
        config, _ = load_model_config(paths.model_config)
        S.inference_cache = (paths.checkpoint, mtime, load_model(config, paths.checkpoint, S.dev.device))
    model = S.inference_cache[2]

    prompt = req.prompt[:2000]
    t0 = time.time()
    text = generate_text(model, S.enc, prompt, S.dev.device,
                         max_new_tokens=max(1, min(req.max_new_tokens, 1000)),
                         temperature=max(0.05, min(req.temperature, 3.0)),
                         top_k=req.top_k if req.top_k > 0 else None)
    completion = text[len(prompt):] if text.startswith(prompt) else text
    return {"prompt": prompt, "completion": completion, "seconds": round(time.time() - t0, 2)}


# ---------- static front end ----------

@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """Make browsers recheck the page and its files, so a `git pull` shows up on the next reload."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Web front end for training the model step by step.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}"
    print(f"\n  TinyStories LM Studio → {url}\n  Device: {S.dev.device} ({S.dev.dtype}) on {S.chip}\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
